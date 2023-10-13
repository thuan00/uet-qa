from typing import List
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from haystack.schema import Document
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import ElasticsearchDocumentStore

from uetqa.util import compute_recall_mrr_at_k


class ESSentenceTransformersRetriever(EmbeddingRetriever):
    """
    """
    def __init__(
        self,
        document_store: ElasticsearchDocumentStore,
        embedding_model: str,
        **kwargs,
    ):
        kwargs['model_format'] = 'sentence_transformers'
        super().__init__(embedding_model, document_store, **kwargs)

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """ Override haystack logic, and offset score to ensure positive score as required by Elasticsearch
        """
        document_store = self.document_store
        query_emb = self.embed_queries([query])[0].tolist() # np.ndarray.tolist
        similarity_fn = 'dotProduct' if document_store.similarity != 'cosine' else 'cosineSimilarity'
        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": similarity_fn+"(params.query_vector,'embedding')+100", #
                        "lang": "painless",
                        "params": {
                            "query_vector": query_emb
                        }
                    },
                    "boost": 1.0
                }
            },
            "_source": {
                "includes": [],
                "excludes": ["embedding"]
            }
        }
        result = document_store.client.search(index=document_store.index, body=body)["hits"]["hits"]
        documents = [ self._convert_es_hit_to_document(hit) for hit in result ]
        return documents

    def _convert_es_hit_to_document(self, hit):
        data = hit['_source']
        doc = {'meta': {}, 'score': hit['_score']-100, 'content': data['content']}
        doc['meta'] = data
        del data['content'], data['content_type']
        return Document.from_dict(doc)

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """ override haystack's document embedding method, which puts doc behind the <sep> token
        """
        return self.embedding_encoder.embed([ d.content for d in docs ])


class HybridRetriever:
    """ Hybrid  dense + sparse
    """
    def __init__(
        self,
        dense_searcher,
        sparse_searcher,
        weight: float = 0.1,
        weight_on_dense: bool = False,
        normalization: bool = False
    ):
        assert (dense_searcher.top_k and sparse_searcher.top_k) is not None, "Make sure top_k is specified"
        self.dense_searcher = dense_searcher
        self.sparse_searcher = sparse_searcher
        self.weight = weight
        self.weight_on_dense = weight_on_dense
        self.normalization = normalization

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Document]:
        """"""
        dense_hits = self.dense_searcher.retrieve(query)
        sparse_hits = self.sparse_searcher.retrieve(query)
        return self._hybrid_results(dense_hits, sparse_hits, top_k)

    def _hybrid_results(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
        k: int,
    ) -> List[Document]:
        """"""
        dense_hits = {hit.id: hit.score for hit in dense_results}
        sparse_hits = {hit.id: hit.score for hit in sparse_results}
        doc_map = {doc.id: doc for doc in sparse_results + dense_results}
        hybrid_result = []

        min_dense_score = min(dense_hits.values())
        max_dense_score = max(dense_hits.values())
        dense_avg_score = (min_dense_score + max_dense_score) / 2
        dense_range = max_dense_score - min_dense_score

        min_sparse_score = min(sparse_hits.values())
        max_sparse_score = max(sparse_hits.values())
        sparse_avg_score = (min_sparse_score + max_sparse_score) / 2
        sparse_range = max_sparse_score - min_sparse_score

        for doc in doc_map.keys():
            if doc not in dense_hits:
                sparse_score = sparse_hits[doc]
                dense_score = min_dense_score
            elif doc not in sparse_hits:
                sparse_score = min_sparse_score
                dense_score = dense_hits[doc]
            else:
                sparse_score = sparse_hits[doc]
                dense_score = dense_hits[doc]

            if self.normalization:
                sparse_score = (sparse_score - sparse_avg_score) / sparse_range
                dense_score = (dense_score - dense_avg_score) / dense_range

            if self.weight_on_dense:
                score = sparse_score + self.weight * dense_score
            else:
                score = sparse_score * self.weight + dense_score

            hybrid_doc = deepcopy(doc_map[doc])
            hybrid_doc.score = score
            hybrid_result.append(hybrid_doc)

        hybrid_result = sorted(hybrid_result, key=lambda x: x.score, reverse=True)[:k]
        return hybrid_result

    def find_best_weight(
        self,
        queries: List[str],
        queries_ground_truth_context_ids: List[List[str]],
        k: int = 10,
        weights: List[float] = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
    ):
        assert k <= max(self.dense_searcher.top_k, self.sparse_searcher.top_k), "Make sure top_k is specified correctly"
        recalls = []
        mrrs = []

        dense_results = []
        sparse_results = []
        for q in tqdm(queries):
            dense_result = self.dense_searcher.retrieve(q)
            sparse_result = self.sparse_searcher.retrieve(q)
            dense_results.append(dense_result)
            sparse_results.append(sparse_result)

        for w in weights:
            self.weight = w

            queries_results = []
            for i in range(len(queries)):
                result = self._hybrid_results(dense_results[i], sparse_results[i], k)
                queries_results.append(result)

            queries_retrieved_context_ids = [ [doc.meta['id'] for doc in query_results] for query_results in queries_results ]

            recall, mrr, failures = compute_recall_mrr_at_k(
                queries,
                queries_retrieved_context_ids,
                queries_ground_truth_context_ids,
                k
            )
            recalls.append(recall)
            mrrs.append(mrr)
            print('\n', w, '\t', recall.__round__(3), '\t', mrr.__round__(3))

        best_recall_id = recalls.index(max(recalls))
        best_mrr_id = mrrs.index(max(mrrs))

        print('\nbest recall weight:', weights[best_recall_id])
        print('\nbest mrr weight:', weights[best_mrr_id])

        self.weight = weights[best_mrr_id]

