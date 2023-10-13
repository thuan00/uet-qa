import json
import argparse
from tqdm import tqdm
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever

from uetqa.retriever import ESSentenceTransformersRetriever, HybridRetriever
from uetqa.util import compute_recall_mrr_at_k



parser = argparse.ArgumentParser()
parser.add_argument('--testset', type=str, default=None,
                    help='')
parser.add_argument('--index_name', type=str, default='document',
                    help='name of the index in elasticsearch')
parser.add_argument('--analyzer', type=str, default='standard',
                    help='')
parser.add_argument('--similarity', type=str, default='cosine',
                    help='')
parser.add_argument('--dense_model_path', type=str, default=None,
                    help='')
parser.add_argument('--max_seq_len', type=int, default=256,
                    help='')
parser.add_argument('--embedding_dim', type=int, default=768,
                    help='')
parser.add_argument('--batch_size', type=int, default=32,
                    help='')
parser.add_argument('--k', type=int, default=10,
                    help='')
args = parser.parse_args()

with open(args.testset, mode='r', encoding='utf-8') as f:
    qas = json.load(f)



# Init
#
document_store = ElasticsearchDocumentStore(
    index=args.index_name,
    analyzer=args.analyzer,
    embedding_dim=args.embedding_dim,
    similarity=args.similarity,
)
sparse_retriever = BM25Retriever(
    document_store,
    top_k=args.k,
)
dense_retriever = ESSentenceTransformersRetriever(
    document_store=document_store,
    top_k=args.k,
    embedding_model=args.dense_model_path,
    max_seq_len=args.max_seq_len,
    batch_size=args.batch_size,
    progress_bar=False,
)
retriever = HybridRetriever(
    dense_retriever,
    sparse_retriever,
    weight_on_dense=True,
    normalization=False,
)



# Retrieve
#
queries = [ qa['question'] for qa in qas ]
queries_ground_truth_context_ids = [ qa['ground_truth_context_ids'] for qa in qas ]

retriever.find_best_weight(
    queries,
    queries_ground_truth_context_ids,
    k=args.k,
    weights=[0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 8, 16, 9999],
)
breakpoint() # stop to consider possible weights

queries_results = []
for q in tqdm(queries):
    result = retriever.retrieve(q, top_k=args.k)
    queries_results.append(result)

queries_retrieved_context_ids = [ [doc.meta['id'] for doc in query_results] for query_results in queries_results ]



# Evaluate
#
for k in range(1,args.k+1):
    recall, mrr, failures = compute_recall_mrr_at_k(
        queries,
        queries_retrieved_context_ids,
        queries_ground_truth_context_ids,
        k
    )
    print(f'{k},{round(recall,3)},{round(mrr,3)}')
    # print(failures)
