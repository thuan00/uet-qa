import json
import argparse
import time
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import ElasticsearchRetriever

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
parser.add_argument('--pooling_strategy', type=str, default='reduce_mean',
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
sparse_retriever = ElasticsearchRetriever(
    document_store,
    top_k=args.k,
)
dense_retriever = ESSentenceTransformersRetriever(
    document_store=document_store,
    top_k=args.k,
    embedding_model=args.dense_model_path,
    pooling_strategy=args.pooling_strategy,
    max_seq_len=args.max_seq_len,
    batch_size=args.batch_size,
    progress_bar=False,
)
retriever = HybridRetriever(
    sparse_retriever,
    dense_retriever,
    weight_on_dense=True,
    normalization=True,
)



# Retrieve
#
queries = [ qa['question'] for qa in qas ]
queries_ground_truth_context_ids = [ qa['ground_truth_context_ids'] for qa in qas ]

retriever.find_best_weight(queries, queries_ground_truth_context_ids)
breakpoint() # stop to consider possible weights

queries_results = []
start_time = time.time()

for q in queries:
    result = retriever.retrieve(q, top_k=args.k)
    queries_results.append(result)

print("time (s): ", time.time() - start_time)

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
