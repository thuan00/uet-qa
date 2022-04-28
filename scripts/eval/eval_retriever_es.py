import json
import argparse
import time
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import ElasticsearchRetriever

from uetqa.util import compute_recall_mrr_at_k



parser = argparse.ArgumentParser()
parser.add_argument('--testset', type=str, default=None,
                    help='')
parser.add_argument('--index_name', type=str, default='document',
                    help='name of the index in elasticsearch')
parser.add_argument('--analyzer', type=str, default='standard',
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
)
retriever = ElasticsearchRetriever(document_store)



# Retrieve
#
queries = [ qa['question'] for qa in qas ]
queries_ground_truth_context_ids = [ qa['ground_truth_context_ids'] for qa in qas ]

queries_results = []
start_time = time.time()

for q in queries:
    result = retriever.retrieve(q, top_k=args.k)
    queries_results.append(result)

print("time (s): ", time.time() - start_time)

queries_retrieved_context_ids = [ [doc.meta['id'] for doc in query_results] for query_results in queries_results ]



# Evaluate
#
print('k, recall, mrr')
for k in range(1,args.k+1):
    recall, mrr, failures = compute_recall_mrr_at_k(
        queries,
        queries_retrieved_context_ids,
        queries_ground_truth_context_ids,
        k
    )
    print('\n', k, round(recall,3), round(mrr,3))
    # print(failures)
