import json
import argparse
import time
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import ElasticsearchRetriever, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline

from uetqa.util import compute_em_f1



parser = argparse.ArgumentParser()
parser.add_argument('--testset', type=str, default=None,
                    help='')
parser.add_argument('--index_name', type=str, default='document',
                    help='name of the index in elasticsearch')
parser.add_argument('--analyzer', type=str, default='standard',
                    help='')
parser.add_argument('--retriever_k', type=int, default=10,
                    help='')
parser.add_argument('--model_path', type=str, default=None,
                    help='')
parser.add_argument('--reader_k', type=int, default=5,
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
retriever = ElasticsearchRetriever(
    document_store,
    top_k=args.retriever_k,
)
reader = TransformersReader(
    args.model_path,
    max_seq_len=384,
    top_k=args.reader_k,
    top_k_per_candidate=3,
    return_no_answers=False,
)
pipe = ExtractiveQAPipeline(reader, retriever)



# Infer
#
predictions = {}
start_time = time.time()

for qa in qas:
    answers = pipe.run(
        query=qa['question'],
    )
    predictions[qa['id']] = [ ans.answer for ans in answers ]

print("time (s): ", time.time() - start_time)



# Evaluate
#
labels = { qa['id']:qa['answers'] for qa in qas }

print('k, em, f1')
for k in range(1,args.reader_k+1):

    predictions_at_k = { qid:answers[:k] for qid, answers in predictions.items() }

    result = compute_em_f1(labels, predictions_at_k)
    
    print('\n', round(result['exact_match'],4), round(result['f1'],4))
