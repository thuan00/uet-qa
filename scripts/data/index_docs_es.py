import argparse
import os
import json
from glob import glob
from haystack.document_stores import ElasticsearchDocumentStore
from uetqa.retriever import ESSentenceTransformersRetriever


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default=None,
                    help='')
parser.add_argument('--index_name', type=str, default='document',
                    help='name of the index in elasticsearch')
parser.add_argument('--analyzer', type=str, default='standard',
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
args = parser.parse_args()


files = glob(os.path.abspath(args.input_dir)+'/*.json')

docs = []
for file in files:
    with open(file) as f:
        docs.extend(json.load(f))


document_store = ElasticsearchDocumentStore(
    index=args.index_name,
    analyzer=args.analyzer,
    embedding_dim=args.embedding_dim,
)
document_store.write_documents(docs)


if args.dense_model_path:
    retriever = ESSentenceTransformersRetriever(
        document_store=document_store,
        embedding_model=args.dense_model_path,
        pooling_strategy=args.pooling_strategy,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )
    document_store.update_embeddings(retriever)
