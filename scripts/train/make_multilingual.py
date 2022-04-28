"""
This script contains an example how to extend an existent sentence embedding model to new languages.

Given a (monolingual) teacher model you would like to extend to new languages, which is specified in the teacher_model_name
variable. We train a multilingual student model to imitate the teacher model (variable student_model_name)
on multiple languages.

For training, you need parallel sentence data (machine translation training data). You need tab-seperated files (.tsv)
with the first column a sentence in a language understood by the teacher model, e.g. English,
and the further columns contain the according translations for languages you want to extend to.

This scripts downloads automatically the TED2020 corpus: https://github.com/UKPLab/sentence-transformers/blob/master/docs/datasets/TED2020.md
This corpus contains transcripts from
TED and TEDx talks, translated to 100+ languages. For other parallel data, see get_parallel_data_[].py scripts

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://arxiv.org/abs/2004.09813
"""

from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
from sentence_transformers.datasets import ParallelSentencesDataset
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm.autonotebook import tqdm

import os
import logging
import sentence_transformers.util
import csv
import numpy as np
import zipfile
import io
import json
import argparse

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='output/', help='')

parser.add_argument('--train_file', type=str, required=True, help='')
parser.add_argument('--dev_file', type=str, default=None, help='')
parser.add_argument('--dev_ir_queries_file', type=str, default=None, help='')
parser.add_argument('--dev_ir_corpus_file', type=str, default=None, help='')

parser.add_argument('--teacher_model_name', type=str, required=True, help='Our monolingual teacher model, we want to convert to multiple languages')
parser.add_argument('--student_model_name', type=str, required=True, help='Multilingual base model we use to imitate the teacher model')

parser.add_argument('--max_seq_length', type=int, default=512, help='Student model max. lengths for inputs tokens')
parser.add_argument('--train_batch_size', type=int, default=8, help='')
parser.add_argument('--inference_batch_size', type=int, default=8, help='')
parser.add_argument('--max_sentences_per_language', type=int, default=1000000, help='Maximum number of parallel sentences for training')
parser.add_argument('--train_max_sentence_length', type=int, default=None, help='Maximum length (characters) for parallel training sentences, longer sentence will be dropped')

parser.add_argument('--lr', type=float, default=1e-5, help='')
parser.add_argument('--weight_decay', type=float, default=0.001, help='')
parser.add_argument('--use_amp', action='store_true', help='')
parser.add_argument('--num_warmup_steps', type=int, default=1000, help='')
parser.add_argument('--num_epochs', type=int, default=1, help='')
parser.add_argument('--num_evaluation_steps', type=int, default=1000, help='Evaluate performance after every n steps')

args = parser.parse_args()

# msmarco-MiniLM-L12-cos-v5 || multi-qa-MiniLM-L6-dot-v1 -> paraphrase-multilingual-MiniLM-L12-v2
# msmarco-distilbert-dot-v5 -> sentence-transformers/paraphrase-xlm-r-multilingual-v1
teacher_model_name = args.teacher_model_name
student_model_name = args.student_model_name

max_seq_length = args.max_seq_length
train_batch_size = args.train_batch_size
inference_batch_size = args.inference_batch_size
max_sentences_per_language = args.max_sentences_per_language
train_max_sentence_length = args.train_max_sentence_length

num_epochs = args.num_epochs
num_warmup_steps = args.num_warmup_steps
num_evaluation_steps = args.num_evaluation_steps
lr = args.lr
weight_decay = args.weight_decay
use_amp= args.use_amp

train_file = args.train_file
dev_file = args.dev_file
dev_ir_queries_file = args.dev_ir_queries_file
dev_ir_corpus_file = args.dev_ir_corpus_file

output_path = args.output_path


######## Start the extension of the teacher model to multiple languages ########
logger.info("Load teacher")
teacher_model = SentenceTransformer(teacher_model_name)

logger.info("Load student")
word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



###### Read Parallel Sentences Dataset ######
logger.info("Load train dataset")
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=True)
train_data.load_data(train_file, max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)


#### Evaluate cross-lingual performance on different tasks #####
evaluators = []         #evaluators has a list of different evaluator classes we call periodically

if (dev_file):
    logger.info("Create evaluator for " + dev_file)
    src_sentences = []
    trg_sentences = []
    with open(dev_file, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            splits = line.strip().split('\t')
            if splits[0] != "" and splits[1] != "" and max(len(splits[0]),len(splits[1])) < train_max_sentence_length:
                src_sentences.append(splits[0])
                trg_sentences.append(splits[1])
    logger.info(f"Num dev pairs: {len(src_sentences)}")

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    dev_mse = evaluation.MSEEvaluator(
        src_sentences,
        trg_sentences,
        name=os.path.basename(dev_file),
        teacher_model=teacher_model,
        batch_size=inference_batch_size
    )
    evaluators.append(dev_mse)

# Retrieval Evaluator
if (dev_ir_queries_file):
    logger.info("Create evaluator IR")

    read_jsonl = lambda file: [json.loads(i) for i in file.read().splitlines()]

    with open(dev_ir_queries_file, mode='r', encoding='utf-8') as f:
        qas = read_jsonl(f)

    with open(dev_ir_corpus_file, mode='r', encoding='utf-8') as f:
        contexts = read_jsonl(f)

    corpus = { context['id']:context['contents'] for context in contexts }
    queries = { qa['qid']:qa['question'] for qa in qas }
    relevant_docs = { qa['qid']:[qa['context_id']] for qa in qas }

    dev_ir = evaluation.InformationRetrievalEvaluator(
        queries,
        corpus,
        relevant_docs,
        batch_size=inference_batch_size,
        mrr_at_k = [10],
        ndcg_at_k = [10],
        accuracy_at_k = [10],
        precision_recall_at_k = [10],
    )
    evaluators.append(dev_ir)
breakpoint()

# Train the model
student_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=num_warmup_steps,
    evaluator=evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: np.mean(scores)),
    evaluation_steps=num_evaluation_steps,
    output_path=output_path,
    save_best_model=True,
    optimizer_params={'lr': lr, 'eps': 1e-6, 'correct_bias': False},
    weight_decay=weight_decay,
    use_amp=use_amp,
)
