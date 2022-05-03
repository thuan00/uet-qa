import logging
import torch
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)



class QABase():
    """ """
    def __init__(self, model_path):

        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        assert self.tokenizer.is_fast, "fast tokenizer only"

        self.need_token_type = self.qa_model.config.type_vocab_size > 1
        #.config.model_type not in {"xlm", "roberta", "distilbert", "camembert", "bart", "longformer", "deberta",}

    def process_query(self, query, documents, top_k_per_doc=3):
        """ """
        input_queries = [query] * len(documents)
        passages = [ doc['content'] for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            input_queries,
            passages,
            padding=True,
            truncation='only_second',
            stride=100,
            return_attention_mask=True,
            return_token_type_ids=self.need_token_type,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors='pt',
        ).to(self.qa_model.device)
        n_examples = inputs.input_ids.shape[0]

        with torch.no_grad():
            output = self.qa_model(
                inputs.input_ids,
                inputs.attention_mask,
                token_type_ids=inputs.token_type_ids if self.need_token_type else None,
            )
            cls_scores = (output.start_logits[:,0] + output.end_logits[:,0]).tolist()

        # for each example, decode start-end logits to answer span & score
        predictions = []
        for example_idx in range(n_examples):
            start, end, score, _ = self.decode_logits(
                output.start_logits[example_idx:example_idx+1],
                output.end_logits[example_idx:example_idx+1],
                topk=top_k_per_doc
            )
            for i in range(top_k_per_doc):

                passage_idx = inputs.overflow_to_sample_mapping[example_idx].item()
                start_char = inputs.offset_mapping[example_idx, start[i], 0].item()
                end_char = inputs.offset_mapping[example_idx, end[i], 1].item()

                doc = documents[passage_idx]
                answer_score = float(score[i])

                prediction = {
                    'answer': passages[passage_idx][start_char:end_char],
                    'answer_score': answer_score,
                    'start': start_char,
                    'end': end_char,
                    'doc': doc,
                    'cls_score': cls_scores[example_idx],
                }
                predictions.append(prediction)
            
        predictions.sort(key=lambda x: x['answer_score'], reverse=True)

        return predictions


    def decode_logits(self, start_logits, end_logits, topk=1, max_answer_len=None):
        """
        Take the output of :obj:`MIXOutput` and formulate answer spans with scores.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
        answer end position being before the starting position. The method supports output the k-best answer through
        the topk argument.

        Args:
            start_logits (:obj:`tensor`): Individual start logits for each token. # shape: batch, len(input_ids[0])
            end_logits (:obj:`tensor`): Individual end logits for each token.
            topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
        Output:
            starts:  top_n predicted start indices
            ends:  top_n predicted end indices
            scores:  top_n prediction scores
            idx_sort:  top_n batch element ids
        """
        start = start_logits.cpu().numpy().clip(min=0.0)
        end = end_logits.cpu().numpy().clip(min=0.0)
        max_answer_len = max_answer_len or start.shape[1]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen et al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        idx_sort, starts, ends = np.unravel_index(idx_sort, candidates.shape)
        scores = candidates[idx_sort, starts, ends]

        return starts, ends, scores, idx_sort
