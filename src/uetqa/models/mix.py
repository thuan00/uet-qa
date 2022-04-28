from dataclasses import dataclass
from typing import Optional, Tuple, Union

import logging
import torch
import numpy as np
from transformers import AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class MIXOutput(QuestionAnsweringModelOutput):
    cls_logits: torch.FloatTensor = None


class MIX(PreTrainedModel):
    """"""
    def __init__(self, config):
        super().__init__(config)

        qa_model = AutoModelForQuestionAnswering.from_pretrained(config._name_or_path, config=config)

        encoder_attr_name = self.get_encoder_attr_name(qa_model)
        qa_model_encoder = getattr(qa_model, encoder_attr_name)
        qa_model_outputs = getattr(qa_model, "qa_outputs")

        setattr(self, encoder_attr_name, qa_model_encoder)
        self.encoder_attr_name = encoder_attr_name
        self.qa_outputs = qa_model_outputs
        self.cls_output = torch.nn.Linear(config.hidden_size, 1)
        # self.alpha = 2.0

        self.need_token_type = encoder_attr_name not in {
            "xlm", "roberta", "distilbert", "camembert", "bart", "longformer"
        }

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute, for pretrained weights init
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Deberta"):
            return "deberta"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("XLMRoberta"):
            return "roberta"
        elif model_class_name.startswith("RemBert"):
            return "rember"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def _init_weights(self, module):
        # print(module)
        pass

    @classmethod
    def _load_state_dict_into_model(
        cls, model, state_dict, pretrained_model_name_or_path, ignore_mismatched_sizes=False, _fast_init=True
    ):
        if 'cls_output.weight' in state_dict:
            model_state_dict = model.state_dict()
            model_state_dict['cls_output.weight'] = state_dict['cls_output.weight']
            model_state_dict['cls_output.bias'] = state_dict['cls_output.bias']
            logger.warning("\nLoaded pretrained cls_output layer\n")
        else:
            model.cls_output.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            model.cls_output.bias.data.zero_()
            logger.warning("\nInitialized new cls_output layer\n")
        del state_dict
        return model, [],[],[],[]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, MIXOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        """
        base_model = getattr(self, self.encoder_attr_name)

        outputs = base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state

        ans_logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = ans_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        cls_token_state = last_hidden_state[:,0]
        cls_logits = self.cls_output(cls_token_state).squeeze(-1).contiguous()

        # Cal loss
        total_loss = 0.0
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            total_loss += loss_fct(cls_logits, labels) #*self.alpha

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # hard_negative mask, ignore the QA task for negative samples
            hard_negatives_mask = (labels == 0)*99999
            start_positions += hard_negatives_mask
            end_positions += hard_negatives_mask

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1) # max_seq_len
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss += (start_loss + end_loss) / 2

        # breakpoint()
        return MIXOutput(
            loss=total_loss if total_loss > 0 else None,
            start_logits=start_logits,
            end_logits=end_logits,
            cls_logits=cls_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def process_query(self, tokenizer, query, documents, top_k_per_doc=3):
        """ """
        input_queries = [query] * len(documents)
        passages = [ doc['content'] for doc in documents]

        # Tokenize
        inputs = tokenizer(
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
        ).to(self.device)
        n_examples = inputs.input_ids.shape[0]

        with torch.no_grad():
            if self.need_token_type:
                output = self.forward(
                    inputs.input_ids,
                    inputs.attention_mask,
                    token_type_ids=inputs.token_type_ids,
                )
            else:
                output = self.forward(
                    inputs.input_ids,
                    inputs.attention_mask,
                )
            cls_scores = output.cls_logits.tolist()

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
