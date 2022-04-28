import string
from collections import Counter


def white_space_fix(text):
        return ' '.join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    """Given top_n predictions and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        for prediction in predictions:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_em_f1(ground_truths, predictions):
    total = len(ground_truths)
    f1 = exact_match = 0
    for qid, ground_truth in ground_truths.items():
        if qid not in predictions:
            message = "Unanswered question " + qid + " will receive score 0."
            print(message)
            continue
        prediction = predictions[qid]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truth)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truth)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


def find_rank(retrieved_context_ids, ground_truth_context_ids):
    """ return best_rank if retrieval succeed, else return 0
    """
    success = False
    best_rank = float('infinity')

    for gt in ground_truth_context_ids:
        try:
            rank = retrieved_context_ids.index(gt) + 1
            if rank < best_rank:
                best_rank = rank
                success = True
        except:
            pass

    return best_rank if success else 0

def compute_recall_mrr_at_k(queries, queries_retrieved_context_ids, queries_ground_truth_context_ids, k):
    mrr = 0.0
    success_count = 0
    failures = []

    for i in range(len(queries)):

        retrieved_context_ids = queries_retrieved_context_ids[i][:k]
        ground_truth_context_ids = queries_ground_truth_context_ids[i]

        rank = find_rank(retrieved_context_ids, ground_truth_context_ids)

        if rank:
            success_count += 1
            mrr += 1/rank
        else:
            failures.append(i)

    return success_count/len(queries), mrr/len(queries), failures
