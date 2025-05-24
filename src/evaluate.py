# src/evaluate.py

from collections import Counter
import re

def normalize_answer(s):
    """Loại bỏ dấu, viết thường và chuẩn hóa khoảng trắng."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    pred_tokens = get_tokens(prediction)
    ground_tokens = get_tokens(ground_truth)
    if not pred_tokens or not ground_tokens:
        return int(pred_tokens == ground_tokens)
    common = Counter(pred_tokens) & Counter(ground_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0
    precision = 1.0 * num_common / len(pred_tokens)
    recall = 1.0 * num_common / len(ground_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_answer(prediction: str, ground_truth: str) -> dict:
    """
    Đánh giá câu trả lời dựa trên Exact Match và F1 score.
    """
    em_score = compute_exact_match(prediction, ground_truth)
    f1_score = compute_f1(prediction, ground_truth)
    return {
        "exact_match": em_score,
        "f1": f1_score
    }

def evaluate_batch(predictions: list[str], ground_truths: list[str]) -> dict:
    """
    Đánh giá một batch các câu trả lời.
    """
    total_em = 0
    total_f1 = 0
    for prediction, ground_truth in zip(predictions, ground_truths):
        scores = evaluate_answer(prediction, ground_truth)
        total_em += scores["exact_match"]
        total_f1 += scores["f1"]

    avg_em = total_em / len(predictions) if predictions else 0
    avg_f1 = total_f1 / len(predictions) if predictions else 0
    return {
        "exact_match": avg_em,
        "f1": avg_f1
    }
