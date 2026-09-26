"""HotpotQA metrics, ported from the official hotpot_evaluate_v1.py
(https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).

Behaviour is identical to the official script: answer EM/F1 on normalized strings (lowercase, no
punctuation or articles, collapsed whitespace; yes/no/noanswer only match exactly), supporting
fact (title, sentence index) precision/recall/F1/EM, and joint metrics (products of answer and
supporting-fact precision and recall; joint EM = answer EM * sp EM). Averages are over all gold
questions.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """(f1, precision, recall), official semantics."""
    pred, gold = normalize_answer(prediction), normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)
    if pred in ("yes", "no", "noanswer") and pred != gold:
        return zero
    if gold in ("yes", "no", "noanswer") and pred != gold:
        return zero
    pred_tokens, gold_tokens = pred.split(), gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return zero
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall), precision, recall


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def sp_scores(
    prediction: Iterable[tuple[str, int]], gold: Iterable[tuple[str, int]]
) -> tuple[float, float, float, float]:
    """(em, f1, precision, recall) over supporting-fact (title, sentence) pairs."""
    pred, true = set(map(tuple, prediction)), set(map(tuple, gold))
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, f1, precision, recall


def score_example(
    answer: str, sp: Iterable[tuple[str, int]], gold_answer: str, gold_sp: Iterable[tuple[str, int]]
) -> dict[str, float]:
    """All official per-question metrics for one prediction."""
    em = exact_match(answer, gold_answer)
    f1, prec, recall = f1_score(answer, gold_answer)
    sp_em, sp_f1, sp_prec, sp_recall = sp_scores(sp, gold_sp)
    joint_prec, joint_recall = prec * sp_prec, recall * sp_recall
    joint_f1 = (
        2 * joint_prec * joint_recall / (joint_prec + joint_recall)
        if joint_prec + joint_recall > 0
        else 0.0
    )
    return {
        "em": em,
        "f1": f1,
        "prec": prec,
        "recall": recall,
        "sp_em": sp_em,
        "sp_f1": sp_f1,
        "sp_prec": sp_prec,
        "sp_recall": sp_recall,
        "joint_em": em * sp_em,
        "joint_f1": joint_f1,
        "joint_prec": joint_prec,
        "joint_recall": joint_recall,
    }
