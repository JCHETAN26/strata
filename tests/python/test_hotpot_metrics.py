"""rag/hotpot_metrics.py must reproduce the official hotpot_evaluate_v1.py.

Expected values below were produced by running the official script's functions on these inputs
(including its quirks, e.g. 'the the the' vs 'a' is an exact match with F1 0).
"""

from __future__ import annotations

import pytest
from hotpot_metrics import exact_match, f1_score, normalize_answer, score_example, sp_scores

OFFICIAL = [
    # prediction, gold, EM, (F1, precision, recall)
    ("The Eiffel Tower", "eiffel tower", True, (1.0, 1.0, 1.0)),
    ("yes", "Yes.", True, (1.0, 1.0, 1.0)),
    ("yes", "no", False, (0.0, 0.0, 0.0)),
    ("no answer", "noanswer", False, (0.0, 0.0, 0.0)),
    ("Paris, France", "Paris", False, (2 / 3, 0.5, 1.0)),
    ("1,000 people", "1000", False, (2 / 3, 0.5, 1.0)),
    ("an apple a day", "apple day", True, (1.0, 1.0, 1.0)),
    ("Barack Obama", "Obama", False, (2 / 3, 0.5, 1.0)),
    ("", "Obama", False, (0.0, 0.0, 0.0)),
    ("noanswer", "noanswer", True, (1.0, 1.0, 1.0)),
    ("the the the", "a", True, (0.0, 0.0, 0.0)),
]


@pytest.mark.parametrize(("prediction", "gold", "em", "f1"), OFFICIAL)
def test_answer_metrics_match_official(prediction: str, gold: str, em: bool, f1: tuple) -> None:
    assert exact_match(prediction, gold) == float(em)
    assert f1_score(prediction, gold) == pytest.approx(f1)


def test_normalization() -> None:
    assert normalize_answer("  The  U.S.  Army! ") == "us army"


def test_supporting_fact_metrics_match_official() -> None:
    em, f1, prec, recall = sp_scores([("A", 0), ("B", 1), ("C", 2)], [("A", 0), ("B", 1), ("B", 2)])
    assert (em, f1, prec, recall) == pytest.approx((0.0, 2 / 3, 2 / 3, 2 / 3))
    assert sp_scores([], [("A", 0)]) == (0.0, 0.0, 0.0, 0.0)
    assert sp_scores([("A", 0)], [("A", 0)]) == (1.0, 1.0, 1.0, 1.0)


def test_joint_metrics() -> None:
    s = score_example("Obama", [("A", 0)], "Barack Obama", [("A", 0), ("B", 1)])
    # answer: p=1, r=0.5; sp: p=1, r=0.5 -> joint p=1, r=0.25, f1=0.4; joint em = 0
    assert s["joint_prec"] == 1.0 and s["joint_recall"] == 0.25
    assert s["joint_f1"] == pytest.approx(0.4)
    assert s["joint_em"] == 0.0
