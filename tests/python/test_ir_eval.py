"""bench/ir_eval.py must agree with trec_eval (via pytrec_eval) per query, including ties."""

from __future__ import annotations

import random

import pytest
from ir_eval import mean, ndcg_at_k, ranked, recall_at_k, round_scores

pytrec_eval = pytest.importorskip("pytrec_eval")


def random_case(seed: int) -> tuple[dict, dict]:
    rng = random.Random(seed)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q in range(40):
        qid = f"q{q}"
        docs = [f"d{rng.randrange(200)}" for _ in range(rng.randrange(0, 60))]
        # Few distinct scores, so many ties.
        run[qid] = {d: float(rng.choice([0.5, 1.0, 1.5, 2.0, 2.25])) for d in docs}
        judged = {f"d{rng.randrange(200)}": rng.choice([0, 1, 1, 2, 3]) for _ in range(12)}
        qrels[qid] = judged
    return run, qrels


@pytest.mark.parametrize("seed", range(5))
def test_matches_trec_eval_per_query(seed: int) -> None:
    run, qrels = random_case(seed)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
    reference = evaluator.evaluate(run)
    ndcg = ndcg_at_k(run, qrels, 10)
    recall = recall_at_k(run, qrels, 100)
    for qid, values in reference.items():
        assert ndcg[qid] == pytest.approx(values["ndcg_cut_10"], abs=1e-12), qid
        assert recall[qid] == pytest.approx(values["recall_100"], abs=1e-12), qid


def test_ties_break_by_docid_descending() -> None:
    assert ranked({"d1": 1.0, "d3": 1.0, "d2": 0.5}) == ["d3", "d1", "d2"]


def test_trec_eval_c_semantics() -> None:
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}}
    run = {"q1": {"a": 1.0}}  # q2 has no results, q3 has no relevant documents
    per_query = ndcg_at_k(run, qrels, 10)
    assert per_query == {"q1": 1.0, "q2": 0.0, "q3": 0.0}
    assert mean(per_query) == pytest.approx(1 / 3)


def test_round_scores_matches_percent_f() -> None:
    assert round_scores({"q": {"d": 1.23456789}})["q"]["d"] == float(f"{1.23456789:f}")
