"""Reranking: pool construction, ordering, pinned models, and the evaluation (no model download)."""

from __future__ import annotations

import numpy as np
import pytest
from rerank import RERANKERS, Reranker, rerank, union_pool


def test_union_pool_round_robin_dedup() -> None:
    bm25 = ["a", "b", "c", "d"]
    dense = ["c", "e", "a", "f"]
    assert union_pool([bm25, dense], 1) == ["a", "c"]
    assert union_pool([bm25, dense], 2) == ["a", "c", "b", "e"]
    assert union_pool([bm25, dense], 4) == ["a", "c", "b", "e", "d", "f"]
    assert union_pool([bm25, []], 3) == ["a", "b", "c"]  # a short list is fine
    # Deeper pools contain shallower ones (scores for the deepest pool can be reused).
    assert set(union_pool([bm25, dense], 2)) <= set(union_pool([bm25, dense], 3))


def test_rerank_sorts_by_score_and_is_stable() -> None:
    pool = ["a", "b", "c", "d"]
    scores = {"a": 0.1, "b": 0.9, "c": 0.9, "d": -1.0}
    assert rerank(pool, scores, 4) == ["b", "c", "a", "d"]  # b before c: tie keeps pool order
    assert rerank(pool, scores, 2) == ["b", "c"]
    assert rerank([], {}, 5) == []


class FakeCrossEncoder:
    """Scores a pair by word overlap; records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        self.calls.append(pairs)
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


def test_reranker_scores_pairs_with_injected_model() -> None:
    model = FakeCrossEncoder()
    reranker = Reranker("minilm-l6", model=model)
    scores, seconds = reranker.score("red apple pie", ["apple pie", "blue sky", "red apple"])
    np.testing.assert_array_equal(scores, [2, 0, 2])
    assert seconds >= 0 and model.calls[0][0] == ("red apple pie", "apple pie")
    assert reranker.score("q", []) == (pytest.approx(np.zeros(0)), 0.0)


def test_models_are_pinned() -> None:
    for spec in RERANKERS.values():
        assert len(spec.revision) == 40 and spec.max_length == 512
