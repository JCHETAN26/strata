from __future__ import annotations

import numpy as np
import pytest
from significance import compare_all, holm, paired_bootstrap_ci, randomization_test


def test_identical_methods_are_not_different() -> None:
    a = np.random.default_rng(0).uniform(size=200)
    lo, hi = paired_bootstrap_ci(a, a)
    assert lo == hi == 0.0
    assert randomization_test(a, a) == 1.0


def test_clear_difference_is_detected() -> None:
    rng = np.random.default_rng(1)
    b = rng.uniform(size=300)
    a = b + 0.05 + rng.normal(0, 0.02, size=300)  # a better on almost every query
    lo, hi = paired_bootstrap_ci(a, b)
    assert 0.04 < lo < 0.05 < hi < 0.06
    assert randomization_test(a, b) < 0.001


def test_no_true_difference_rarely_rejected() -> None:
    # Under H0, p < 0.05 should happen about 5% of the time.
    rejections = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        b = rng.uniform(size=100)
        a = b + rng.normal(0, 0.1, size=100)
        rejections += randomization_test(a, b, n_permutations=2000, seed=seed) < 0.05
    assert rejections / 200 < 0.10


def test_ci_is_paired_not_independent() -> None:
    # Large between-query variance but a tiny consistent difference: a paired interval is narrow.
    rng = np.random.default_rng(2)
    b = rng.uniform(0, 1, size=300)
    a = b + 0.01
    lo, hi = paired_bootstrap_ci(a, b)
    assert lo == pytest.approx(0.01) and hi == pytest.approx(0.01)


def test_holm_adjustment() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == [1.0, 1.0]
    assert holm([]) == []


def test_compare_all_pairs_and_counts() -> None:
    per_query = {
        "x": {"q1": 1.0, "q2": 0.5, "q3": 0.0},
        "y": {"q1": 0.5, "q2": 0.5, "q3": 0.5},
        "z": {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 1.0},  # q4 not shared: ignored
    }
    results = compare_all(per_query, n_resamples=500)
    assert [(r.a, r.b) for r in results] == [("x", "y"), ("x", "z"), ("y", "z")]
    xy = results[0]
    assert (xy.queries, xy.wins, xy.losses, xy.ties) == (3, 1, 1, 1)
    assert xy.mean_diff == pytest.approx(0.0)
    assert all(r.p_holm >= r.p_value for r in results)


def test_compare_all_with_a_planned_family() -> None:
    per_query = {
        "base": {"q1": 0.0, "q2": 0.5, "q3": 1.0},
        "a": {"q1": 1.0, "q2": 0.5, "q3": 1.0},
        "b": {"q1": 0.5, "q2": 0.5, "q3": 1.0},
        "unused": {"q1": 0.0},  # not in the family: does not shrink the shared query set
    }
    results = compare_all(per_query, pairs=[("a", "base"), ("a", "b")], n_resamples=200)
    assert [(r.a, r.b) for r in results] == [("a", "base"), ("a", "b")]
    assert all(r.queries == 3 for r in results)
    assert results[0].mean_diff == pytest.approx(1 / 3)
