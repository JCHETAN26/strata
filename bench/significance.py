"""Paired significance tests over queries for comparing retrieval methods.

Each method gets a per-query score (e.g. nDCG@10) on the same queries; tests work on the
per-query differences d_q = a_q - b_q.

- paired_bootstrap_ci: resample queries with replacement (the same resample for both methods),
  take the mean difference each time; the percentile interval of those means is the CI.
- randomization_test: under H0 (no difference) each d_q is equally likely to have either sign;
  flip signs at random and count how often |mean| is at least the observed |mean| (two-sided).
  The p-value includes the observed assignment ((count + 1) / (n_permutations + 1)), so it is
  never 0.
- holm: Holm-Bonferroni adjustment when several pairs are tested on the same queries.

References: Efron & Tibshirani (1993) for the bootstrap; Smucker, Allan & Carterette (CIKM 2007)
for randomization tests in IR evaluation. Deterministic for a given seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class PairComparison:
    a: str
    b: str
    mean_a: float
    mean_b: float
    mean_diff: float  # mean over queries of (a - b)
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float  # two-sided paired randomization test
    p_holm: float  # Holm-adjusted across all pairs in the family
    queries: int
    wins: int  # queries where a > b
    losses: int
    ties: int

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def randomization_test(
    a: Sequence[float], b: Sequence[float], *, n_permutations: int = 10_000, seed: int = 0
) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(d)))
    permuted = np.abs((signs * d).mean(axis=1))
    # Tolerance so float noise in exact ties does not flip the comparison.
    extreme = int(np.count_nonzero(permuted >= observed - 1e-12))
    return (extreme + 1) / (n_permutations + 1)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def compare_all(
    per_query: dict[str, dict[str, float]],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[PairComparison]:
    """Every pair of methods, on the queries all methods share (sorted ids for determinism)."""
    methods = list(per_query)
    queries = sorted(set.intersection(*(set(per_query[m]) for m in methods)))
    raw = []
    for a, b in combinations(methods, 2):
        xa = [per_query[a][q] for q in queries]
        xb = [per_query[b][q] for q in queries]
        d = np.asarray(xa) - np.asarray(xb)
        lo, hi = paired_bootstrap_ci(
            xa, xb, n_resamples=n_resamples, confidence=confidence, seed=seed
        )
        p = randomization_test(xa, xb, n_permutations=n_resamples, seed=seed)
        raw.append(
            (
                a,
                b,
                float(np.mean(xa)),
                float(np.mean(xb)),
                float(d.mean()),
                lo,
                hi,
                p,
                int((d > 0).sum()),
                int((d < 0).sum()),
                int((d == 0).sum()),
            )
        )
    adjusted = holm([r[7] for r in raw])
    return [
        PairComparison(
            a=a,
            b=b,
            mean_a=ma,
            mean_b=mb,
            mean_diff=md,
            ci_low=lo,
            ci_high=hi,
            confidence=confidence,
            p_value=p,
            p_holm=ph,
            queries=len(queries),
            wins=w,
            losses=l_,
            ties=t,
        )
        for (a, b, ma, mb, md, lo, hi, p, w, l_, t), ph in zip(raw, adjusted, strict=True)
    ]
