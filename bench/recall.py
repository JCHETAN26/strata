"""Recall@k in Python, matching src/eval/recall.cpp (tie-aware and by-id)."""

from __future__ import annotations

import numpy as np

RECALL_TOLERANCE = 1e-5  # relative; same as strata::kRecallTolerance


def distances(base: np.ndarray, query: np.ndarray, ids: np.ndarray, metric: str) -> np.ndarray:
    """Distances (lower = closer, Strata convention) from one query to base[ids]."""
    vectors = base[ids].astype(np.float32)
    q = query.astype(np.float32)
    if metric == "l2":
        return ((vectors - q) ** 2).sum(axis=1)
    if metric == "ip":
        return -(vectors @ q)
    if metric in ("cosine", "angular"):
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(norms == 0, 0.0, (vectors @ q) / norms)
        return 1.0 - cos
    raise ValueError(f"unknown metric {metric}")


def recall_with_ties(
    base: np.ndarray,
    queries: np.ndarray,
    groundtruth: np.ndarray,
    labels: np.ndarray,
    metric: str,
    k: int,
) -> float:
    hits = 0
    for i in range(len(queries)):
        kth = distances(base, queries[i], groundtruth[i, k - 1 : k], metric)[0]
        threshold = kth + RECALL_TOLERANCE * max(1.0, abs(kth))
        found = labels[i, :k]
        found = found[found >= 0]  # FAISS pads with -1
        hits += int((distances(base, queries[i], found, metric) <= threshold).sum())
    return hits / (len(queries) * k)


def recall_by_id(groundtruth: np.ndarray, labels: np.ndarray, k: int) -> float:
    hits = sum(len(set(labels[i, :k]) & set(groundtruth[i, :k])) for i in range(len(labels)))
    return hits / (len(labels) * k)
