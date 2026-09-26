"""Hybrid retrieval: one vector index and one Bm25Index over the same documents, fused.

HybridIndex keeps the two indexes' internal ids aligned by construction: every add() puts
document i into both, in the same order, under one lock, and remove() tombstones it in both.
External document ids (strings) map to internal ids.

Fusion methods:
- "rrf": reciprocal rank fusion, k = 60 by default (Cormack et al. 2009); ranks only.
- "weighted": weight * dense + (1 - weight) * bm25 over min-max-normalized scores. The weight is
  a parameter; choose it on training queries (see bench/eval_hybrid_beir.py), never on test.
- "dense", "bm25": either retriever alone (for comparisons).

Thread safety: add() and remove() are serialized by a lock, so the two indexes never disagree on
ids. Searches take no Python lock (each index has its own reader lock and releases the GIL); a
search running concurrently with an add may see a new document in one retriever before the other.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from strata._core import Bm25Index, BruteForceIndex, fuse_rrf, fuse_weighted

METHODS = ("rrf", "weighted", "dense", "bm25")


class HybridIndex:
    """Vector index + BM25 index with shared ids and result fusion."""

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        *,
        vector_index: Any | None = None,
        bm25: Bm25Index | None = None,
    ) -> None:
        """metric applies to the default BruteForceIndex. For L2-normalized embeddings, "ip"
        (negated dot product) ranks like cosine and skips the norm computations.

        vector_index: any Strata vector index (BruteForceIndex, PqIndex, HnswIndex); it must be
        empty. bm25: a configured, empty Bm25Index (default: Anserini settings)."""
        self.vector_index = (
            vector_index if vector_index is not None else BruteForceIndex(dim, metric)
        )
        self.bm25 = bm25 if bm25 is not None else Bm25Index()
        if len(self.vector_index) != 0 or len(self.bm25) != 0:
            raise ValueError("HybridIndex needs empty indexes so their ids line up")
        self.dim = dim
        self._doc_ids: list[str] = []
        self._internal: dict[str, int] = {}
        self._removed: set[int] = set()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._doc_ids) - len(self._removed)

    def add(self, doc_ids: Sequence[str], texts: Sequence[str], vectors: np.ndarray) -> np.ndarray:
        """Add documents to both indexes; returns their internal ids.

        Everything is validated before either index changes, so a failed add leaves both
        untouched. Adds are serialized."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape (n, {self.dim}), got {vectors.shape}")
        if not (len(doc_ids) == len(texts) == len(vectors)):
            raise ValueError("doc_ids, texts, and vectors must have the same length")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("duplicate doc ids in the batch")
        with self._lock:
            existing = [d for d in doc_ids if d in self._internal]
            if existing:
                raise ValueError(f"doc ids already present: {existing[:5]}")
            vector_ids = self.vector_index.add(vectors)
            text_ids = self.bm25.add(list(texts))
            if not np.array_equal(vector_ids, text_ids):  # cannot happen with empty-start indexes
                raise RuntimeError("vector and BM25 indexes assigned different ids")
            for doc_id, internal in zip(doc_ids, vector_ids.tolist(), strict=True):
                self._internal[doc_id] = internal
                self._doc_ids.append(doc_id)
        return vector_ids

    def remove(self, doc_id: str) -> None:
        """Tombstone a document in both indexes. KeyError if unknown or already removed."""
        with self._lock:
            internal = self._internal.pop(doc_id)
            self.vector_index.remove(internal)
            self.bm25.remove(internal)
            self._removed.add(internal)

    def doc_id(self, internal: int) -> str:
        return self._doc_ids[internal]

    def search(
        self,
        query_texts: Sequence[str],
        query_vectors: np.ndarray,
        k: int = 10,
        *,
        method: str = "rrf",
        candidates: int = 100,
        rrf_k: float = 60.0,
        weight: float = 0.5,
        threads: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch search. Returns (internal ids, scores), shape (num_queries, k), -1 / -inf padded.

        candidates: how many results each retriever contributes to fusion.
        weight: dense weight for method="weighted" (BM25 gets 1 - weight)."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if query_vectors.ndim != 2 or len(query_vectors) != len(query_texts):
            raise ValueError("need one query vector (row) per query text")
        depth = max(candidates, k)
        if method == "dense":
            ids, dist = self.vector_index.search(query_vectors, k, threads=threads)
            return ids, np.where(ids >= 0, -dist, -np.inf).astype(np.float32)
        if method == "bm25":
            return self.bm25.search(list(query_texts), k, threads=threads)
        dense_ids, dense_dist = self.vector_index.search(query_vectors, depth, threads=threads)
        bm25_ids, bm25_scores = self.bm25.search(list(query_texts), depth, threads=threads)
        if method == "rrf":
            return fuse_rrf([dense_ids, bm25_ids], k, rrf_k=rrf_k, threads=threads)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        dense_scores = np.where(dense_ids >= 0, -dense_dist, 0.0).astype(np.float32)
        return fuse_weighted(
            [dense_ids, bm25_ids],
            [dense_scores, np.where(bm25_ids >= 0, bm25_scores, 0.0).astype(np.float32)],
            [weight, 1.0 - weight],
            k,
            threads=threads,
        )

    def search_doc_ids(self, *args: Any, **kwargs: Any) -> tuple[list[list[str]], np.ndarray]:
        """search(), with external document ids instead of internal ids."""
        ids, scores = self.search(*args, **kwargs)
        return [[self._doc_ids[i] for i in row if i >= 0] for row in ids], scores
