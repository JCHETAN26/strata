"""Cross-encoder reranking of retrieval candidates.

A cross-encoder reads the query and one passage together and outputs a relevance score; it is
far more expensive than the bi-encoder or BM25 that produced the candidates, so it only rescores
a small pool. Pools are the union of each retriever's top N (union_pool): reciprocal rank fusion
can bury a passage that only one retriever finds, and the reranker should still see it.

Models are pinned to Hugging Face commit hashes, like the embedding models (rag/../scripts/
embed_beir.py), so every machine loads identical weights. Passage text is title + " " + text,
the BEIR convention for neural models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RerankerSpec:
    hf_name: str
    revision: str
    max_length: int
    source: str


RERANKERS: dict[str, RerankerSpec] = {
    "minilm-l6": RerankerSpec(
        hf_name="cross-encoder/ms-marco-MiniLM-L6-v2",  # formerly ms-marco-MiniLM-L-6-v2
        revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
        max_length=512,
        source="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2",
    ),
    "bge-reranker-base": RerankerSpec(
        hf_name="BAAI/bge-reranker-base",
        revision="2cfc18c9415c912f9d8155881c133215df768a70",
        max_length=512,
        source="https://huggingface.co/BAAI/bge-reranker-base",
    ),
}


def union_pool(ranked_lists: list[list[str]], n: int) -> list[str]:
    """Union of each list's top n, deduplicated, in round-robin rank order (rank 1 of every list,
    then rank 2, ...). Deterministic; the order is only a tie-break for the reranker."""
    pool: list[str] = []
    seen: set[str] = set()
    for rank in range(n):
        for ranked in ranked_lists:
            if rank < len(ranked) and ranked[rank] not in seen:
                seen.add(ranked[rank])
                pool.append(ranked[rank])
    return pool


class Reranker:
    """Scores (query, passage) pairs with a pinned cross-encoder.

    model: any object with predict(list[tuple[str, str]], batch_size=...) -> scores; loaded from
    RERANKERS[key] on first use if None (tests inject a fake). Thread safety: not thread-safe.
    """

    def __init__(
        self, key: str, *, device: str = "cpu", batch_size: int = 32, model: Any | None = None
    ) -> None:
        self.key = key
        self.spec = RERANKERS[key]
        self.device = device
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.spec.hf_name,
                revision=self.spec.revision,
                device=self.device,
                max_length=self.spec.max_length,
            )
        return self._model

    def score(self, query: str, passages: list[str]) -> tuple[np.ndarray, float]:
        """Scores for each passage, and the wall-clock seconds the scoring took."""
        if not passages:
            return np.zeros(0, dtype=np.float32), 0.0
        start = time.perf_counter()
        scores = self.model.predict([(query, p) for p in passages], batch_size=self.batch_size)
        return np.asarray(scores, dtype=np.float32), time.perf_counter() - start


def rerank(pool: list[str], scores: dict[str, float], k: int) -> list[str]:
    """Pool ids sorted by descending score; ties keep pool order (stable). Top k."""
    order = sorted(range(len(pool)), key=lambda i: -scores[pool[i]])
    return [pool[i] for i in order][:k]
