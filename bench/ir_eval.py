"""Retrieval evaluation with trec_eval semantics (what Anserini and BEIR report).

- Ranking: documents sorted by score descending, ties broken by document id *descending*
  (trec_eval's comparator), regardless of the order they were given in.
- nDCG@k (trec_eval ndcg_cut.k): gain = relevance grade, discount 1 / log2(rank + 1); the ideal
  ranking uses every judged document with grade > 0. A query with no relevant documents scores 0.
- Recall@k (trec_eval recall.k): relevant (grade > 0) documents in the top k / all relevant.
- Averages follow `trec_eval -c`: over every query in the qrels; queries without results count 0.
- Anserini writes scores with %f (6 decimals), which creates ties; round_scores() reproduces that.

Tested against pytrec_eval (trec_eval's C code) in tests/python/test_ir_eval.py.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path

Run = Mapping[str, Mapping[str, float]]  # query id -> doc id -> score
Qrels = Mapping[str, Mapping[str, int]]  # query id -> doc id -> relevance grade


def ranked(scores: Mapping[str, float]) -> list[str]:
    """Doc ids in trec_eval order: score descending, then doc id descending."""
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)]


def ndcg_at_k(run: Run, qrels: Qrels, k: int) -> dict[str, float]:
    out = {}
    for qid, judged in qrels.items():
        gains = sorted((g for g in judged.values() if g > 0), reverse=True)[:k]
        ideal = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        if ideal == 0:
            out[qid] = 0.0
            continue
        docs = ranked(run.get(qid, {}))[:k]
        dcg = sum(max(judged.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(docs))
        out[qid] = dcg / ideal
    return out


def recall_at_k(run: Run, qrels: Qrels, k: int) -> dict[str, float]:
    out = {}
    for qid, judged in qrels.items():
        relevant = {d for d, g in judged.items() if g > 0}
        if not relevant:
            out[qid] = 0.0
            continue
        docs = ranked(run.get(qid, {}))[:k]
        out[qid] = len(relevant.intersection(docs)) / len(relevant)
    return out


def mean(per_query: Mapping[str, float]) -> float:
    return sum(per_query.values()) / len(per_query) if per_query else 0.0


def round_scores(run: Run, decimals: int = 6) -> dict[str, dict[str, float]]:
    """Scores as they appear in a TREC run file written with %f (Anserini's format)."""
    return {q: {d: round(s, decimals) for d, s in docs.items()} for q, docs in run.items()}


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """BEIR qrels TSV: header row, then query-id, corpus-id, score."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, doc, grade in reader:
            qrels.setdefault(qid, {})[doc] = int(grade)
    return qrels


def write_trec_run(path: Path, run: Run, tag: str) -> None:
    """TREC run format, scores with %f, ranks in trec_eval order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for qid in sorted(run):
            for rank, doc in enumerate(ranked(run[qid]), start=1):
                f.write(f"{qid} Q0 {doc} {rank} {run[qid][doc]:f} {tag}\n")
