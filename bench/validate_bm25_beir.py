"""Reproduce Anserini's published BM25 (flat) results on BEIR with Strata's Bm25Index.

    uv run python bench/validate_bm25_beir.py --dataset scifact

Setup matched to Anserini's regression (src/main/resources/reproduce/from-document-collection/
configs/beir-v1.0.0-scifact.flat.yaml): BeirFlatCollection contents = title + "\\n" + text,
DefaultEnglishAnalyzer (StandardTokenizer, possessive, lowercase, Lucene English stopwords,
Porter), BM25 k1=0.9 b=0.4 with Lucene's length encoding, -hits 1000, -removeQuery, scores
written with %f, evaluated with trec_eval -c (ndcg_cut.10, recall.100, recall.1000).

Pass criteria (documented in docs/devlog.md): every metric within MARGIN of the published value,
and the index's total term count within TERMS_MARGIN (relative) of Anserini's index stats.
Saves the run, metrics, and timing under results/bm25/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import strata
from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new
from ir_eval import mean, ndcg_at_k, read_qrels, recall_at_k, round_scores, write_trec_run

# Published Anserini BM25 flat results (beir-v1.0.0-<name>.flat.yaml) and index stats.
PUBLISHED = {
    "scifact": {
        "nDCG@10": 0.6789,
        "R@100": 0.9253,
        "R@1000": 0.9767,
        "documents": 5183,
        "total_terms": 838128,
    },
}
MARGIN = 0.002  # absolute, per metric
TERMS_MARGIN = 0.001  # relative


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="scifact", choices=sorted(PUBLISHED))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)

    data = REPO_ROOT / "data" / "beir" / args.dataset
    if not (data / "corpus.jsonl").exists():
        print(f"missing {data}: run scripts/prepare_beir.py {args.dataset}", file=sys.stderr)
        return 1
    corpus = load_jsonl(data / "corpus.jsonl")
    queries = {q["_id"]: q["text"] for q in load_jsonl(data / "queries.jsonl")}
    qrels = read_qrels(data / "qrels" / "test.tsv")
    test_qids = sorted(qrels)

    index = strata.Bm25Index(k1=0.9, b=0.4, length_encoding="lucene")
    doc_ids = [d["_id"] for d in corpus]
    start = time.perf_counter()
    index.add([f"{d['title']}\n{d['text']}" for d in corpus])
    index_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ids, scores = index.search([queries[q] for q in test_qids], 1000, threads=args.threads)
    search_seconds = time.perf_counter() - start

    run: dict[str, dict[str, float]] = {}
    for row, qid in enumerate(test_qids):
        hits = {}
        for i, s in zip(ids[row], scores[row], strict=True):
            if i < 0:
                break
            doc = doc_ids[i]
            if doc != qid:  # Anserini -removeQuery
                hits[doc] = float(s)
        run[qid] = hits
    run = round_scores(run)  # as written by Anserini's %f

    metrics = {
        "nDCG@10": mean(ndcg_at_k(run, qrels, 10)),
        "R@100": mean(recall_at_k(run, qrels, 100)),
        "R@1000": mean(recall_at_k(run, qrels, 1000)),
    }
    published = PUBLISHED[args.dataset]
    checks = {name: abs(value - published[name]) <= MARGIN for name, value in metrics.items()}
    terms_ok = (
        abs(index.total_terms - published["total_terms"]) <= TERMS_MARGIN * published["total_terms"]
    )
    docs_ok = len(index) == published["documents"]

    print(f"{args.dataset}: {len(index)} docs, {index.total_terms} terms "
          f"(Anserini {published['total_terms']}), vocabulary {index.vocabulary_size}")  # fmt: skip
    for name, value in metrics.items():
        diff = value - published[name]
        mark = "ok" if checks[name] else "FAIL"
        print(f"  {name:8s} {value:.4f}  published {published[name]:.4f}  diff {diff:+.4f}  {mark}")
    qps = len(test_qids) / search_seconds
    print(
        f"  indexing {index_seconds:.2f} s, search {qps:.0f} QPS (k=1000, threads={args.threads})"
    )

    slug = timestamp_slug()
    out = REPO_ROOT / "results" / "bm25"
    write_trec_run(out / f"{args.dataset}-{slug}.run", run, "strata-bm25")
    write_new(
        out / f"{args.dataset}-{slug}.json",
        json.dumps(
            {
                **metadata(),
                "dataset": args.dataset,
                "setup": "BEIR flat, anserini_english analyzer, k1=0.9 b=0.4, lucene lengths",
                "metrics": metrics,
                "published": published,
                "margin": MARGIN,
                "passed": all(checks.values()) and terms_ok and docs_ok,
                "documents": len(index),
                "total_terms": index.total_terms,
                "vocabulary_size": index.vocabulary_size,
                "index_seconds": index_seconds,
                "search_seconds": search_seconds,
                "queries": len(test_qids),
                "qps": qps,
                "threads": args.threads,
            },
            indent=2,
        )
        + "\n",
    )
    ok = all(checks.values()) and terms_ok and docs_ok
    print("PASS" if ok else "FAIL", f"(margin {MARGIN} absolute; terms within {TERMS_MARGIN:.1%})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
