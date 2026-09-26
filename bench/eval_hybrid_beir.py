"""Evaluate BM25, dense, and fused retrieval on a BEIR dataset.

    uv run python bench/eval_hybrid_beir.py --dataset scifact --model bge-small-en-v1.5

Protocol (fixed before looking at test results):
- Dense retrieval: embeddings from scripts/embed_beir.py (model and revision pinned there and
  recorded here), exact search (BruteForceIndex, negated inner product on L2-normalized vectors).
- BM25: Bm25Index with Anserini settings over title + "\\n" + text (as in the BM25 validation).
- RRF: k = 60 (Cormack et al. 2009), not tuned. Each retriever contributes its top `candidates`.
- Weighted fusion: weight * dense + (1 - weight) * bm25 over min-max-normalized scores. The weight
  is chosen on the *train* split (grid 0.00..1.00 step 0.05, best nDCG@10; ties go to the smaller
  weight) and then applied unchanged to the test split. Train and test query ids are checked to
  be disjoint. Test queries are never used for tuning.
- Metrics: nDCG@10 and R@100 with trec_eval semantics (bench/ir_eval.py), documents whose id
  equals the query id removed (Anserini -removeQuery / BEIR ignore_identical_ids).

Saves everything (embedding model, revision, and formatting; tuning curve; test metrics; timing)
to results/hybrid/<dataset>-<model>-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import strata
from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new
from ir_eval import mean, ndcg_at_k, read_qrels, recall_at_k

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_datasets import read_bin

WEIGHT_GRID = [round(0.05 * i, 2) for i in range(21)]
# Published single-retriever results the baselines must reproduce (pinned sources inside).
ANSERINI_REFERENCE = json.loads(
    (REPO_ROOT / "results" / "bm25" / "anserini_reference.json").read_text()
)
DENSE_REFERENCE = json.loads(
    (REPO_ROOT / "results" / "hybrid" / "dense_reference.json").read_text()
)
REFERENCE_MARGIN = 0.002  # absolute, as for the BM25 validation


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default="bge-small-en-v1.5")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "hybrid")
    args = parser.parse_args(argv)

    data = REPO_ROOT / "data" / "beir" / args.dataset
    emb_dir = REPO_ROOT / "data" / "embeddings" / args.dataset / args.model
    if not (emb_dir / "meta.json").exists():
        print(f"missing {emb_dir}: run scripts/embed_beir.py", file=sys.stderr)
        return 1
    emb_meta = json.loads((emb_dir / "meta.json").read_text())
    beir_meta = json.loads((data / "meta.json").read_text())
    if emb_meta["dataset_sha256"] != beir_meta["source_sha256"]:
        print("embeddings were computed from a different copy of the dataset", file=sys.stderr)
        return 1

    corpus = load_jsonl(data / "corpus.jsonl")
    query_text = {q["_id"]: q["text"] for q in load_jsonl(data / "queries.jsonl")}
    doc_vectors = read_bin(emb_dir / "corpus.fbin", np.float32)
    query_vectors = read_bin(emb_dir / "queries.fbin", np.float32)
    query_row = {
        qid: i for i, qid in enumerate(json.loads((emb_dir / "query_ids.json").read_text()))
    }
    if json.loads((emb_dir / "corpus_ids.json").read_text()) != [d["_id"] for d in corpus]:
        print("embedding rows do not match the corpus order", file=sys.stderr)
        return 1

    splits = {name: read_qrels(data / "qrels" / f"{name}.tsv") for name in ("train", "test")}
    overlap = set(splits["train"]) & set(splits["test"])
    if overlap:
        print(f"train and test share query ids: {sorted(overlap)[:5]}", file=sys.stderr)
        return 1

    index = strata.HybridIndex(doc_vectors.shape[1], metric="ip")
    start = time.perf_counter()
    index.add(
        [d["_id"] for d in corpus], [f"{d['title']}\n{d['text']}" for d in corpus], doc_vectors
    )
    build_seconds = time.perf_counter() - start

    def evaluate(split: str, method: str, weight: float = 0.5) -> dict[str, float]:
        qrels = splits[split]
        qids = sorted(qrels)
        texts = [query_text[q] for q in qids]
        vectors = query_vectors[[query_row[q] for q in qids]]
        start = time.perf_counter()
        ids, scores = index.search(
            texts,
            vectors,
            100,
            method=method,
            candidates=args.candidates,
            rrf_k=args.rrf_k,
            weight=weight,
            threads=args.threads,
        )
        seconds = time.perf_counter() - start
        run = {}
        for row, qid in enumerate(qids):
            hits = zip(ids[row], scores[row], strict=True)
            run[qid] = {
                index.doc_id(i): float(s) for i, s in hits if i >= 0 and index.doc_id(i) != qid
            }
        return {
            "nDCG@10": mean(ndcg_at_k(run, qrels, 10)),
            "R@100": mean(recall_at_k(run, qrels, 100)),
            "queries": len(qids),
            "qps": len(qids) / seconds,
        }

    # Tune the fusion weight on train only.
    curve = {w: evaluate("train", "weighted", w)["nDCG@10"] for w in WEIGHT_GRID}
    best_weight = max(WEIGHT_GRID, key=lambda w: (curve[w], -w))

    test = {
        "bm25": evaluate("test", "bm25"),
        "dense": evaluate("test", "dense"),
        "rrf": evaluate("test", "rrf"),
        "weighted": evaluate("test", "weighted", best_weight),
    }

    # Baselines against published references (when this dataset/model has one).
    checks = {}
    bm25_ref = ANSERINI_REFERENCE.get(args.dataset)
    if bm25_ref:
        checks["bm25"] = {
            m: (test["bm25"][m], bm25_ref["published"][m]) for m in ("nDCG@10", "R@100")
        }
    dense_ref = DENSE_REFERENCE.get(args.dataset, {}).get(args.model)
    if dense_ref and dense_ref["revision"] == emb_meta["revision"]:
        checks["dense"] = {
            m: (test["dense"][m], dense_ref["published"][m]) for m in ("nDCG@10", "R@100")
        }
    baselines_ok = all(
        abs(ours - ref) <= REFERENCE_MARGIN for c in checks.values() for ours, ref in c.values()
    )

    print(f"{args.dataset} test ({test['bm25']['queries']} queries), dense = "
          f"{emb_meta['hf_name']}@{emb_meta['revision'][:12]}")  # fmt: skip
    print(f"  weighted fusion weight (dense) = {best_weight} chosen on train "
          f"({len(splits['train'])} queries, train nDCG@10 {curve[best_weight]:.4f})")  # fmt: skip
    for name, m in test.items():
        ref = checks.get(name, {}).get("nDCG@10")
        note = f"  (published {ref[1]:.4f})" if ref else ""
        print(f"  {name:9s} nDCG@10 {m['nDCG@10']:.4f}  R@100 {m['R@100']:.4f}  "
              f"{m['qps']:.0f} QPS{note}")  # fmt: skip
    print("baselines match published references" if baselines_ok else "BASELINE MISMATCH")

    record = {
        **metadata(),
        "dataset": args.dataset,
        "dataset_sha256": beir_meta["source_sha256"],
        "embedding": emb_meta,
        "protocol": {
            "candidates": args.candidates,
            "rrf_k": args.rrf_k,
            "rrf_k_tuned": False,
            "weight_grid": WEIGHT_GRID,
            "weight_tuned_on": "train",
            "weight": best_weight,
            "tie_break": "smaller weight",
            "remove_query_id": True,
            "bm25": "Bm25Index(k1=0.9, b=0.4, lucene lengths, anserini_english), title\\ntext",
            "dense_search": "BruteForceIndex, negated inner product, exact",
        },
        "train_curve_ndcg10": {str(w): v for w, v in curve.items()},
        "test": test,
        "baseline_checks": {
            name: {m: {"ours": o, "published": r} for m, (o, r) in c.items()}
            for name, c in checks.items()
        },
        "baselines_match_published": baselines_ok,
        "references": {"bm25": bm25_ref, "dense": dense_ref},
        "build_seconds": build_seconds,
        "threads": args.threads,
    }
    path = args.out_dir / f"{args.dataset}-{args.model}-{timestamp_slug()}.json"
    write_new(path, json.dumps(record, indent=2) + "\n")
    print(f"saved {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    return 0 if baselines_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
