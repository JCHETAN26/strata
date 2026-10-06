"""BEIR runner: BM25, dense, and fused retrieval, with paired significance tests.

    uv run python bench/eval_hybrid_beir.py --dataset scifact --model bge-small-en-v1.5
    uv run python bench/eval_hybrid_beir.py --dataset scifact nfcorpus fiqa

Protocol (pre-declared; fixed before looking at test results):
- Dense retrieval: embeddings from scripts/embed_beir.py (model and revision pinned there and
  recorded here), exact search (BruteForceIndex, negated inner product on L2-normalized vectors).
- BM25: Bm25Index with Anserini settings over title + "\\n" + text (as in the BM25 validation).
- RRF: k = 60 (Cormack et al. 2009), not tuned. Each retriever contributes its top `candidates`.
- Weighted fusion: weight * dense + (1 - weight) * bm25 over min-max-normalized scores. The weight
  comes from WEIGHT_RULE: tune on the dataset's *dev* split if it exists, else on *train*, else
  use a fixed 0.5. Tuning: grid 0.00..1.00 step 0.05, best nDCG@10, ties to the smaller weight.
  The split used ("dev" / "train" / "fixed") is recorded per dataset. The tuning split must not
  share query ids with test (checked). Test queries are never used for tuning.
- Metrics: nDCG@10 and R@100 with trec_eval semantics (bench/ir_eval.py), documents whose id
  equals the query id removed (Anserini -removeQuery / BEIR ignore_identical_ids).
- Significance: for every pair of methods, the mean per-query difference with a 95% paired
  bootstrap CI (10,000 resamples) and a two-sided paired randomization test (10,000 sign flips),
  Holm-adjusted across the pairs (bench/significance.py). Per-query scores are saved so the
  tests can be recomputed.

Saves one result per dataset to results/hybrid/<dataset>-<model>-<timestamp>.json.
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
from significance import compare_all

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_datasets import read_bin

METHODS = ("bm25", "dense", "rrf", "weighted")
WEIGHT_GRID = [round(0.05 * i, 2) for i in range(21)]
# Pre-declared rule for the weighted-fusion weight: first split that exists, else FIXED_WEIGHT.
WEIGHT_RULE = ("dev", "train")
FIXED_WEIGHT = 0.5
# Significance testing.
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE = 0.95
SEED = 0
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


def choose_tuning_split(qrels_dir: Path) -> str | None:
    """The pre-declared rule: dev if present, else train, else None (fixed weight)."""
    for split in WEIGHT_RULE:
        if (qrels_dir / f"{split}.tsv").exists():
            return split
    return None


def run_dataset(dataset: str, args: argparse.Namespace) -> int:
    data = REPO_ROOT / "data" / "beir" / dataset
    emb_dir = REPO_ROOT / "data" / "embeddings" / dataset / args.model
    if not (emb_dir / "meta.json").exists():
        print(f"missing {emb_dir}: run scripts/embed_beir.py --dataset {dataset}", file=sys.stderr)
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
    query_ids = json.loads((emb_dir / "query_ids.json").read_text())
    query_row = {qid: i for i, qid in enumerate(query_ids)}
    if json.loads((emb_dir / "corpus_ids.json").read_text()) != [d["_id"] for d in corpus]:
        print("embedding rows do not match the corpus order", file=sys.stderr)
        return 1

    tuning_split = choose_tuning_split(data / "qrels")
    splits = {"test": read_qrels(data / "qrels" / "test.tsv")}
    if tuning_split is not None:
        splits[tuning_split] = read_qrels(data / "qrels" / f"{tuning_split}.tsv")
        overlap = set(splits[tuning_split]) & set(splits["test"])
        if overlap:
            print(
                f"{tuning_split} and test share query ids: {sorted(overlap)[:5]}", file=sys.stderr
            )
            return 1

    index = strata.HybridIndex(doc_vectors.shape[1], metric="ip")
    start = time.perf_counter()
    texts = [f"{d['title']}\n{d['text']}" for d in corpus]
    index.add([d["_id"] for d in corpus], texts, doc_vectors)
    build_seconds = time.perf_counter() - start

    def evaluate(split: str, method: str, weight: float = FIXED_WEIGHT) -> dict:
        qrels = splits[split]
        qids = sorted(qrels)
        start = time.perf_counter()
        ids, scores = index.search(
            [query_text[q] for q in qids],
            query_vectors[[query_row[q] for q in qids]],
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
        ndcg = ndcg_at_k(run, qrels, 10)
        recall = recall_at_k(run, qrels, 100)
        return {
            "nDCG@10": mean(ndcg),
            "R@100": mean(recall),
            "queries": len(qids),
            "qps": len(qids) / seconds,
            "per_query": {"nDCG@10": ndcg, "R@100": recall},
        }

    # Methods to evaluate, in the canonical order; each search over a large corpus is slow, so
    # print progress as it happens rather than only at the end.
    selected = [m for m in METHODS if m in args.methods]
    print(f"{dataset} test ({len(splits['test'])} queries, {len(corpus):,} passages), dense = "
          f"{emb_meta['hf_name']}@{emb_meta['revision'][:12]}", flush=True)  # fmt: skip
    print(f"  methods: {', '.join(selected)}", flush=True)

    # Weighted-fusion weight, by the pre-declared rule. Only tuned when 'weighted' is evaluated:
    # the grid re-searches the whole corpus once per point, which is the dominant cost at scale.
    curve: dict[float, float] = {}
    if "weighted" in selected and tuning_split is not None:
        n_tune = len(splits[tuning_split])
        print(
            f"  tuning weighted weight on {tuning_split} ({n_tune} q), {len(WEIGHT_GRID)} points:",
            flush=True,
        )
        for w in WEIGHT_GRID:
            curve[w] = evaluate(tuning_split, "weighted", w)["nDCG@10"]
            print(f"    weight={w:.2f}  nDCG@10 {curve[w]:.4f}", flush=True)
        weight = max(WEIGHT_GRID, key=lambda w: (curve[w], -w))
    else:
        weight = FIXED_WEIGHT

    results = {}
    for m in selected:
        results[m] = evaluate("test", m, weight)
        r = results[m]
        print(f"  {m:9s} nDCG@10 {r['nDCG@10']:.4f}  R@100 {r['R@100']:.4f}  "
              f"{r['qps']:.0f} QPS", flush=True)  # fmt: skip
    per_query = {m: r.pop("per_query") for m, r in results.items()}
    significance = (
        {
            metric: [
                c.to_dict()
                for c in compare_all(
                    {m: per_query[m][metric] for m in selected},
                    n_resamples=BOOTSTRAP_RESAMPLES,
                    confidence=CONFIDENCE,
                    seed=SEED,
                )
            ]
            for metric in ("nDCG@10", "R@100")
        }
        if len(selected) >= 2
        else {"nDCG@10": [], "R@100": []}
    )

    # Baselines against published references (when this dataset/model has one and it was run).
    checks = {}
    bm25_ref = ANSERINI_REFERENCE.get(dataset)
    if bm25_ref and "bm25" in results:
        checks["bm25"] = {
            m: (results["bm25"][m], bm25_ref["published"][m]) for m in ("nDCG@10", "R@100")
        }
    dense_ref = DENSE_REFERENCE.get(dataset, {}).get(args.model)
    if dense_ref and dense_ref["revision"] == emb_meta["revision"] and "dense" in results:
        checks["dense"] = {
            m: (results["dense"][m], dense_ref["published"][m]) for m in ("nDCG@10", "R@100")
        }
    baselines_ok = all(
        abs(ours - ref) <= REFERENCE_MARGIN for c in checks.values() for ours, ref in c.values()
    )

    source = (
        f"tuned on {tuning_split} ({len(splits[tuning_split])} q, nDCG@10 {curve[weight]:.4f})"
        if curve
        else ("fixed (no dev or train split)" if "weighted" in selected else "not evaluated")
    )
    print(f"  weighted fusion weight (dense) = {weight}, {source}")
    for name in selected:
        ref = checks.get(name, {}).get("nDCG@10")
        if ref:
            ours = results[name]["nDCG@10"]
            print(f"    {name:9s} published nDCG@10 {ref[1]:.4f} "
                  f"(ours {ours:.4f}, Δ {ours - ref[1]:+.4f})")  # fmt: skip
    if significance["nDCG@10"]:
        print(f"  paired differences in nDCG@10 "
              f"({CONFIDENCE:.0%} bootstrap CI, randomization p, Holm):")  # fmt: skip
        for c in significance["nDCG@10"]:
            wlt = f"{c['wins']}/{c['losses']}/{c['ties']}"
            print(f"    {c['a']:>8s} - {c['b']:<8s} {c['mean_diff']:+.4f} "
                  f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  p={c['p_value']:.4f}  "
                  f"p_holm={c['p_holm']:.4f}  WLT {wlt}")  # fmt: skip
    if checks:
        print("  baselines match published references" if baselines_ok else "  BASELINE MISMATCH")

    record = {
        **metadata(),
        "dataset": dataset,
        "dataset_sha256": beir_meta["source_sha256"],
        "embedding": emb_meta,
        "protocol": {
            "methods": selected,
            "candidates": args.candidates,
            "rrf_k": args.rrf_k,
            "rrf_k_tuned": False,
            "weight_rule": "dev if present, else train, else fixed 0.5",
            "weight_source": (tuning_split or "fixed") if curve else "not tuned (weighted skipped)",
            "weight_tuned_on": tuning_split if curve else None,
            "tuning_queries": len(splits[tuning_split]) if curve else 0,
            "weight_grid": WEIGHT_GRID,
            "weight": weight,
            "tie_break": "smaller weight",
            "remove_query_id": True,
            "bm25": "Bm25Index(k1=0.9, b=0.4, lucene lengths, anserini_english), title\\ntext",
            "dense_search": "BruteForceIndex, negated inner product, exact",
            "significance": {
                "ci": f"paired bootstrap, {BOOTSTRAP_RESAMPLES} resamples, {CONFIDENCE:.0%}",
                "test": f"two-sided paired randomization, {BOOTSTRAP_RESAMPLES} sign flips",
                "multiple_comparisons": "Holm-Bonferroni over all method pairs, per metric",
                "seed": SEED,
            },
        },
        "tuning_curve_ndcg10": {str(w): v for w, v in curve.items()},
        "test": results,
        "per_query": per_query,
        "significance": significance,
        "baseline_checks": {
            name: {m: {"ours": o, "published": r} for m, (o, r) in c.items()}
            for name, c in checks.items()
        },
        "baselines_match_published": baselines_ok,
        "references": {"bm25": bm25_ref, "dense": dense_ref},
        "build_seconds": build_seconds,
        "threads": args.threads,
    }
    path = args.out_dir / f"{dataset}-{args.model}-{timestamp_slug()}.json"
    write_new(path, json.dumps(record, indent=2) + "\n")
    print(f"  saved {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    return 0 if baselines_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", nargs="+", default=["scifact"])
    parser.add_argument("--model", default="bge-small-en-v1.5")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="which methods to evaluate (default: all). On a huge corpus, drop 'weighted' to skip "
        "its per-weight tuning grid, which re-searches the whole corpus once per grid point.",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "hybrid")
    args = parser.parse_args(argv)
    status = 0
    for dataset in args.dataset:
        status |= run_dataset(dataset, args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
