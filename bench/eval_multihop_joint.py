"""Joint reranking of single-hop and hop-2 candidates on BEIR-HotpotQA subsets.

    uv run python bench/eval_multihop_joint.py \\
        --rerank-result results/rerank/hotpotqa-subset-n100-seed0-bg20000-<stamp>.json \\
        --multihop-result results/multihop/hotpotqa-subset-n100-seed0-bg20000-<stamp>.json

Follow-up to bench/eval_multihop.py, whose bge variant kept the first `keep` fused passages and
could drop a gold passage below them (one comparison question). Here nothing is kept by rule:

- Single-hop pool: union of BM25 top N and dense top N, scored by bge-reranker-base against the
  question (as in bench/eval_rerank.py).
- Hop-2 pools: for each of the fused top m passages p, the hop-2 query (question + expansion of
  p, with the expansion chosen in the --multihop-result run's no-model stage) retrieves BM25 and
  dense top `depth`; the union is scored by bge against that hop-2 query.
- Final top 5: every candidate in any pool, ranked by its best score over the pools containing it
  (rag/multihop.py joint_rank).

Protocol (pre-declared):
- Tuning on the dev subset only, by R@5, over 8 settings: N in {10, 20} x depth in {5, 10} x
  m in {1, 2}; ties to smaller m, then smaller depth, then smaller N.
- Test: joint_bge against fused_top5, union_top5_no_model, rerank_bge-reranker-base (from
  --rerank-result) and multihop_no_model, multihop_bge (from --multihop-result). Planned family
  (Holm over 3 per metric): joint vs fused, joint vs single-hop bge, joint vs two-hop no-model.
- Metrics, ceiling, per-type split, comparison-question check (passes if the mean R@5
  difference from fused_top5 on comparison questions is >= 0) and per-query end-to-end latency
  as in bench/eval_multihop.py.

Saves results/multihop/<dataset>-joint-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new
from eval_multihop import (
    BGE,
    RERANKER,
    Encoder,
    Hops,
    bge_query_encoder,
    latency_summary,
    retrieve_hop2,
    score_hop2,
    type_check,
)
from eval_rerank import Dataset, load_jsonl, recall
from ir_eval import read_qrels
from significance import compare_all

sys.path.insert(0, str(REPO_ROOT / "rag"))
from multihop import joint_rank
from rerank import RERANKERS, Reranker, union_pool

K = 5
GRID = {"m": [1, 2], "depth": [5, 10], "N": [10, 20]}
PLANNED = [
    ("joint_bge", "fused_top5"),
    ("joint_bge", BGE),
    ("joint_bge", "multihop_no_model"),
]


def single_hop_scores(
    hops: Hops, reranker: Reranker, q: str, bm25: list[str], dense: list[str], n: int
) -> tuple[dict[str, float], float]:
    pool = union_pool([bm25, dense], n)
    s, sec = reranker.score(hops.data.query_text[q], [hops.data.passage[d] for d in pool])
    return dict(zip(pool, s.tolist(), strict=True)), sec


def pools_for(
    single: tuple[list[str], list[str]],
    single_scores: dict[str, float],
    hop2: list[tuple[list[str], list[str]]],
    hop2_scores: list[dict[str, float]],
    m: int,
    depth: int,
    n: int,
) -> list[tuple[list[str], dict[str, float]]]:
    """The scored pools for one setting; scores computed on deeper pools are reused (subsets)."""
    pools = [(union_pool(list(single), n), single_scores)]
    pools += [
        (union_pool([bm25, dense], depth), scores)
        for (bm25, dense), scores in zip(hop2[:m], hop2_scores[:m], strict=True)
    ]
    return pools


def main(
    argv: list[str] | None = None,
    *,
    reranker: Reranker | None = None,
    encode: Encoder | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="hotpotqa-subset-n100-seed0-bg20000")
    parser.add_argument("--tune-dataset", default="hotpotqa-dev-subset-n100-seed0-bg20000")
    parser.add_argument("--rerank-result", type=Path, required=True)
    parser.add_argument("--multihop-result", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "multihop")
    args = parser.parse_args(argv)
    if args.tune_dataset == args.dataset:
        raise SystemExit("tune on a different (dev) subset than the test subset")
    rerank_base = json.loads(args.rerank_result.read_text())
    multihop_base = json.loads(args.multihop_result.read_text())
    for path, rec in ((args.rerank_result, rerank_base), (args.multihop_result, multihop_base)):
        if rec["dataset"] != args.dataset:
            raise SystemExit(f"{path} is for {rec['dataset']}, not {args.dataset}")
    expansion = multihop_base["configs"]["multihop_no_model"]["expansion"]
    reranker = reranker or Reranker(RERANKER)
    encode = encode or bge_query_encoder()
    m_max, d_max, n_max = max(GRID["m"]), max(GRID["depth"]), max(GRID["N"])

    def load(name: str, split: str) -> tuple[Hops, list[str], dict[str, set[str]], dict]:
        data = Dataset(name)
        qrels = read_qrels(data.dir / "qrels" / f"{split}.tsv")
        qids = sorted(q for q in qrels if any(g > 0 for g in qrels[q].values()))
        relevant = {q: {d for d, g in qrels[q].items() if g > 0} for q in qids}
        types = {a["_id"]: a["type"] for a in load_jsonl(data.dir / "answers.jsonl")}
        return Hops(data, encode), qids, relevant, types

    # --- Tuning (dev subset only): score the deepest pools once, evaluate every setting.
    tune_hops, tune_qids, tune_relevant, _ = load(args.tune_dataset, "dev")
    ranks = tune_hops.hop1(tune_qids, n_max)
    hop1 = {q: ranks["rrf"][q][:K] for q in tune_qids}
    lists = retrieve_hop2(tune_hops, tune_qids, hop1, m_max, [expansion], d_max)
    single, s_scores, h_scores = {}, {}, {}
    for q in tune_qids:
        single[q] = (ranks["bm25"][q], ranks["dense"][q])
        s_scores[q], _ = single_hop_scores(tune_hops, reranker, q, *single[q], n_max)
        h_scores[q], _ = score_hop2(
            tune_hops, reranker, q, hop1[q], lists[(q, expansion)], expansion, d_max
        )
    rows = []
    for m, depth, n in itertools.product(*GRID.values()):
        r5 = np.mean(
            [
                recall(
                    joint_rank(
                        pools_for(
                            single[q], s_scores[q], lists[(q, expansion)], h_scores[q], m, depth, n
                        ),
                        K,
                    ),
                    tune_relevant[q],
                )
                for q in tune_qids
            ]
        )
        rows.append({"m": m, "depth": depth, "N": n, "R@5": float(r5)})
    best = max(rows, key=lambda r: (r["R@5"], -r["m"], -r["depth"], -r["N"]))
    cfg = {k: best[k] for k in GRID}
    print(f"tune ({len(rows)} settings, {len(tune_qids)} queries, expansion {expansion}): ", end="")
    print(f"{cfg} R@5 {best['R@5']:.4f}")

    # --- Test: each query end to end, timed.
    hops, qids, relevant, types = load(args.dataset, "test")
    data = hops.data
    test_ranks = hops.hop1(qids, K)
    methods: dict[str, dict[str, list[str]]] = {
        m: {q: rerank_base["top5"][m][q] for q in qids}
        for m in ("fused_top5", "union_top5_no_model", BGE)
    }
    methods |= {m: {q: multihop_base["top5"][m][q] for q in qids} for m in multihop_base["top5"]}
    if {q: test_ranks["rrf"][q][:K] for q in qids} != methods["fused_top5"]:
        raise SystemExit(f"fused_top5 differs from {args.rerank_result}")
    methods["joint_bge"], ms, pool_sets = {}, {}, {}
    for q in qids:
        t0 = time.perf_counter()
        text, vec = [data.query_text[q]], data.query_vec[q][None, :]
        found = {}
        for method, depth in (("rrf", K), ("bm25", cfg["N"]), ("dense", cfg["N"])):
            ids, _ = data.index.search(text, vec, depth + 1, method=method)
            found[method] = [
                data.index.doc_id(i) for i in ids[0] if i >= 0 and data.index.doc_id(i) != q
            ][:depth]
        if found["rrf"] != methods["fused_top5"][q]:
            raise SystemExit(f"{q}: per-query hop 1 differs from the batched ranking")
        t1 = time.perf_counter()
        texts = hops.queries(q, found["rrf"], cfg["m"], expansion)
        vectors = hops.encode(texts)
        t2 = time.perf_counter()
        hop2 = hops.search([q] * len(texts), texts, vectors, cfg["depth"])
        t3 = time.perf_counter()
        single_scores, _ = single_hop_scores(
            hops, reranker, q, found["bm25"], found["dense"], cfg["N"]
        )
        hop2_scores, _ = score_hop2(hops, reranker, q, found["rrf"], hop2, expansion, cfg["depth"])
        pools = pools_for(
            (found["bm25"], found["dense"]),
            single_scores,
            hop2,
            hop2_scores,
            cfg["m"],
            cfg["depth"],
            cfg["N"],
        )
        methods["joint_bge"][q] = joint_rank(pools, K)
        t4 = time.perf_counter()
        pool_sets[q] = {d for pool, _ in pools for d in pool}
        ms[q] = {
            "hop1_search": (t1 - t0) * 1000,
            "hop2_embed": (t2 - t1) * 1000,
            "hop2_search": (t3 - t2) * 1000,
            "rerank_and_combine": (t4 - t3) * 1000,
            "total": (t4 - t0) * 1000,
        }
    latency = latency_summary(ms)
    ceiling = float(np.mean([relevant[q] <= pool_sets[q] for q in qids]))
    mean_pool = float(np.mean([len(pool_sets[q]) for q in qids]))

    per_query: dict[str, dict[str, dict[str, float]]] = {"R@5": {}, "all_relevant@5": {}}
    for m, top in methods.items():
        per_query["R@5"][m] = {q: recall(top[q], relevant[q]) for q in qids}
        per_query["all_relevant@5"][m] = {q: float(relevant[q] <= set(top[q])) for q in qids}
    by_type = {t: sorted(q for q in qids if types[q] == t) for t in sorted(set(types.values()))}
    summary = {
        m: {
            "mean_passages": float(np.mean([len(methods[m][q]) for q in qids])),
            **{
                f"{metric}{'' if t is None else '_' + t}": float(
                    np.mean([per_query[metric][m][q] for q in (qids if t is None else by_type[t])])
                )
                for metric in per_query
                for t in (None, *by_type)
            },
        }
        for m in methods
    }
    significance = {
        metric: [c.to_dict() for c in compare_all(per_query[metric], pairs=PLANNED)]
        for metric in per_query
    }
    comparison = by_type.get("comparison", [])
    comparison_check = [
        type_check(per_query["R@5"], comparison, "joint_bge", b) for b in ("fused_top5", BGE)
    ]
    passes = comparison_check[0]["mean_diff"] >= 0

    print(f"test {args.dataset}: {len(qids)} queries { ({t: len(v) for t, v in by_type.items()}) }")
    for m, s in summary.items():
        print(
            f"  {m:28s} R@5 {s['R@5']:.3f} (bridge {s['R@5_bridge']:.3f}, comparison "
            f"{s['R@5_comparison']:.3f})  all-relevant@5 {s['all_relevant@5']:.2f}"
        )
    stages = ", ".join(f"{k} {v['ms_mean']:.0f}" for k, v in latency.items() if k != "total")
    print(
        f"  joint_bge: {cfg}  pool {mean_pool:.1f} passages, ceiling {ceiling:.2f}, total "
        f"{latency['total']['ms_mean']:.0f} ms/query (p50 {latency['total']['ms_p50']:.0f}, "
        f"p95 {latency['total']['ms_p95']:.0f}; {stages})"
    )
    for metric, rows_ in significance.items():
        for c in rows_:
            print(
                f"  {metric} {c['a']} - {c['b']}: {c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, "
                f"{c['ci_high']:+.3f}] p_holm={c['p_holm']:.4f}"
            )
    for c in comparison_check:
        print(
            f"  comparison R@5 {c['method']} - {c['baseline']}: {c['mean_diff']:+.3f} "
            f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] worse {len(c['worse'])} better "
            f"{len(c['better'])}"
        )
    print(f"  comparison no-regression check vs fused_top5: {'pass' if passes else 'FAIL'}")

    record = {
        **metadata(),
        "dataset": args.dataset,
        "tune_dataset": args.tune_dataset,
        "rerank_source": str(args.rerank_result),
        "multihop_source": str(args.multihop_result),
        "test_queries": len(qids),
        "questions_by_type": {t: len(v) for t, v in by_type.items()},
        "protocol": {
            "k": K,
            "grid": GRID,
            "expansion": expansion,
            "scoring": "best bge score over the pools containing a candidate",
            "tuning_metric": "R@5",
            "tie_break": "smaller m, smaller depth, smaller N",
            "planned_pairs": PLANNED,
        },
        "reranker": vars(RERANKERS[RERANKER]),
        "embedding": data.embedding,
        "tuning": rows,
        "config": cfg,
        "summary": summary,
        "latency": latency,
        "mean_pool_size": mean_pool,
        "ceiling_all_relevant_in_pool": ceiling,
        "significance": significance,
        "comparison_check": comparison_check,
        "comparison_no_regression_vs_fused": passes,
        "per_query": per_query,
        "top5": {"joint_bge": methods["joint_bge"]},
    }
    path = args.out_dir / f"{args.dataset}-joint-{timestamp_slug()}.json"
    write_new(path, json.dumps(record, indent=1) + "\n")
    print(f"  saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
