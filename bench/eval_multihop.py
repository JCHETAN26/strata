"""Two-hop retrieval on BEIR-HotpotQA subsets: quality, per-query latency, and a per-type check.

    uv run python bench/eval_multihop.py \\
        --rerank-result results/rerank/hotpotqa-subset-n100-seed0-bg20000-<stamp>.json

Protocol (pre-declared; the method itself is described in rag/multihop.py):
- Hop 1: the fused (RRF) hybrid ranking, as in the current pipeline.
- Hop-2 queries: question + expansion of each of the top m hop-1 passages, embedded with the
  pinned bge-small-en-v1.5 and its query instruction; BM25 and dense retrieval each return the
  top `depth` passages. Passages whose id equals the query id are removed.
- Tuning on the dev subset only (--tune-dataset), by R@5; ties go to the cheaper configuration
  (smaller m, smaller depth, shorter expansion, then larger keep, i.e. more of hop 1 kept):
  stage 1 (no model): m in {1,2,3} x expansion in {title, title_first_sentence, full} x depth in
  {5,10,20} x keep in {1,2,3,4}; stage 2 (bge-reranker-base orders hop-2 candidates using the
  hop-2 query): m and expansion fixed to stage 1's choice, depth in {5,10} x keep in {1,2,3,4}.
  Stage 2 is restricted because the cross-encoder costs ~90 ms per pair on the M2 CPU.
- Test methods: multihop_no_model and multihop_bge (each stage's choice) against fused_top5,
  union_top5_no_model and rerank_bge-reranker-base (top-5 lists from the --rerank-result file).
  Ceilings: the single-hop bge pool ceiling from that file, and the two-hop ceiling (all gold
  passages in hop-1 top 5 or the hop-2 candidate pool).
- Metrics: R@5 (for the union baseline, recall of its ~7 passages) and all-relevant@5; per
  question type (bridge / comparison). Planned comparisons, Holm-adjusted as one family:
  multihop_bge vs fused_top5, vs rerank_bge; multihop_no_model vs fused_top5, vs union_top5.
- Comparison-question check: on comparison questions, each multi-hop method's mean R@5 difference
  from fused_top5 and from rerank_bge (bootstrap CI), and the questions that lost a gold passage.
  It passes if the mean difference from fused_top5 is >= 0.
- Latency: each test query run alone end to end (hop-1 search, hop-2 query embedding, hop-2
  searches, reranking), wall-clock ms: mean / p50 / p95 and per stage. The original question's
  embedding is precomputed for every method and excluded.

Saves results/multihop/<dataset>-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new
from eval_rerank import Dataset, load_jsonl, recall
from ir_eval import read_qrels
from significance import compare_all, paired_bootstrap_ci

sys.path.insert(0, str(REPO_ROOT / "rag"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from multihop import EXPANSIONS, MultiHopConfig, combine, hop2_candidates, hop2_query
from rerank import RERANKERS, Reranker, union_pool

K = 5
RERANKER = "bge-reranker-base"
BGE = f"rerank_{RERANKER}"
STAGE1 = {"m": [1, 2, 3], "expansion": list(EXPANSIONS), "depth": [5, 10, 20], "keep": [1, 2, 3, 4]}
STAGE2 = {"depth": [5, 10], "keep": [1, 2, 3, 4]}
PLANNED = [
    ("multihop_bge", "fused_top5"),
    ("multihop_bge", BGE),
    ("multihop_no_model", "fused_top5"),
    ("multihop_no_model", "union_top5_no_model"),
]
Encoder = Callable[[list[str]], np.ndarray]


def bge_query_encoder() -> Encoder:
    """Pinned bge-small with its query instruction, L2-normalized (as for the original queries)."""
    from embed_beir import MODELS, format_queries
    from eval_hotpotqa import EMBED_MODEL
    from sentence_transformers import SentenceTransformer

    spec = MODELS[EMBED_MODEL]
    model = SentenceTransformer(spec.hf_name, revision=spec.revision, device="cpu")
    return lambda texts: np.asarray(
        model.encode(format_queries(spec, texts), normalize_embeddings=True), dtype=np.float32
    )


class Hops:
    """Hop-1 rankings and hop-2 retrieval for one dataset."""

    def __init__(self, data: Dataset, encode: Encoder) -> None:
        self.data = data
        self.encode = encode
        self.doc = {d["_id"]: d for d in data.corpus}

    def hop1(self, qids: list[str], depth: int) -> dict[str, dict[str, list[str]]]:
        return self.data.rankings(qids, depth)

    def queries(self, q: str, hop1: list[str], m: int, expansion: str) -> list[str]:
        text = self.data.query_text[q]
        return [
            hop2_query(text, self.doc[p]["title"], self.doc[p]["sentences"], expansion)
            for p in hop1[:m]
        ]

    def search(
        self, qids: list[str], texts: list[str], vectors: np.ndarray, depth: int
    ) -> list[tuple[list[str], list[str]]]:
        """(BM25 top depth, dense top depth) per hop-2 query; qids[i] is removed from row i."""
        out: dict[str, list[list[str]]] = {}
        for method in ("bm25", "dense"):
            ids, _ = self.data.index.search(texts, vectors, depth + 1, method=method)
            out[method] = [
                [
                    self.data.index.doc_id(i)
                    for i in row
                    if i >= 0 and self.data.index.doc_id(i) != q
                ][:depth]
                for row, q in zip(ids, qids, strict=True)
            ]
        return list(zip(out["bm25"], out["dense"], strict=True))


def retrieve_hop2(
    hops: Hops,
    qids: list[str],
    hop1: dict[str, list[str]],
    m: int,
    expansions: list[str],
    depth: int,
) -> dict[tuple[str, str], list[tuple[list[str], list[str]]]]:
    """Batched hop-2 lists for tuning: (q, expansion) -> one (bm25, dense) pair per p in top m."""
    keys, texts, owners = [], [], []
    for q in qids:
        for e in expansions:
            qs = hops.queries(q, hop1[q], m, e)
            keys.append((q, e, len(qs)))
            texts += qs
            owners += [q] * len(qs)
    lists = hops.search(owners, texts, hops.encode(texts), depth)
    out, i = {}, 0
    for q, e, n in keys:
        out[(q, e)] = lists[i : i + n]
        i += n
    return out


def score_hop2(
    hops: Hops,
    reranker: Reranker,
    q: str,
    hop1: list[str],
    lists: list[tuple[list[str], list[str]]],
    expansion: str,
    depth: int,
) -> tuple[list[dict[str, float]], float]:
    """Cross-encoder scores of each p's hop-2 pool against that p's hop-2 query; seconds."""
    scores, seconds = [], 0.0
    for query, (bm25, dense) in zip(
        hops.queries(q, hop1, len(lists), expansion), lists, strict=True
    ):
        pool = union_pool([bm25, dense], depth)
        s, sec = reranker.score(query, [hops.data.passage[d] for d in pool])
        scores.append(dict(zip(pool, s.tolist(), strict=True)))
        seconds += sec
    return scores, seconds


def choose(rows: list[tuple[MultiHopConfig, float]]) -> MultiHopConfig:
    """Highest R@5; ties to smaller m, smaller depth, shorter expansion, larger keep."""
    return max(
        rows,
        key=lambda r: (r[1], -r[0].m, -r[0].depth, -EXPANSIONS.index(r[0].expansion), r[0].keep),
    )[0]


def tune(
    hops: Hops, qids: list[str], relevant: dict[str, set[str]], reranker: Reranker
) -> tuple[MultiHopConfig, MultiHopConfig, dict]:
    hop1 = hops.hop1(qids, K)["rrf"]
    m_max, d_max = max(STAGE1["m"]), max(STAGE1["depth"])
    lists = retrieve_hop2(hops, qids, hop1, m_max, STAGE1["expansion"], d_max)
    stage1 = []
    for m, e, d, keep in itertools.product(*STAGE1.values()):
        cfg = MultiHopConfig(m=m, expansion=e, depth=d, rerank=False, keep=keep)
        top = {q: combine(hop1[q], hop2_candidates(lists[(q, e)][:m], d), keep, K) for q in qids}
        stage1.append((cfg, float(np.mean([recall(top[q], relevant[q]) for q in qids]))))
    no_model = choose(stage1)
    print(f"stage 1 ({len(stage1)} configs, {len(qids)} queries): {no_model} ", end="")
    print(f"R@5 {dict((c, v) for c, v in stage1)[no_model]:.4f}")

    m, e, d_max = no_model.m, no_model.expansion, max(STAGE2["depth"])
    scores = {}
    for q in qids:
        scores[q], _ = score_hop2(hops, reranker, q, hop1[q], lists[(q, e)][:m], e, d_max)
    stage2 = []
    for d, keep in itertools.product(*STAGE2.values()):
        cfg = MultiHopConfig(m=m, expansion=e, depth=d, rerank=True, keep=keep)
        top = {
            q: combine(hop1[q], hop2_candidates(lists[(q, e)][:m], d, scores[q]), keep, K)
            for q in qids
        }
        stage2.append((cfg, float(np.mean([recall(top[q], relevant[q]) for q in qids]))))
    with_model = choose(stage2)
    print(f"stage 2 ({len(stage2)} configs): {with_model} ", end="")
    print(f"R@5 {dict((c, v) for c, v in stage2)[with_model]:.4f}")
    table = {
        "stage1": [{**c.to_dict(), "R@5": v} for c, v in stage1],
        "stage2": [{**c.to_dict(), "R@5": v} for c, v in stage2],
        "hop1_R@5": float(np.mean([recall(hop1[q][:K], relevant[q]) for q in qids])),
    }
    return no_model, with_model, table


def run_one(
    hops: Hops, q: str, cfg: MultiHopConfig, reranker: Reranker | None
) -> tuple[list[str], list[str], set[str], dict[str, float]]:
    """One query end to end: top K, hop-1 top K, hop-1 top K + hop-2 pool (for the ceiling), and
    ms per stage."""
    data = hops.data
    t0 = time.perf_counter()
    ids, _ = data.index.search(
        [data.query_text[q]], data.query_vec[q][None, :], K + 1, method="rrf"
    )
    hop1 = [data.index.doc_id(i) for i in ids[0] if i >= 0 and data.index.doc_id(i) != q][:K]
    t1 = time.perf_counter()
    texts = hops.queries(q, hop1, cfg.m, cfg.expansion)
    vectors = hops.encode(texts)
    t2 = time.perf_counter()
    lists = hops.search([q] * len(texts), texts, vectors, cfg.depth)
    t3 = time.perf_counter()
    scores = None
    if cfg.rerank:
        assert reranker is not None
        scores, _ = score_hop2(hops, reranker, q, hop1, lists, cfg.expansion, cfg.depth)
    hop2 = hop2_candidates(lists, cfg.depth, scores)
    top = combine(hop1, hop2, cfg.keep, K)
    t4 = time.perf_counter()
    ms = {
        "hop1_search": (t1 - t0) * 1000,
        "hop2_embed": (t2 - t1) * 1000,
        "hop2_search": (t3 - t2) * 1000,
        "rerank_and_combine": (t4 - t3) * 1000,
        "total": (t4 - t0) * 1000,
    }
    return top, hop1, set(hop1) | set(hop2), ms


def latency_summary(ms: dict[str, dict[str, float]]) -> dict:
    out = {}
    for stage in next(iter(ms.values())):
        x = np.array([v[stage] for v in ms.values()])
        out[stage] = {
            "ms_mean": float(x.mean()),
            "ms_p50": float(np.percentile(x, 50)),
            "ms_p95": float(np.percentile(x, 95)),
        }
    return out


def type_check(
    per_query: dict[str, dict[str, float]], qids: list[str], method: str, baseline: str
) -> dict:
    a = [per_query[method][q] for q in qids]
    b = [per_query[baseline][q] for q in qids]
    lo, hi = paired_bootstrap_ci(a, b)
    return {
        "method": method,
        "baseline": baseline,
        "questions": len(qids),
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "ci_low": lo,
        "ci_high": hi,
        "worse": sorted(q for q, x, y in zip(qids, a, b, strict=True) if x < y),
        "better": sorted(q for q, x, y in zip(qids, a, b, strict=True) if x > y),
    }


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
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "multihop")
    args = parser.parse_args(argv)
    if args.tune_dataset == args.dataset:
        raise SystemExit("tune on a different (dev) subset than the test subset")
    baseline = json.loads(args.rerank_result.read_text())
    if baseline["dataset"] != args.dataset:
        raise SystemExit(f"{args.rerank_result} is for {baseline['dataset']}, not {args.dataset}")
    reranker = reranker or Reranker(RERANKER)
    encode = encode or bge_query_encoder()

    def load(name: str, split: str) -> tuple[Hops, list[str], dict[str, set[str]], dict]:
        data = Dataset(name)
        qrels = read_qrels(data.dir / "qrels" / f"{split}.tsv")
        qids = sorted(q for q in qrels if any(g > 0 for g in qrels[q].values()))
        relevant = {q: {d for d, g in qrels[q].items() if g > 0} for q in qids}
        types = {a["_id"]: a["type"] for a in load_jsonl(data.dir / "answers.jsonl")}
        return Hops(data, encode), qids, relevant, types

    # --- Tuning (dev subset only).
    tune_hops, tune_qids, tune_relevant, _ = load(args.tune_dataset, "dev")
    no_model, with_model, tuning = tune(tune_hops, tune_qids, tune_relevant, reranker)
    configs = {"multihop_no_model": no_model, "multihop_bge": with_model}

    # --- Test.
    hops, qids, relevant, types = load(args.dataset, "test")
    ranks = hops.hop1(qids, K)
    methods: dict[str, dict[str, list[str]]] = {
        "fused_top5": {q: ranks["rrf"][q][:K] for q in qids},
        "union_top5_no_model": {
            q: union_pool([ranks["bm25"][q], ranks["dense"][q]], K) for q in qids
        },
        BGE: {q: baseline["top5"][BGE][q] for q in qids},
    }
    for m in ("fused_top5", "union_top5_no_model"):  # the baselines must be the same lists
        if methods[m] != {q: baseline["top5"][m][q] for q in qids}:
            raise SystemExit(f"{m} differs from {args.rerank_result}")
    latency, ceiling = {}, {}
    for name, cfg in configs.items():
        methods[name], pools, ms = {}, {}, {}
        for q in qids:
            methods[name][q], hop1, pools[q], ms[q] = run_one(hops, q, cfg, reranker)
            if hop1 != ranks["rrf"][q][:K]:  # the per-query path must match the batched one
                raise SystemExit(f"{q}: per-query hop 1 differs from the batched ranking")
        latency[name] = latency_summary(ms)
        ceiling[name] = float(np.mean([relevant[q] <= pools[q] for q in qids]))

    per_query: dict[str, dict[str, dict[str, float]]] = {"R@5": {}, "all_relevant@5": {}}
    for m, found in methods.items():
        per_query["R@5"][m] = {q: recall(found[q], relevant[q]) for q in qids}
        per_query["all_relevant@5"][m] = {q: float(relevant[q] <= set(found[q])) for q in qids}
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
        type_check(per_query["R@5"], comparison, m, b) for m in configs for b in ("fused_top5", BGE)
    ]
    passes = all(c["mean_diff"] >= 0 for c in comparison_check if c["baseline"] == "fused_top5")

    print(f"test {args.dataset}: {len(qids)} queries { ({t: len(v) for t, v in by_type.items()}) }")
    for m, s in summary.items():
        print(
            f"  {m:28s} R@5 {s['R@5']:.3f} (bridge {s['R@5_bridge']:.3f}, comparison "
            f"{s['R@5_comparison']:.3f})  all-relevant@5 {s['all_relevant@5']:.2f}  "
            f"({s['mean_passages']:.1f} passages)"
        )
    for name, lat in latency.items():
        stages = ", ".join(f"{k} {v['ms_mean']:.0f}" for k, v in lat.items() if k != "total")
        print(
            f"  {name}: {configs[name]}  ceiling {ceiling[name]:.2f}  total "
            f"{lat['total']['ms_mean']:.0f} ms/query (p50 {lat['total']['ms_p50']:.0f}, p95 "
            f"{lat['total']['ms_p95']:.0f}; {stages})"
        )
    print(f"  single-hop bge pool ceiling {baseline['ceiling_all_relevant_in_pool'][RERANKER]:.2f}")
    for metric, rows in significance.items():
        for c in rows:
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
        "rerank_source": str(args.rerank_result.relative_to(REPO_ROOT))
        if args.rerank_result.is_absolute()
        else str(args.rerank_result),
        "test_queries": len(qids),
        "questions_by_type": {t: len(v) for t, v in by_type.items()},
        "protocol": {
            "k": K,
            "hop1": "rrf",
            "stage1_grid": STAGE1,
            "stage2_grid": STAGE2,
            "tuning_metric": "R@5",
            "tie_break": "smaller m, smaller depth, shorter expansion, larger keep",
            "planned_pairs": PLANNED,
        },
        "reranker": vars(RERANKERS[RERANKER]),
        "embedding": hops.data.embedding,
        "tuning": tuning,
        "configs": {m: c.to_dict() for m, c in configs.items()},
        "summary": summary,
        "latency": latency,
        "baseline_latency": baseline.get("latency"),
        "ceiling_all_relevant_in_pool": {
            **ceiling,
            f"single_hop_{RERANKER}": baseline["ceiling_all_relevant_in_pool"][RERANKER],
        },
        "significance": significance,
        "comparison_check": comparison_check,
        "comparison_no_regression_vs_fused": passes,
        "per_query": per_query,
        "top5": {m: methods[m] for m in configs},
    }
    path = args.out_dir / f"{args.dataset}-{timestamp_slug()}.json"
    write_new(path, json.dumps(record, indent=1) + "\n")
    print(f"  saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
