"""Cross-encoder reranking on BEIR-format data: quality, per-query latency, and the ceiling.

    uv run python bench/eval_rerank.py --dataset hotpotqa-subset-n100-seed0-bg20000 \\
        --tune-dataset hotpotqa-dev-subset-n100-seed0-bg20000 --tune-split dev
    uv run python bench/eval_rerank.py --dataset scifact --tune-dataset scifact --tune-split train

Protocol (pre-declared):
- Candidates: HybridIndex over the dataset: BM25 top N and dense (bge-small-en-v1.5, pinned)
  top N; the reranking pool is their union (rag/rerank.py union_pool).
- Pool depth N is chosen per model on the *tuning* split only (a dev-split subset for HotpotQA;
  SciFact's train split, capped at --tune-max-queries seeded queries), from --pools, by R@5 (the
  share of relevant passages among the 5 a generator would see); ties go to the smaller N.
- Test methods: fused top 5 (RRF, the current pipeline); union top 5 without a model (BM25 top 5
  and dense top 5, ~7 passages, unranked); each reranker's top 5 over its chosen pool. Also the
  ceiling: queries with every relevant passage inside the chosen pool.
- Metrics per query: R@5 (for the union baseline, recall of its whole ~7-passage set, with the
  set size reported), all-relevant@5 (both gold passages for HotpotQA), nDCG@10 for ranked
  methods. Paired bootstrap CIs and randomization tests between methods (bench/significance.py).
- Latency: wall-clock seconds to score each query's pool on this machine's CPU, reported as
  mean / p50 / p95 per query with the mean pool size and torch thread count.
- Passages whose id equals the query id are removed (BEIR ignore_identical_ids).

Scores are computed once per query on the deepest pool and reused for smaller pools (subsets).
Saves results/rerank/<dataset>-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new
from ir_eval import ndcg_at_k, read_qrels
from significance import compare_all

sys.path.insert(0, str(REPO_ROOT / "rag"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from rerank import RERANKERS, Reranker, rerank, union_pool

K = 5
EMBED_MODEL = "bge-small-en-v1.5"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


class Dataset:
    """A BEIR-format dataset with a HybridIndex and dense query vectors."""

    def __init__(self, name: str) -> None:
        import strata
        from eval_hotpotqa import embed
        from prepare_datasets import read_bin

        self.name = name
        self.dir = REPO_ROOT / "data" / "beir" / name
        self.corpus = load_jsonl(self.dir / "corpus.jsonl")
        self.query_text = {q["_id"]: q["text"] for q in load_jsonl(self.dir / "queries.jsonl")}
        self.passage = {d["_id"]: f"{d.get('title', '')} {d['text']}".strip() for d in self.corpus}
        emb = REPO_ROOT / "data" / "embeddings" / name / EMBED_MODEL
        if (emb / "meta.json").exists():  # full BEIR datasets: scripts/embed_beir.py output
            docs = read_bin(emb / "corpus.fbin", np.float32)
            queries = read_bin(emb / "queries.fbin", np.float32)
            qids = json.loads((emb / "query_ids.json").read_text())
            self.query_vec = {q: queries[i] for i, q in enumerate(qids)}
            self.embedding = json.loads((emb / "meta.json").read_text())
        else:  # subsets: embedded (and cached) on demand
            qids = sorted(self.query_text)
            docs, queries, self.embedding = embed(
                [self.passage[d["_id"]] for d in self.corpus],
                [self.query_text[q] for q in qids],
                REPO_ROOT / "data" / "embeddings" / name,
            )
            self.query_vec = {q: queries[i] for i, q in enumerate(qids)}
        self.index = strata.HybridIndex(docs.shape[1], metric="ip")
        self.index.add(
            [d["_id"] for d in self.corpus],
            [f"{d.get('title', '')}\n{d['text']}" for d in self.corpus],
            docs,
        )

    def rankings(self, qids: list[str], depth: int) -> dict[str, dict[str, list[str]]]:
        """Per method ("bm25", "dense", "rrf"), per query: ranked ids, query id removed."""
        texts = [self.query_text[q] for q in qids]
        vectors = np.stack([self.query_vec[q] for q in qids])
        out = {}
        for method in ("bm25", "dense", "rrf"):
            ids, _ = self.index.search(texts, vectors, depth + 1, method=method, candidates=100)
            out[method] = {
                q: [self.index.doc_id(i) for i in ids[r] if i >= 0 and self.index.doc_id(i) != q][
                    :depth
                ]
                for r, q in enumerate(qids)
            }
        return out


def recall(found: list[str], relevant: set[str]) -> float:
    return len(relevant & set(found)) / len(relevant)


def score_pools(
    data: Dataset, qids: list[str], ranks: dict, reranker: Reranker, depth: int
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Cross-encoder scores for every passage in each query's deepest pool, and scoring time."""
    scores, seconds = {}, {}
    for q in qids:
        pool = union_pool([ranks["bm25"][q], ranks["dense"][q]], depth)
        s, sec = reranker.score(data.query_text[q], [data.passage[d] for d in pool])
        scores[q] = dict(zip(pool, s.tolist(), strict=True))
        seconds[q] = sec
    return scores, seconds


def main(argv: list[str] | None = None, rerankers: dict[str, Reranker] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--tune-dataset", required=True)
    parser.add_argument("--tune-split", required=True)
    parser.add_argument("--tune-max-queries", type=int, default=300)
    parser.add_argument("--models", nargs="+", default=sorted(RERANKERS))
    parser.add_argument("--pools", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for the cross-encoder: cpu, cuda, or mps. Recorded per model in the "
        "latency block, so CPU and GPU runs are directly comparable.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="cross-encoder batch size")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "rerank")
    args = parser.parse_args(argv)
    import torch

    rerankers = rerankers or {
        m: Reranker(m, device=args.device, batch_size=args.batch_size) for m in args.models
    }
    depth = max(args.pools)

    # --- Tuning: choose each model's pool depth on the tuning split only.
    tune = Dataset(args.tune_dataset)
    tune_qrels = read_qrels(tune.dir / "qrels" / f"{args.tune_split}.tsv")
    tune_qids = sorted(q for q in tune_qrels if any(g > 0 for g in tune_qrels[q].values()))
    if len(tune_qids) > args.tune_max_queries:
        rng = np.random.default_rng(args.seed)
        tune_qids = sorted(rng.choice(tune_qids, args.tune_max_queries, replace=False).tolist())
    test = tune if args.tune_dataset == args.dataset else Dataset(args.dataset)
    test_qrels = read_qrels(test.dir / "qrels" / f"{args.test_split}.tsv")
    test_qids = sorted(q for q in test_qrels if any(g > 0 for g in test_qrels[q].values()))
    if args.tune_dataset == args.dataset and set(tune_qids) & set(test_qids):
        raise SystemExit("tuning and test queries overlap")
    relevant = {
        **{q: {d for d, g in tune_qrels[q].items() if g > 0} for q in tune_qids},
        **{q: {d for d, g in test_qrels[q].items() if g > 0} for q in test_qids},
    }

    tune_ranks = tune.rankings(tune_qids, depth)
    tuning, chosen = {}, {}
    for name, model in rerankers.items():
        scores, _ = score_pools(tune, tune_qids, tune_ranks, model, depth)
        rows = {}
        for n in args.pools:
            top = {
                q: rerank(
                    union_pool([tune_ranks["bm25"][q], tune_ranks["dense"][q]], n), scores[q], K
                )
                for q in tune_qids
            }
            rows[n] = float(np.mean([recall(top[q], relevant[q]) for q in tune_qids]))
        chosen[name] = max(args.pools, key=lambda n: (rows[n], -n))
        tuning[name] = {"R@5_by_pool": {str(n): v for n, v in rows.items()}, "chosen": chosen[name]}
        print(
            f"tune {name} on {args.tune_dataset}/{args.tune_split} ({len(tune_qids)} queries): "
            f"R@5 by N { ({n: round(v, 4) for n, v in rows.items()}) } -> N={chosen[name]}"
        )

    # --- Test.
    ranks = test.rankings(test_qids, max(depth, 100))
    methods: dict[str, dict[str, list[str]]] = {
        "fused_top5": {q: ranks["rrf"][q][:K] for q in test_qids},
        "union_top5_no_model": {
            q: union_pool([ranks["bm25"][q], ranks["dense"][q]], K) for q in test_qids
        },
    }
    ranked_lists = {"fused_top5": {q: ranks["rrf"][q] for q in test_qids}}
    latency, ceiling = {}, {}
    for name, model in rerankers.items():
        n = chosen[name]
        pools = {q: union_pool([ranks["bm25"][q], ranks["dense"][q]], n) for q in test_qids}
        scores, seconds = score_pools(test, test_qids, ranks, model, n)
        key = f"rerank_{name}"
        ranked_lists[key] = {q: rerank(pools[q], scores[q], len(pools[q])) for q in test_qids}
        methods[key] = {q: ranked_lists[key][q][:K] for q in test_qids}
        per_query = np.array([seconds[q] for q in test_qids]) * 1000
        latency[name] = {
            "pool_depth_N": n,
            "mean_pool_size": float(np.mean([len(pools[q]) for q in test_qids])),
            "ms_mean": float(per_query.mean()),
            "ms_p50": float(np.percentile(per_query, 50)),
            "ms_p95": float(np.percentile(per_query, 95)),
            "device": model.device,
            "batch_size": model.batch_size,
            "torch_threads": torch.get_num_threads(),
        }
        ceiling[name] = float(np.mean([relevant[q] <= set(pools[q]) for q in test_qids]))

    per_query = {"R@5": {}, "all_relevant@5": {}, "nDCG@10": {}}
    summary = {}
    qrels_rel = {q: {d: g for d, g in test_qrels[q].items()} for q in test_qids}
    for m, found in methods.items():
        per_query["R@5"][m] = {q: recall(found[q], relevant[q]) for q in test_qids}
        per_query["all_relevant@5"][m] = {q: float(relevant[q] <= set(found[q])) for q in test_qids}
        summary[m] = {
            "R@5": float(np.mean(list(per_query["R@5"][m].values()))),
            "all_relevant@5": float(np.mean(list(per_query["all_relevant@5"][m].values()))),
            "mean_passages": float(np.mean([len(found[q]) for q in test_qids])),
        }
        if m in ranked_lists:
            run = {q: {d: -float(i) for i, d in enumerate(ranked_lists[m][q])} for q in test_qids}
            ndcg = ndcg_at_k(run, qrels_rel, 10)
            per_query["nDCG@10"][m] = ndcg
            summary[m]["nDCG@10"] = float(np.mean(list(ndcg.values())))
    significance = {
        metric: [c.to_dict() for c in compare_all(per_query[metric])]
        for metric in ("R@5", "all_relevant@5", "nDCG@10")
    }

    print(f"test {args.dataset}/{args.test_split}: {len(test_qids)} queries")
    for m, s in summary.items():
        extra = f"  nDCG@10 {s['nDCG@10']:.4f}" if "nDCG@10" in s else ""
        print(
            f"  {m:28s} R@5 {s['R@5']:.4f}  all-relevant@5 {s['all_relevant@5']:.3f}  "
            f"({s['mean_passages']:.1f} passages){extra}"
        )
    for name, lat in latency.items():
        print(
            f"  {name}: N={lat['pool_depth_N']} pool {lat['mean_pool_size']:.1f} passages, "
            f"{lat['ms_mean']:.0f} ms/query (p50 {lat['ms_p50']:.0f}, p95 {lat['ms_p95']:.0f}), "
            f"ceiling all-relevant {ceiling[name]:.3f}"
        )
    for c in significance["R@5"]:
        print(
            f"  R@5 {c['a']} - {c['b']}: {c['mean_diff']:+.4f} [{c['ci_low']:+.4f}, "
            f"{c['ci_high']:+.4f}] p_holm={c['p_holm']:.4f}"
        )

    record = {
        **metadata(),
        "dataset": args.dataset,
        "test_split": args.test_split,
        "tune_dataset": args.tune_dataset,
        "tune_split": args.tune_split,
        "tune_queries": len(tune_qids),
        "test_queries": len(test_qids),
        "protocol": {
            "k": K,
            "pools": args.pools,
            "pool": "union of BM25 top N and dense top N",
            "tuning_metric": "R@5",
            "tie_break": "smaller N",
            "seed": args.seed,
        },
        "rerankers": {m: vars(RERANKERS[m]) for m in rerankers},
        "embedding": test.embedding,
        "tuning": tuning,
        "summary": summary,
        "latency": latency,
        "ceiling_all_relevant_in_pool": ceiling,
        "significance": significance,
        "per_query": per_query,
        # The passages each method would hand a generator (used by bench/eval_hotpotqa_beir.py).
        "top5": methods,
    }
    path = args.out_dir / f"{args.dataset}-{timestamp_slug()}.json"
    write_new(path, json.dumps(record, indent=1) + "\n")
    print(f"  saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
