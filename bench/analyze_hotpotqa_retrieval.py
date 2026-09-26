"""Where do HotpotQA retrieval failures sit? Ranks of missed gold passages, by question type.

    uv run python bench/analyze_hotpotqa_retrieval.py

Rebuilds exactly the retrieval of bench/eval_hotpotqa_beir.py (HybridIndex RRF, each retriever
contributing its top 100) over the subset, finds the questions whose 2 gold passages are not both
in the top 5, and for each missing gold passage reports its rank in the fused top 100 and in the
BM25-only and dense-only top 100 ("not found" if absent), split by bridge vs comparison.

It also reports the reranking ceiling: the share of questions whose gold passages are all inside
the fused top N, for several N. A reranker that reorders the top N can at best bring those
questions' gold passages into the top 5; passages outside the pool need better first-stage or
multi-hop retrieval instead.

Heuristic: `title_in_question` marks a gold passage whose title appears (case-insensitively) in
the question. In bridge questions the passage not named in the question is typically the second
hop, which the question alone may not retrieve.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new
from eval_hotpotqa import embed
from ir_eval import read_qrels

BUCKETS = [(6, 10), (11, 20), (21, 50), (51, 100)]
POOLS = [5, 10, 20, 50, 100]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def bucket(rank: int | None) -> str:
    if rank is None:
        return "not found"
    for lo, hi in BUCKETS:
        if lo <= rank <= hi:
            return f"{lo}-{hi}"
    return "1-5"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subset", default="hotpotqa-subset-n100-seed0-bg20000")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "rag")
    args = parser.parse_args(argv)

    import strata

    data = REPO_ROOT / "data" / "beir" / args.subset
    corpus = load_jsonl(data / "corpus.jsonl")
    queries = load_jsonl(data / "queries.jsonl")
    answers = {a["_id"]: a for a in load_jsonl(data / "answers.jsonl")}
    qrels = read_qrels(data / "qrels" / "test.tsv")
    title = {d["_id"]: d["title"] for d in corpus}

    doc_vecs, query_vecs, _ = embed(
        [f"{d['title']} {d['text']}".strip() for d in corpus],
        [q["text"] for q in queries],
        REPO_ROOT / "data" / "embeddings" / args.subset,
    )
    index = strata.HybridIndex(doc_vecs.shape[1], metric="ip")
    index.add([d["_id"] for d in corpus], [f"{d['title']}\n{d['text']}" for d in corpus], doc_vecs)
    texts = [q["text"] for q in queries]
    rankings = {}
    for method in ("rrf", "bm25", "dense"):
        ids, _ = index.search(texts, query_vecs, 100, method=method, candidates=100)
        rankings[method] = {
            q["_id"]: [index.doc_id(i) for i in ids[r] if i >= 0] for r, q in enumerate(queries)
        }

    def rank(method: str, qid: str, doc: str) -> int | None:
        ranked = rankings[method][qid]
        return ranked.index(doc) + 1 if doc in ranked else None

    failures = []
    for q in queries:
        qid = q["_id"]
        top = set(rankings["rrf"][qid][: args.k])
        if set(qrels[qid]) <= top:
            continue
        for doc in sorted(set(qrels[qid]) - top):
            failures.append(
                {
                    "query_id": qid,
                    "type": answers[qid]["type"],
                    "question": q["text"],
                    "missing_doc": doc,
                    "missing_title": title[doc],
                    "title_in_question": title[doc].lower() in q["text"].lower(),
                    "other_gold_in_top_k": len(set(qrels[qid]) & top) > 0,
                    "rank_fused": rank("rrf", qid, doc),
                    "rank_bm25": rank("bm25", qid, doc),
                    "rank_dense": rank("dense", qid, doc),
                }
            )

    failed_queries = {f["query_id"] for f in failures}
    types = sorted({a["type"] for a in answers.values()})
    by_type = {t: [f for f in failures if f["type"] == t] for t in types}
    summary = {
        "questions": len(queries),
        "questions_by_type": dict(Counter(a["type"] for a in answers.values())),
        "failed_questions": len(failed_queries),
        "failed_questions_by_type": dict(Counter(answers[q]["type"] for q in failed_queries)),
        "missing_passages": len(failures),
        "missing_by_type": {t: len(v) for t, v in by_type.items()},
        "fused_rank_buckets": {
            t: dict(Counter(bucket(f["rank_fused"]) for f in v)) for t, v in by_type.items()
        },
        "title_in_question": {
            t: dict(
                Counter("named in question" if f["title_in_question"] else "not named" for f in v)
            )
            for t, v in by_type.items()
        },
        "not_found_in_any_top100": sum(
            f["rank_fused"] is None and f["rank_bm25"] is None and f["rank_dense"] is None
            for f in failures
        ),
    }
    # Reranking ceiling: share of questions with all gold passages inside the fused top N.
    ceiling = defaultdict(dict)
    for t in [*types, "all"]:
        qs = [q["_id"] for q in queries if t == "all" or answers[q["_id"]]["type"] == t]
        for n in POOLS:
            ok = sum(set(qrels[q]) <= set(rankings["rrf"][q][:n]) for q in qs)
            ceiling[t][f"top{n}"] = {"questions": ok, "of": len(qs), "share": ok / len(qs)}
    summary["all_gold_within_fused_top_n"] = ceiling
    # Candidate pools for a reranker: the union of BM25 top N and dense top N (RRF can bury a
    # passage that only one retriever finds), with the pool size it implies.
    union = defaultdict(dict)
    for t in [*types, "all"]:
        qs = [q["_id"] for q in queries if t == "all" or answers[q["_id"]]["type"] == t]
        for n in POOLS:
            pools = {q: set(rankings["bm25"][q][:n]) | set(rankings["dense"][q][:n]) for q in qs}
            ok = sum(set(qrels[q]) <= pools[q] for q in qs)
            size = sum(len(p) for p in pools.values()) / len(qs)
            union[t][f"top{n}"] = {
                "questions": ok,
                "of": len(qs),
                "share": ok / len(qs),
                "mean_pool_size": size,
            }
    summary["all_gold_within_union_bm25_dense_top_n"] = union

    print(
        f"{summary['failed_questions']} of {len(queries)} questions miss a gold passage in the "
        f"top {args.k}: {summary['failed_questions_by_type']}; {len(failures)} missing passages"
    )
    for t in types:
        print(
            f"  {t}: ranks of missing passages in fused top 100: {summary['fused_rank_buckets'][t]}"
            f"; {summary['title_in_question'][t]}"
        )
    missing_fused = [f for f in failures if f["rank_fused"] is None]
    print(
        f"  missing from the fused top 100: {len(missing_fused)}; of those, BM25 top 100 ranks "
        f"{[f['rank_bm25'] for f in missing_fused]}, dense top 100 ranks "
        f"{[f['rank_dense'] for f in missing_fused]}"
    )
    print("  all gold passages within fused top N (reranking ceiling for both-gold@5):")
    for t, row in ceiling.items():
        cells = ", ".join(f"N={n[3:]} {v['questions']}/{v['of']}" for n, v in row.items())
        print(f"    {t:10s} {cells}")

    print("  all gold within union of BM25 top N and dense top N (mean pool size):")
    for t, row in union.items():
        cells = ", ".join(
            f"N={n[3:]} {v['questions']}/{v['of']} ({v['mean_pool_size']:.0f})"
            for n, v in row.items()
        )
        print(f"    {t:10s} {cells}")

    path = args.out_dir / f"hotpotqa-retrieval-failures-{args.subset}-{timestamp_slug()}.json"
    write_new(
        path,
        json.dumps(
            {
                **metadata(),
                "subset": args.subset,
                "k": args.k,
                "summary": summary,
                "failures": failures,
            },
            indent=1,
        )
        + "\n",
    )
    print(f"  saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
