"""HotpotQA in the BEIR setting: retrieval over BEIR passages, cited answers, groundedness.

    uv run python scripts/prepare_hotpotqa_beir.py --n 100 --seed 0 --background 20000
    uv run python bench/eval_hotpotqa_beir.py                      # cost estimate only (default)
    uv run python bench/eval_hotpotqa_beir.py --run --max-cost-usd 1.00

Data: a subset built by scripts/prepare_hotpotqa_beir.py (BEIR HotpotQA test queries, BEIR
passages: gold + HotpotQA distractors + a uniform background sample). On the IdeaPad the same
script runs against the full 5.2M-passage corpus. Subset retrieval numbers are not comparable to
full-corpus BEIR numbers.

Conditions (same questions, same generator):
- retrieved: HybridIndex top-k (RRF of BM25 and bge-small-en-v1.5 at a pinned revision) over
  the subset corpus, the passages Strata's pipeline would hand the generator;
- gold: the 2 qrels passages only (an oracle for retrieval), which isolates generation.

Generator: rag/answer.py (claude-haiku-4-5, temperature 0, sentence-level native citations,
responses cached by request hash). Metrics: official HotpotQA answer EM/F1, supporting-fact
P/R/F1/EM of the cited sentences, joint EM/F1, answer-in-citations, citation coverage,
abstention, retrieval nDCG@10 / R@k against BEIR qrels, and paired significance tests between
conditions.

Cost control: without --run the script only estimates. It builds every request, counts input
tokens (exactly, with the free count_tokens endpoint, when an API key is available; otherwise
the documented offline estimate), and prices expected and worst-case (max_tokens) output.
--run refuses to start if the worst-case cost of uncached requests exceeds --max-cost-usd.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new
from eval_hotpotqa import answer_in_citations, embed, summarize
from ir_eval import mean, ndcg_at_k, read_qrels, recall_at_k
from significance import compare_all

sys.path.insert(0, str(REPO_ROOT / "rag"))
from answer import (
    EXPECTED_OUTPUT_TOKENS,
    MAX_TOKENS,
    MODEL,
    PROMPT_VERSION,
    AnswerGenerator,
    Passage,
    build_request,
    cost_usd,
    estimate_input_tokens,
    load_env,
    request_chars,
)
from hotpot_metrics import score_example

# Condition names for methods taken from a bench/eval_rerank.py result.
CONDITION_NAMES = {
    "rerank_bge-reranker-base": "reranked_bge",
    "rerank_minilm-l6": "reranked_minilm",
    "union_top5_no_model": "union_top5",  # ~7 passages: BM25 top 5 + dense top 5, deduplicated
}
# Pre-declared comparisons (one Holm family): each alternative against the current pipeline
# ("retrieved", fused top 5), and the reranker against its zero-cost competitor.
PLANNED = [
    ("reranked_bge", "retrieved"),
    ("union_top5", "retrieved"),
    ("reranked_bge", "union_top5"),
]
SIG_METRICS = ("em", "f1", "sp_f1", "joint_f1")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main(argv: list[str] | None = None, generator: AnswerGenerator | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subset", default="hotpotqa-subset-n100-seed0-bg20000")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per question")
    parser.add_argument("--run", action="store_true", help="call the API (default: estimate only)")
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "beir")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "rag")
    parser.add_argument(
        "--rerank-result",
        type=Path,
        help="a bench/eval_rerank.py result on this subset: adds a 'reranked' condition using its "
        "saved top-5 passages",
    )
    parser.add_argument(
        "--rerank-methods",
        nargs="+",
        default=["rerank_bge-reranker-base", "union_top5_no_model"],
        help="methods from the rerank result to add as conditions",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="subset of conditions to estimate/run (default: all); cached ones cost nothing",
    )
    args = parser.parse_args(argv)

    import strata

    data = args.data_dir / args.subset
    meta = json.loads((data / "meta.json").read_text())
    corpus = load_jsonl(data / "corpus.jsonl")
    queries = load_jsonl(data / "queries.jsonl")
    answers = {a["_id"]: a for a in load_jsonl(data / "answers.jsonl")}
    qrels = read_qrels(data / "qrels" / "test.tsv")
    passages = {d["_id"]: Passage(d["_id"], d["title"], d["sentences"]) for d in corpus}

    # Retrieval over the subset corpus.
    doc_vecs, query_vecs, emb_meta = embed(
        [f"{d['title']} {d['text']}".strip() for d in corpus],
        [q["text"] for q in queries],
        REPO_ROOT / "data" / "embeddings" / args.subset,
    )
    index = strata.HybridIndex(doc_vecs.shape[1], metric="ip")
    index.add([d["_id"] for d in corpus], [f"{d['title']}\n{d['text']}" for d in corpus], doc_vecs)
    depth = max(args.k, 100)
    ids, scores = index.search([q["text"] for q in queries], query_vecs, depth, method="rrf")
    run = {
        q["_id"]: {
            index.doc_id(i): float(s) for i, s in zip(ids[r], scores[r], strict=True) if i >= 0
        }
        for r, q in enumerate(queries)
    }
    top_k = {
        q["_id"]: [index.doc_id(i) for i in ids[r][: args.k] if i >= 0]
        for r, q in enumerate(queries)
    }
    retrieval = {
        "corpus_passages": len(corpus),
        "k": args.k,
        "nDCG@10": mean(ndcg_at_k(run, qrels, 10)),
        f"R@{args.k}": mean(recall_at_k(run, qrels, args.k)),
        "R@100": mean(recall_at_k(run, qrels, 100)),
        f"both_gold@{args.k}": float(np.mean([set(qrels[q]) <= set(top_k[q]) for q in qrels])),
        "method": "HybridIndex rrf (k=60), BM25 + bge-small-en-v1.5; subset corpus",
    }
    print(f"{args.subset}: {len(queries)} queries, {len(corpus)} passages (subset corpus)")
    print(
        f"  retrieval: nDCG@10 {retrieval['nDCG@10']:.3f}  "
        f"R@{args.k} {retrieval[f'R@{args.k}']:.3f}  "
        f"R@100 {retrieval['R@100']:.3f}  both gold in top {args.k}: "
        f"{retrieval[f'both_gold@{args.k}']:.0%}"
    )

    conditions = {
        "retrieved": {q["_id"]: [passages[d] for d in top_k[q["_id"]]] for q in queries},
        "gold": {q["_id"]: [passages[d] for d in sorted(qrels[q["_id"]])] for q in queries},
    }
    if args.rerank_result is not None:
        rerank_record = json.loads(args.rerank_result.read_text())
        if rerank_record["dataset"] != args.subset:
            raise SystemExit(
                f"{args.rerank_result} is for {rerank_record['dataset']}, not {args.subset}"
            )
        rerank_source = {
            "result": args.rerank_result.name,
            "commit": rerank_record["git"]["commit"],
        }
        for method in args.rerank_methods:
            name = CONDITION_NAMES.get(method, method)
            chosen = rerank_record["top5"][method]
            conditions[name] = {q["_id"]: [passages[d] for d in chosen[q["_id"]]] for q in queries}
            rerank_source[name] = method
    else:
        rerank_source = None
    if args.conditions:
        conditions = {c: conditions[c] for c in args.conditions}
    question = {q["_id"]: q["text"] for q in queries}

    # Cost estimate, before any API call.
    load_env(REPO_ROOT)
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    injected = generator is not None  # tests pass a scripted generator
    if generator is None:
        generator = AnswerGenerator(cache_dir=REPO_ROOT / "data" / "cache" / "answers")
    counter = generator.client if have_key else None
    estimate = {}
    for name, per_q in conditions.items():
        tokens, method, uncached_in, uncached = 0, "estimate", 0, 0
        per_request = {}
        for qid, ps in per_q.items():
            request = build_request(question[qid], ps)
            n, method = estimate_input_tokens(request, counter)
            offline, _ = estimate_input_tokens(request)
            per_request[qid] = {
                "chars": request_chars(request),
                "passages": len(ps),
                "input_tokens": n,
                "method": method,
                "offline_estimate": offline,
            }
            tokens += n
            if not generator.is_cached(question[qid], ps):
                uncached += 1
                uncached_in += n
        estimate[name] = {
            "requests": len(per_q),
            "uncached_requests": uncached,
            "input_tokens": tokens,
            "input_token_method": method,
            "expected_cost_usd": cost_usd(uncached_in, uncached * EXPECTED_OUTPUT_TOKENS),
            "worst_case_cost_usd": cost_usd(uncached_in, uncached * MAX_TOKENS),
            "offline_estimate_tokens": sum(r["offline_estimate"] for r in per_request.values()),
            "per_request": per_request,
        }
    total_expected = sum(e["expected_cost_usd"] for e in estimate.values())
    total_worst = sum(e["worst_case_cost_usd"] for e in estimate.values())
    print(f"  cost estimate ({MODEL}, $1/MTok in, $5/MTok out; cached requests are free):")
    for name, e in estimate.items():
        print(
            f"    {name:9s} {e['uncached_requests']}/{e['requests']} uncached, "
            f"{e['input_tokens']:,} input tokens ({e['input_token_method']}), "
            f"expected ${e['expected_cost_usd']:.3f}, worst case ${e['worst_case_cost_usd']:.3f}"
        )
    print(f"    total: expected ${total_expected:.3f}, worst case ${total_worst:.3f}")

    if not args.run:
        print("  estimate only; re-run with --run --max-cost-usd <cap> to generate answers")
        return 0
    if total_worst > args.max_cost_usd:
        print(
            f"  worst case ${total_worst:.3f} exceeds --max-cost-usd {args.max_cost_usd}; "
            "not running",
            file=sys.stderr,
        )
        return 3

    if not have_key and not injected and any(e["uncached_requests"] for e in estimate.values()):
        print("  no ANTHROPIC_API_KEY (env or .env)", file=sys.stderr)
        return 2

    results: dict[str, dict] = {}
    start = time.perf_counter()
    for name, per_q in conditions.items():
        qids = [q["_id"] for q in queries]
        with ThreadPoolExecutor(args.workers) as pool:
            got = list(
                pool.map(lambda qid, ps=per_q: generator.answer(question[qid], ps[qid]), qids)
            )
        records = []
        for qid, a in zip(qids, got, strict=True):
            gold = answers[qid]
            records.append(
                {
                    "id": qid,
                    "type": gold["type"],
                    "gold_answer": gold["answer"],
                    "short_answer": a.short_answer,
                    "abstained": a.abstained,
                    "ok": a.ok,
                    "scores": score_example(
                        a.short_answer,
                        a.cited_sentences,
                        gold["answer"],
                        [tuple(f) for f in gold["supporting_facts"]],
                    ),
                    "cited_sentences": sorted(a.cited_sentences),
                    "num_citations": sum(len(s.citations) for s in a.statements),
                    "answer_in_citations": answer_in_citations(a),
                    "passages": [p.doc_id for p in per_q[qid]],
                    "input_tokens": a.input_tokens,
                    "output_tokens": a.output_tokens,
                    "cost_usd": a.cost_usd,
                    "from_cache": a.from_cache,
                    "text": a.text,
                    "stop_reason": a.stop_reason,
                    # Not a complete answer: truncated (max_tokens), refused, or other non-end_turn.
                    "flagged": a.stop_reason != "end_turn",
                }
            )
        complete = [r for r in records if not r["flagged"]]
        results[name] = {
            # Headline metrics: complete answers only. Flagged responses are listed, not scored.
            "summary": summarize(complete) if complete else {},
            # For transparency: every question, flagged ones scored as zero.
            "summary_flagged_as_zero": summarize(
                [
                    r if not r["flagged"] else {**r, "scores": dict.fromkeys(r["scores"], 0.0)}
                    for r in records
                ]
            ),
            "stop_reasons": {
                s: sum(r["stop_reason"] == s for r in records)
                for s in {r["stop_reason"] for r in records}
            },
            "flagged": [
                {
                    "id": r["id"],
                    "stop_reason": r["stop_reason"],
                    "output_tokens": r["output_tokens"],
                }
                for r in records
                if r["flagged"]
            ],
            "records": records,
        }
    seconds = time.perf_counter() - start
    # Paired tests on questions answered completely in every condition.
    flagged_ids = {f["id"] for c in conditions for f in results[c]["flagged"]}
    per_question = {
        m: {
            c: {
                r["id"]: r["scores"][m] for r in results[c]["records"] if r["id"] not in flagged_ids
            }
            for c in conditions
        }
        for m in SIG_METRICS
    }
    significance = {m: [x.to_dict() for x in compare_all(pq)] for m, pq in per_question.items()}
    planned = [(a, b) for a, b in PLANNED if a in conditions and b in conditions]
    planned_comparisons = (
        {m: [x.to_dict() for x in compare_all(pq, pairs=planned)] for m, pq in per_question.items()}
        if planned
        else {}
    )
    for name, r in results.items():
        print(
            f"  {name:12s} stop reasons {r['stop_reasons']}; flagged (not scored): "
            f"{[f['id'] for f in r['flagged']] or 'none'}"
        )
        s = r["summary"]
        print(
            f"  {name:12s} EM {s['em']:.3f}  F1 {s['f1']:.3f}  | cited SP F1 {s['sp_f1']:.3f} "
            f"(P {s['sp_prec']:.3f}, R {s['sp_recall']:.3f})  | joint F1 {s['joint_f1']:.3f}  "
            f"| abstain {s['abstention_rate']:.0%}, answered EM {s['answered_em']:.3f} "
            f"F1 {s['answered_f1']:.3f} (n={s['answered']})  | "
            f"${s['cost_usd']:.3f} ({s['from_cache']} of {s['questions']} from cache: no new spend)"
        )

    for m, pairs in planned_comparisons.items():
        for c in pairs:
            print(
                f"  planned {m:8s} {c['a']} - {c['b']}: {c['mean_diff']:+.3f} "
                f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] p={c['p_value']:.4f} "
                f"p_holm={c['p_holm']:.4f} W/L/T {c['wins']}/{c['losses']}/{c['ties']}"
            )

    slug = timestamp_slug()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        **metadata(),
        "setting": "beir",
        "subset": args.subset,
        "dataset": meta,
        "generator": {"model": MODEL, "prompt_version": PROMPT_VERSION, "temperature": 0.0},
        "retrieval": retrieval,
        "rerank_source": rerank_source,
        "embedding": emb_meta,
        "cost_estimate": estimate,
        "results": results,
        "significance": significance,
        "planned_comparisons": planned_comparisons,
        "planned_pairs": planned,
        "seconds": seconds,
    }
    write_new(args.out_dir / f"{args.subset}-{slug}.json", json.dumps(record, indent=1) + "\n")
    print(f"  saved {args.out_dir / f'{args.subset}-{slug}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
