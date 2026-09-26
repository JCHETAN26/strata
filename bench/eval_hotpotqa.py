"""HotpotQA: answer quality and groundedness of Strata's cited-answer RAG pipeline.

    uv sync --group embed                           # for the dense half of retrieval
    uv run python scripts/prepare_hotpotqa.py --n 100 --seed 0
    uv run python bench/eval_hotpotqa.py --subset subset-n100-seed0

Conditions (same questions, same generator):
- distractor: each question's own 10 paragraphs (2 gold + 8 distractors), the standard HotpotQA
  "distractor" setting, so answer and supporting-fact scores are comparable to published work.
- retrieved: Strata's pipeline. All paragraphs of the subset form one corpus (deduplicated by
  title and text); HybridIndex retrieves the top `k` per question (RRF of BM25 and
  bge-small-en-v1.5 at a pinned revision), and those passages go to the generator.

Generator: rag/answer.py (claude-haiku-4-5, temperature 0, custom-content documents of sentences
with citations enabled; responses cached on disk by request hash).

Metrics (official HotpotQA definitions, rag/hotpot_metrics.py):
- Answer quality: EM, F1 of the "Answer:" line against the gold answer.
- Groundedness: supporting-fact precision / recall / F1 / EM of the *cited* sentences against
  the gold supporting facts (title, sentence index), and joint EM / F1.
- Also: answer-in-citations rate (for non yes/no answers, the normalized short answer occurs in
  the normalized cited text), citation coverage (answered questions with >= 1 citation),
  abstention rate, format failures, retrieval recall (gold paragraphs in the top k), cost.
- Paired bootstrap CIs and randomization tests between the two conditions (bench/significance.py).

Saves results/rag/hotpotqa-<subset>-<timestamp>.json (all per-question predictions and
citations, metrics, model, prompt version, dataset revision) plus official-format prediction
files for each condition. Without an API key it runs retrieval only and says so.
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
from significance import compare_all

sys.path.insert(0, str(REPO_ROOT / "rag"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from answer import (
    MODEL,
    PROMPT_VERSION,
    AnswerGenerator,
    CitedAnswer,
    Passage,
    load_env,
)
from hotpot_metrics import normalize_answer, score_example

EMBED_MODEL = "bge-small-en-v1.5"
METRICS = ("em", "f1", "sp_em", "sp_f1", "sp_prec", "sp_recall", "joint_em", "joint_f1")


def embed(texts: list[str], queries: list[str], cache: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """bge-small embeddings with the pinned revision and its formatting (cached on disk)."""
    from embed_beir import MODELS, format_queries

    spec = MODELS[EMBED_MODEL]
    meta_path = cache / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["revision"] == spec.revision and meta["num_docs"] == len(texts):
            return np.load(cache / "docs.npy"), np.load(cache / "queries.npy"), meta
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec.hf_name, revision=spec.revision, device="cpu")
    docs = model.encode([spec.doc_prefix + t for t in texts], normalize_embeddings=True)
    qs = model.encode(format_queries(spec, queries), normalize_embeddings=True)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "docs.npy", docs.astype(np.float32))
    np.save(cache / "queries.npy", qs.astype(np.float32))
    meta = {
        "model": spec.hf_name,
        "revision": spec.revision,
        "num_docs": len(texts),
        "query_prefix": spec.query_prefix,
        "normalized": True,
        "device": "cpu",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return docs.astype(np.float32), qs.astype(np.float32), meta


def answer_in_citations(answer: CitedAnswer) -> bool | None:
    """Does the normalized short answer occur in the normalized cited text? None if n/a."""
    short = normalize_answer(answer.short_answer)
    if not short or short in ("yes", "no", "unknown", "noanswer"):
        return None
    cited = " ".join(c.cited_text for s in answer.statements for c in s.citations)
    return short in normalize_answer(cited)


def summarize(records: list[dict]) -> dict:
    n = len(records)
    out = {m: float(np.mean([r["scores"][m] for r in records])) for m in METRICS}
    answered = [r for r in records if not r["abstained"]]
    supported = [r["answer_in_citations"] for r in records if r["answer_in_citations"] is not None]
    out.update(
        {
            "questions": n,
            "abstention_rate": 1 - len(answered) / n,
            # Answer quality among questions the model did answer (abstentions excluded), so a
            # cautious model is not scored as a wrong one. Read with abstention_rate.
            "answered": len(answered),
            "answered_em": float(np.mean([r["scores"]["em"] for r in answered]))
            if answered
            else None,
            "answered_f1": float(np.mean([r["scores"]["f1"] for r in answered]))
            if answered
            else None,
            "format_failures": sum(not r["ok"] for r in records),
            "citation_coverage": (
                float(np.mean([r["num_citations"] > 0 for r in answered])) if answered else 0.0
            ),
            "answer_in_citations_rate": float(np.mean(supported)) if supported else None,
            "answer_in_citations_n": len(supported),
            "input_tokens": sum(r["input_tokens"] for r in records),
            "output_tokens": sum(r["output_tokens"] for r in records),
            "cost_usd": sum(r["cost_usd"] for r in records),
            "from_cache": sum(r["from_cache"] for r in records),
        }
    )
    return out


def main(argv: list[str] | None = None, generator: AnswerGenerator | None = None) -> int:
    """generator: injected in tests; by default a real AnswerGenerator (needs an API key)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subset", default="subset-n100-seed0")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per question")
    parser.add_argument("--workers", type=int, default=4, help="concurrent API requests")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "rag")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "hotpotqa")
    args = parser.parse_args(argv)

    import strata

    data = args.data_dir
    questions = json.loads((data / f"{args.subset}.json").read_text())
    subset_meta = json.loads((data / f"{args.subset}.meta.json").read_text())

    # Pooled corpus: every paragraph of the subset, deduplicated.
    corpus: list[Passage] = []
    seen: dict[tuple[str, tuple[str, ...]], int] = {}
    for q in questions:
        for p in q["context"]:
            key = (p["title"], tuple(p["sentences"]))
            if key not in seen:
                seen[key] = len(corpus)
                corpus.append(Passage(f"p{len(corpus)}", p["title"], p["sentences"]))
    texts = [f"{p.title} {''.join(p.sentences)}".strip() for p in corpus]
    doc_vecs, query_vecs, emb_meta = embed(
        texts,
        [q["question"] for q in questions],
        REPO_ROOT / "data" / "embeddings" / f"hotpotqa-{args.subset}",
    )
    index = strata.HybridIndex(doc_vecs.shape[1], metric="ip")
    index.add(
        [p.doc_id for p in corpus], [f"{p.title}\n{''.join(p.sentences)}" for p in corpus], doc_vecs
    )
    ids, _ = index.search([q["question"] for q in questions], query_vecs, args.k, method="rrf")
    retrieved = [[corpus[i] for i in row if i >= 0] for row in ids]
    gold_titles = [{t for t, _ in q["supporting_facts"]} for q in questions]
    both_gold = [
        gold <= {p.title for p in r} for gold, r in zip(gold_titles, retrieved, strict=True)
    ]
    any_gold = [
        bool(gold & {p.title for p in r}) for gold, r in zip(gold_titles, retrieved, strict=True)
    ]
    retrieval = {
        "k": args.k,
        "corpus_paragraphs": len(corpus),
        "both_gold_in_top_k": float(np.mean(both_gold)),
        "any_gold_in_top_k": float(np.mean(any_gold)),
        "method": "HybridIndex rrf (k=60), BM25 + bge-small-en-v1.5",
    }
    print(f"{args.subset}: {len(questions)} questions, pooled corpus {len(corpus)} paragraphs")
    print(
        f"  retrieval top-{args.k}: both gold {retrieval['both_gold_in_top_k']:.2%}, "
        f"any gold {retrieval['any_gold_in_top_k']:.2%}"
    )

    if generator is None:
        load_env(REPO_ROOT)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "  no ANTHROPIC_API_KEY (env or .env): skipping answer generation", file=sys.stderr
            )
            return 2
        generator = AnswerGenerator(cache_dir=REPO_ROOT / "data" / "cache" / "answers")
    conditions = {
        "distractor": [
            [
                Passage(f"{q['id']}:{j}", p["title"], p["sentences"])
                for j, p in enumerate(q["context"])
            ]
            for q in questions
        ],
        "retrieved": retrieved,
    }
    results: dict[str, dict] = {}
    start = time.perf_counter()
    for name, passage_lists in conditions.items():
        with ThreadPoolExecutor(args.workers) as pool:
            answers = list(
                pool.map(
                    lambda qp: generator.answer(qp[0]["question"], qp[1]),
                    zip(questions, passage_lists, strict=True),
                )
            )
        records = []
        for q, a in zip(questions, answers, strict=True):
            scores = score_example(
                a.short_answer,
                a.cited_sentences,
                q["answer"],
                [tuple(f) for f in q["supporting_facts"]],
            )
            records.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "gold_answer": q["answer"],
                    "short_answer": a.short_answer,
                    "abstained": a.abstained,
                    "ok": a.ok,
                    "scores": scores,
                    "cited_sentences": sorted(a.cited_sentences),
                    "num_citations": sum(len(s.citations) for s in a.statements),
                    "answer_in_citations": answer_in_citations(a),
                    "input_tokens": a.input_tokens,
                    "output_tokens": a.output_tokens,
                    "cost_usd": a.cost_usd,
                    "from_cache": a.from_cache,
                    "text": a.text,
                    "stop_reason": a.stop_reason,
                }
            )
        results[name] = {"summary": summarize(records), "records": records}
    seconds = time.perf_counter() - start

    per_question = {
        m: {c: {r["id"]: r["scores"][m] for r in results[c]["records"]} for c in conditions}
        for m in ("f1", "sp_f1", "joint_f1")
    }
    significance = {m: [x.to_dict() for x in compare_all(pq)] for m, pq in per_question.items()}

    for name, r in results.items():
        s = r["summary"]
        print(
            f"  {name:10s} EM {s['em']:.3f}  F1 {s['f1']:.3f}  | cited-sentence SP F1 "
            f"{s['sp_f1']:.3f} (P {s['sp_prec']:.3f}, R {s['sp_recall']:.3f})  | joint F1 "
            f"{s['joint_f1']:.3f}  | abstain {s['abstention_rate']:.0%}, answer-in-citations "
            f"{s['answer_in_citations_rate']:.0%}, ${s['cost_usd']:.3f}"
        )
    for m, pairs in significance.items():
        for c in pairs:
            print(
                f"  {m:8s} {c['a']} - {c['b']}: {c['mean_diff']:+.3f} "
                f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}], p={c['p_value']:.4f}"
            )

    slug = timestamp_slug()
    record = {
        **metadata(),
        "subset": args.subset,
        "dataset": subset_meta,
        "generator": {"model": MODEL, "prompt_version": PROMPT_VERSION, "temperature": 0.0},
        "retrieval": retrieval,
        "embedding": emb_meta,
        "results": results,
        "significance": significance,
        "seconds": seconds,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_new(
        args.out_dir / f"hotpotqa-{args.subset}-{slug}.json", json.dumps(record, indent=1) + "\n"
    )
    for name, r in results.items():  # official hotpot_evaluate_v1.py input format
        official = {
            "answer": {x["id"]: x["short_answer"] for x in r["records"]},
            "sp": {x["id"]: x["cited_sentences"] for x in r["records"]},
        }
        write_new(
            args.out_dir / f"hotpotqa-{args.subset}-{slug}.{name}.pred.json", json.dumps(official)
        )
    print(f"  saved {args.out_dir / f'hotpotqa-{args.subset}-{slug}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
