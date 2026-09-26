# RAG retrieval methods: results and when to use each

This summarizes the RAG retrieval experiments on HotpotQA. The details, including every command,
estimate and failure, are in `docs/devlog.md`.

**Setting.** 100 BEIR-HotpotQA test questions (81 bridge, 19 comparison) over a 20,906-passage
subset corpus: the gold passages, HotpotQA's distractors, and 20k seeded background passages.
Every parameter was tuned on a separate 100-question dev subset. The answers come from
claude-haiku-4-5 at temperature 0, with sentence-level citations. Latencies are single-query
wall-clock times on a MacBook Air M2 CPU (4 torch threads). These numbers are **not comparable to
full-corpus BEIR results**. The full 5.2M-passage runs, and GPU reranking, are planned for the
Linux machine.

## Results

| Method | Retrieval R@5 | Both gold in top 5 | Answer F1 | Joint F1 | Abstained | Retrieval latency (mean / p95) |
|---|---|---|---|---|---|---|
| Fused top 5 (BM25 + dense, RRF) | 0.820 | 0.65 | 0.517 | 0.411 | 28% | ~1 ms search |
| Union top 5, no model (~7.3 passages) | 0.870 | 0.76 | 0.572 | 0.465 | 24% | ~1 ms search |
| Single-hop bge rerank (pool N=20) | 0.915 | 0.83 | 0.614 | 0.495 | 18% | 3.0 s / 4.9 s (scoring) |
| Two-hop, no model (m=1, keep 3) | 0.885 | 0.78 | 0.605 | 0.498 | 21% | 35 ms / 60 ms |
| Two-hop, bge, keep 2 | 0.905 | 0.82 | not run | not run | — | 2.2 s / 3.7 s |
| **Joint two-hop bge** (m=2, depth 5, N=10) | **0.935** | **0.87** | **0.669** | **0.564** | **12%** | 3.3 s / 5.3 s |
| Gold passages (upper bound) | 1.0 | 1.0 | 0.739 | 0.634 | 3% | — |

- **Significance.** The pre-declared comparisons use one Holm-corrected family per metric, with
  paired bootstrap and randomization tests.
  - Joint two-hop bge beats fused on every answer metric (F1 +0.152, p_holm 0.0006).
  - Single-hop bge beats fused on F1, supporting-fact F1 and joint F1 (p_holm ≤ 0.02).
  - Joint two-hop bge vs single-hop bge (F1 +0.055, p_holm 0.11), two-hop no-model vs fused, and
    the union vs fused are **not significant** on 100 questions.
- **Where the gains come from.** Answered-only F1 is about the same in every condition
  (0.72–0.77). Better retrieval raises scores by letting the model answer more questions
  (abstention 28% → 12%), not by improving the answers it already gives.
- **Latency and cost.** Latency covers retrieval only: generation time was not measured, and the
  question's own embedding is excluded. Generation costs about $0.22 per 100 questions for 5
  passages, and $0.28 for the union's 7.3.

## When to choose each

- **Fused top 5.** The default when latency matters most, or the questions are single-hop. Fast
  and cheap, but it misses the second passage for about 40% of HotpotQA bridge questions.
- **Union top 5, no model.** Don't. Two-hop no-model matches its recall with 5 passages instead
  of 7.3, so it spends fewer generation tokens for the same quality.
- **Two-hop, no model.** The best choice without a reranker. For about 35 ms more per query
  (almost all of it embedding the one hop-2 query), it recovers about 60% of the fused → joint
  F1 gain, and it never lost a comparison question's gold passage. This is the method to use when a
  second model or a multi-second budget isn't available.
- **Single-hop bge rerank.** Worth it when the questions are mostly single-hop and 3 s of CPU
  (much less on a GPU) is acceptable. It never hurt a comparison question in these runs.
- **Two-hop, bge, keep rule.** Superseded by joint reranking. It is slower than no-model two-hop,
  worse than joint, and its keep rule dropped a comparison question's gold passage.
- **Joint two-hop bge.** The best quality: it closes about 68% of the retrieved → gold F1 gap
  with the same 5 passages. Choose it for multi-hop question answering when about 3–5 s of CPU
  reranking per query is acceptable, or on a GPU. Known weakness: the expanded query can favor
  passages *similar to* the first-hop passage. It dropped one comparison question's gold passage
  at retrieval, though that question's answer was wrong in every condition, gold included. The
  devlog records two fixes: also scoring against the original question, and using raw logits.

## Limits

- **Sample size.** 100 test questions (19 of them comparisons): differences below about 0.05 F1
  cannot be resolved.
- **Tuning.** 108 + 8 + 8 settings were tuned on 100 dev questions. Dev-to-test drops
  (0.940 → 0.905 for two-hop bge) suggest mild overfitting.
- **Corpus.** Subset corpus only; full-corpus retrieval is harder and will lower every row.
- **Hardware.** CPU-only reranking times; a GPU changes the latency trade-off substantially.
