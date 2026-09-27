# RAG retrieval methods: results and when to use each

This summarizes the RAG retrieval experiments: HotpotQA end to end on a subset corpus (Mac),
then full-corpus retrieval, SciFact reranking, and GPU latency (Kaggle T4). The details, including
every command, estimate and failure, are in `docs/devlog.md`.

## HotpotQA subset, end to end

**Setting.** 100 BEIR-HotpotQA test questions (81 bridge, 19 comparison) over a 20,906-passage
subset corpus: the gold passages, HotpotQA's distractors, and 20k seeded background passages.
Every parameter was tuned on a separate 100-question dev subset. The answers come from
claude-haiku-4-5 at temperature 0, with sentence-level citations. Latencies are single-query
wall-clock times on a MacBook Air M2 CPU (4 torch threads). These numbers are **not comparable to
full-corpus BEIR results**; see the full-corpus section below.

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

### When to choose each

- **Fused top 5.** The default when latency matters most, or the questions are single-hop. Fast
  and cheap, but it misses the second passage for about 40% of HotpotQA bridge questions.
- **Union top 5, no model.** Don't. Two-hop no-model matches its recall with 5 passages instead
  of 7.3, so it spends fewer generation tokens for the same quality.
- **Two-hop, no model.** The best choice without a reranker. For about 35 ms more per query
  (almost all of it embedding the one hop-2 query), it recovers about 60% of the fused → joint
  F1 gain, and it never lost a comparison question's gold passage. This is the method to use when a
  second model or a multi-second budget isn't available.
- **Single-hop bge rerank.** Worth it on HotpotQA-like questions when 3 s of CPU (about 0.7 s
  on a T4) is acceptable. It never hurt a comparison question in these runs. Not on SciFact-like
  (scientific claim) data, where it gave no gain (see below): measure on your domain first.
- **Two-hop, bge, keep rule.** Superseded by joint reranking. It is slower than no-model two-hop,
  worse than joint, and its keep rule dropped a comparison question's gold passage.
- **Joint two-hop bge.** The best quality: it closes about 68% of the retrieved → gold F1 gap
  with the same 5 passages. Choose it for multi-hop question answering when about 3–5 s of CPU
  reranking per query is acceptable, or on a GPU. Known weakness: the expanded query can favor
  passages *similar to* the first-hop passage. It dropped one comparison question's gold passage
  at retrieval, though that question's answer was wrong in every condition, gold included. The
  devlog records two fixes: also scoring against the original question, and using raw logits.

## Full-corpus HotpotQA retrieval (5.2M passages)

Kaggle T4 session (`results/kaggle/environment.json`), code `43ca3ce`, all 7,405 BEIR test
queries, exact search (no ANN). Result:
`results/hybrid/hotpotqa-bge-small-en-v1.5-20260927-094841-995184.json`.

| Method | nDCG@10 | R@100 | Published reference |
|---|---|---|---|
| BM25 (Strata's C++ index) | 0.6329 | 0.7957 | Anserini 2.3.0 flat: 0.633 / 0.7957 |
| Dense (bge-small-en-v1.5, exact) | 0.6993 | 0.8487 | model card (MTEB): 0.69935 / 0.84862 |
| **RRF (BM25 + dense)** | **0.7297** | **0.8722** | — |

- **The baselines reproduce the published numbers** at full scale. BM25 matches Anserini to the
  fourth decimal, and dense matches bge-small's reported MTEB score. That validates the C++ BM25
  (tokenizer, stemming, Lucene scoring) and the dense pipeline (pinned revision, query
  instruction, normalization) on 5.2M passages, not just on SciFact.
- **RRF gains more at full scale than on SciFact:** +0.030 nDCG@10 over dense ([+0.026, +0.035])
  and +0.097 over BM25, both p_holm 0.0003 over 7,405 queries. On SciFact, RRF was +0.015 over
  dense. A likely reason: HotpotQA's questions name entities (which BM25 catches) and paraphrase
  relations (which dense catches), so the two retrievers fail on different queries. BM25 wins
  1,742 queries against dense and loses 3,000.
- **Throughput:** 41 QPS for BM25 and 3.2 QPS for exact dense (4 host vCPUs, brute force over
  5.2M × 384 floats). This is a correctness baseline, not a speed result; ANN search is Strata's
  HNSW work.

## SciFact reranking (T4 GPU)

300 SciFact test queries; pool depth N tuned on 300 seeded train queries (both models chose
N=10). Code `43ca3ce`. Result: `results/rerank/scifact-20260927-101744-604023.json`.

| Method | nDCG@10 | R@5 | All relevant @5 |
|---|---|---|---|
| Fused top 5 (RRF) | **0.727** | 0.790 | 0.77 |
| Union top 5, no model (~8.0 passages) | — | 0.827 | 0.81 |
| bge-reranker-base | 0.727 | 0.780 | 0.76 |
| MiniLM-L6 | 0.699 | 0.747 | 0.72 |

- **Reranking does not help on SciFact.** bge is indistinguishable from fused (nDCG@10 −0.0003,
  p_holm 0.98). MiniLM is lower by 0.028 (p_holm 0.076, not significant). This is the opposite of
  HotpotQA, where bge raised all-relevant@5 from 0.65 to 0.83.
- **Likely reasons (hypotheses, not tested):**
  - *Domain mismatch.* Both cross-encoders are trained mostly on web search data (MS MARCO-style
    queries and passages). SciFact pairs scientific claims with paper abstracts.
  - *Less headroom.* Fused retrieval is already strong on SciFact: 0.77 of queries have all
    relevant abstracts in the top 5, against a pool ceiling of 0.87, so a perfect reranker could
    add at most 0.10. On the HotpotQA subset the gap was 0.65 → 0.90.
- The union of top 5s (+0.038 R@5 over fused, p_holm 0.027) is the only significant gain, and it
  sends about 8 passages instead of 5.
- **Minor discrepancy:** fused nDCG@10 is 0.7269 here, against 0.7273 in the committed SciFact
  hybrid run, on different hardware and a different code version; the cause has not been
  investigated.

## Reranking latency: M2 CPU vs T4 GPU

Same 100 HotpotQA-subset queries and pools. `results/rerank/rerank_latency_cpu_vs_gpu.json`;
the GPU run is `results/rerank/hotpotqa-subset-n100-seed0-bg20000-20260927-105611-634422.json`.

| Model | Pool (passages) | M2 CPU mean / p95 | T4 GPU mean / p95 | Ratio |
|---|---|---|---|---|
| bge-reranker-base | 32.4 | 3029 / 4887 ms | 688 / 1070 ms | 4.4× |
| MiniLM-L6 | 15.1 | 231 / 464 ms | 47 / 86 ms | 4.9× |

- **Same results on both devices.** The GPU run chose the same pool depths and produced identical
  top-5 lists for all 100 queries and both models. The CPU/GPU numeric differences never changed
  a ranking.
- **Caveats:**
  - *Different machines.* The CPU number is the MacBook Air M2 (4 torch threads, fanless). The
    GPU number is a Kaggle T4 with a 4-vCPU Xeon host. The ratio compares two setups, not "GPU vs
    CPU on one machine"; the Kaggle host's own CPU was not timed.
  - *The GPU is underused.* Each query's pool (15–32 pairs) is one partial batch (batch size 64),
    so kernel launch, host-side tokenization and transfer dominate, and only one of the two T4s
    is used. Batching several queries together would raise throughput a lot; per-query latency
    would improve less.
  - *Passage length matters.* On SciFact, bge takes 565 ms on the T4 for only 16.5 passages,
    because abstracts are much longer than HotpotQA passages.
- For the joint two-hop reranker (about 18 passages, two query types), this suggests something
  like 0.5–1 s per query on a T4, instead of 3.3 s on the M2. That is an extrapolation, not
  measured.

## Limits

- **Sample size.** 100 test questions (19 of them comparisons): differences below about 0.05 F1
  cannot be resolved.
- **Tuning.** 108 + 8 + 8 settings were tuned on 100 dev questions. Dev-to-test drops
  (0.940 → 0.905 for two-hop bge) suggest mild overfitting.
- **Corpus.** The end-to-end and multi-hop results use the subset corpus only; full-corpus
  retrieval is harder and would lower every row. Full-corpus results exist for single-hop
  retrieval (BM25, dense, RRF) but not yet for reranking, multi-hop, or answer generation.
- **Hardware.** The CPU and GPU latencies come from different machines (see above).
- **Datasets.** Reranking was evaluated on two datasets, and the verdict flips between them. FiQA
  and NFCorpus are still to do.
