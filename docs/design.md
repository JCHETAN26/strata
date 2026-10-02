# Strata design

> **Draft. The numbers here are development results** from a fanless MacBook Air (M2): recall
> values are final, and speed numbers are indicative only, because the laptop throttles under
> sustained load. They will be replaced by the x86 results from the AWS session
> ([`aws-plan.md`](aws-plan.md)) and the ARM results from the Oracle machine. Every number links
> to the generated table it comes from.

Strata is a vector search engine in C++20: exact and HNSW search with SIMD distance kernels,
product quantization, metadata filtering, durable storage with a write-ahead log, a gRPC shard
server with a sharding coordinator, and a Python RAG layer (BM25 + dense fusion, reranking, cited
answers). This document explains the main design choices and their trade-offs. The HNSW core has
its own line-by-line explainer: [`explainers/hnsw.md`](explainers/hnsw.md).

## 1. Layout of the system

```
                 Python: bindings (nanobind), RAG layer (BM25 + dense RRF, rerank, Claude answers)
                                   │
 client ──gRPC/TLS──► coordinator ─┼─► shard 0: Collection ─► HnswIndex | BruteForceIndex
                      (scatter,    ├─► shard 1:   ├─ WAL (every write, CRC'd, LSN-ordered)
                       merge top-k)└─► shard N:   └─ snapshot (atomic, versioned)
                                                  distance kernels: scalar | NEON | AVX2
```

- **Library first.** Everything is a C++ library (`include/strata/`). The server, the Python
  bindings, and the benchmark harnesses are thin layers over it.
- **Errors at API boundaries are values** (`Expected<T>`), not exceptions, so hot paths have no
  unwinding cost and callers must look at failures.
- **Thread safety is documented on every public class.** For example, `Collection` takes a shared
  lock for searches and an exclusive one for writes.

## 2. Distance kernels

- **Scalar code is the reference.** NEON (ARM) and AVX2+FMA (x86) kernels are chosen at compile
  time, and each has a test asserting it matches scalar within a documented tolerance.
- **Measured:** brute force over SIFT1M runs at 85.5 QPS with NEON vs 9.3 QPS scalar, a **9.2x**
  speedup (single thread). On SIFT10K it is 6.2x. Source: [`results/tables.md`](../results/tables.md).
- **Why compile-time selection:**
  - Runtime dispatch would put an indirect call in the innermost loop.
  - The targets are known: the Mac and the Oracle machine are ARM, AWS is x86 with AVX2.
  - The scalar fallback is always built.

## 3. HNSW: graph parameters

Faithful to Malkov & Yashunin (2018), including Algorithm 4, the neighbor-selection heuristic.
The deviations are listed in the explainer (section 8).

- **M = 16, ef_construction = 200** by default:
  - M bounds the out-degree (2M on layer 0).
  - ef_construction is the build-time beam width.
  - Higher values buy recall at the cost of build time and memory.

  The explainer's section 6 measures the trade-offs; the headline is that ef_search, not M, is
  the knob that matters at query time.
- **The heuristic matters on clustered data.** On a 100-cluster set it lifts recall@10 at
  ef_search=10 from **0.745 to 0.908**. On SIFT10K the gain is +0.02 at low ef and vanishes by
  ef=40, and the mean layer-0 degree drops from 24.8 to 16.6. Source:
  [`selection_comparison.md`](../results/selection/selection_comparison.md),
  [`tables.md`](../results/tables.md).
- **Against hnswlib and FAISS** (SIFT1M, single thread, same M and ef_construction). Recall
  matches at every ef_search; Strata 0.9283 vs hnswlib 0.9288 vs FAISS 0.9344 at ef=40. Mac QPS
  (indicative): Strata 10,685, hnswlib 6,194, FAISS 12,902. The hnswlib number is unfair on ARM
  (no NEON path); the x86 run settles the speed comparison. Source:
  [`hnsw_vs_reference.md`](../results/hnsw/hnsw_vs_reference.md).
- **Memory layout:**
  - Vectors are stored contiguously.
  - Each node's neighbor lists are fixed-capacity slices of flat arrays.
  - `search_layer` prefetches the next candidate's neighbors and vectors, which gave about 2x on
    the 200k subset ([`results/ab/`](../results/ab/)).
- **Parallel build:**
  - `add_batch` with a thread pool.
  - Each node's lists are guarded by one lock from a striped table (65,536 stripes); no thread
    holds two node locks at once, so lock ordering can't deadlock.
  - Result: 3.2x faster with 4 threads on the 200k subset, with identical recall and graph
    statistics. Source: [`build_scaling`](../results/hnsw_build/build_scaling_sift1m-200k-q1000.md).

## 4. Deletes and persistence

- **Deletes are tombstones.**
  - A deleted node keeps routing searches through the graph but is never returned.
  - The stopping rule widens so that k live results still come back.
  - Heavy deletion slowly degrades the graph; the measurement at 0–90% deleted, against a rebuild,
    is in [`deletes`](../results/hnsw_deletes/deletes_sift1m-200k-q1000.md).
  - Trade-off: no compaction yet. A rebuild is the remedy after mass deletion.
- **Write path:** validate → append to the WAL (CRC32C per record, LSN-ordered; fsync per write
  unless `SyncMode::kNone`) → apply in memory → acknowledge.
  - Crash tests kill the process mid-write and check that no acknowledged write is lost.
  - Cost of this durability: one fsync per insert caps durable inserts at a few hundred per
    second on the Mac. Group commit is the known fix and is listed as optional in the plan.
- **Snapshots:**
  - They hold the whole HNSW graph, including the level generator's state as a seed plus a draw
    count. The text form of `mt19937_64` differs between libc++ and libstdc++, which broke
    cross-platform loads in format v2.
  - Recovery loads the snapshot, then replays later WAL records by LSN, so the recovered graph is
    bit-identical to the one that crashed.
  - Loading a 200k index takes **0.39 s**, against **25.6 s** to rebuild it
    ([`persist`](../results/storage/hnsw_persist_sift1m-200k-q1000.md)).

## 5. Compression: product quantization

- **How it works:**
  - Vectors split into m sub-vectors, each replaced by the index of its nearest of 256 k-means
    centroids (one byte).
  - Search uses asymmetric distance: precomputed per-query lookup tables against the full-precision
    query.
  - Re-ranking uses the original vectors for the top candidates.
- **Trade-off:** memory against recall, recovered by re-ranking. On SIFT10K (codes plus codebooks
  against 5.12 MB of raw float32):

  | m (bytes/vector) | index size | recall@10, no re-rank | re-rank 20 | re-rank 50 | re-rank 100 |
  |---:|---:|---:|---:|---:|---:|
  | 4 | 0.17 MB (30x smaller) | 0.423 | 0.601 | 0.824 | 0.919 |
  | 16 | 0.29 MB (17.6x) | 0.732 | 0.929 | 0.996 | 1.000 |
  | 64 | 0.77 MB (6.6x) | 0.938 | 1.000 | 1.000 | 1.000 |

  - The originals still exist on the side for re-ranking: PQ shrinks the hot, scanned data, not
    the total footprint.
  - Source: the `strata-pq-*` records in `results/search/siftsmall/`; chart
    [`pq_memory_siftsmall.png`](../results/plots/pq_memory_siftsmall.png).
- **Not done:** PQ combined with HNSW (optional in the plan).

## 6. Filtered search

Three strategies, plus automatic selection between them.

- **Pre-filter:** evaluate the filter on every id, then score the matches exactly. Its cost grows
  with n plus the matches. Exact, and best for very selective filters.
- **Graph:**
  - Search HNSW, treating non-matching nodes like tombstones: traverse them but never return them.
  - Cost grows with ef / selectivity.
  - Approximate, and best for broad filters.
  - It degrades on *correlated* filters (matches clustered away from the query), which the
    benchmark covers with k-means-cluster filters.
- **Auto:**
  - Estimates the filter's selectivity once per filter from a sample, sampling more when the
    estimate is near the threshold.
  - Below the threshold it uses the pre-filter, otherwise the graph.
  - A graph search that exceeds its budget, `(fallback_budget + selectivity) × n` distance
    computations, falls back to the pre-filter, so a bad estimate costs time, not recall.

**Measured crossover** on the 200k subset: at **1.03–1.28%** selectivity the pre-filter overtakes
the graph (random to correlated filters, recall targets 0.95 and 0.99). The default threshold is
1.3%, the safe side.
- The crossover depends on n: the pre-filter scales with n, the graph with ef / selectivity.
- So it is re-measured at 1M and 10M on AWS, with a dense 1–3% sweep that counts how often auto
  falls back.
- Source: [`filter`](../results/hnsw_filter/filter_sift1m-200k-q1000.md).

## 7. Sharding

- **Partitioning:**
  - Vectors are dealt round-robin to shards, so data and search work divide evenly without any
    knowledge of the vectors.
  - The trade-off: every query visits every shard (scatter-gather). Partitioning by cluster would
    let a query visit fewer shards, but would need routing and rebalancing, and it skews load.
- **Global ids** are `local × N + shard`:
  - They decode with no lookup table, and no shard needs to know the others' sizes.
  - They are 32-bit, so each shard holds at most about 4.29 billion / N vectors. Shards refuse
    inserts past that bound (RESOURCE_EXHAUSTED) rather than wrapping ids.
  - The `--shard` order is part of every id, so it must stay fixed.
- **Fan-out:**
  - Search, Stats, and InsertBatch call all shards concurrently through gRPC's callback API, with
    a deadline per call. The coordinator merges the per-shard top-k lists by (distance, id).
  - This replaced one thread per shard per query. On the Mac: median latency **−8% (2 shards) and
    −10% (4 shards)**, and throughput with 8 clients **+17% and +23%** (recall identical).
  - Tail latency on one shared-core laptop was noisy in both modes. It is re-measured on separate
    machines on AWS.
  - Source: [`coordinator_latency`](../results/server/coordinator_latency_sift1m-200k-q1000.md).
- **InsertBatch is not atomic across shards:**
  - The response lists each input's id, or a reserved "not inserted" value, plus the first error.
  - A retry carries those ids back, and the shard checks them against the stored vectors instead
    of inserting again, so a retry is idempotent.
  - Remaining gap: a shard that committed but whose reply was lost looks like a failure, and a
    retry duplicates those vectors. Closing it needs a request key in the WAL.
- **Security:**
  - Servers bind 127.0.0.1 by default.
  - Exposing one requires TLS and a shared bearer token (checked in constant time before any
    handler runs), or an explicit `--insecure`. A token without TLS is refused.
  - Not done: mutual TLS and token rotation.

## 8. RAG layer

- **Retrieval:**
  - BM25 (Strata's C++ index, matching Anserini exactly: SciFact nDCG@10 0.6789) and dense
    retrieval (bge-small) are fused with reciprocal rank fusion.
  - On the full HotpotQA corpus (5.2M passages): BM25 0.633, dense 0.699, **RRF 0.730** nDCG@10.
  - Multi-hop retrieval with a cross-encoder lifts answer F1 from 0.517 to 0.669, against 0.739
    with the gold passages.
  - Source: [`rag-results.md`](rag-results.md).
- **Answers** come from Claude Haiku with citations, and the model can abstain.

## 9. Known weaknesses

- **FAISS leads at low ef_search on the Mac** (34k vs 24k QPS at ef=10 on SIFT1M). Whether that
  holds on x86 is one of the AWS questions.
- **No group commit:** durable single inserts are slow.
- **Tombstones are never compacted.**
- **The bulk load into a shard is single-threaded** (insert by insert through the WAL), which
  limits sharded runs to about 1M vectors for now.
- **No AVX-512 kernels** (future work). On AVX-512 x86 CPUs, FAISS and hnswlib can use 512-bit
  vectors while Strata uses 256-bit. The x86 comparison is therefore run twice: at AVX2 for all
  three (the like-for-like result), and with the references at AVX-512, labeled as such.
- **No ThreadSanitizer run of the server** against stock gRPC builds (protobuf changes its layout
  under TSan).
