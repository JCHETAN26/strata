# Strata — Build Plan

**Strata** is a distributed vector search engine written from scratch in C++20, with a RAG layer on top.
Given millions of vectors, it finds the most similar ones in milliseconds, and uses that to answer
questions from documents with cited sources.

**What it proves:** algorithms (from-scratch HNSW), systems depth (SIMD, concurrency, persistence,
sharding), and applied AI (hybrid retrieval, reranking, RAG), benchmarked against FAISS and hnswlib.

**Status (2026-09-28):** Phases 0–2 done; Phases 3–6 and 8 done except the items in the finish
order below; Phase 7 not started. The HNSW core is AI-implemented at my request in reviewed stages
(see `CLAUDE.md`), with prefetching in place. Every number so far is a Mac development result: recall
is final, speed is indicative. Tests: 188 C++ (debug, asan, tsan) and 135 Python, passing on the Mac.
`docs/checklist.md` has the item-by-item status.

---

## Finish order (adopted 2026-09-28)

Heavy work runs one job at a time, with the thermal and busy-machine checks in `bench/`
(`benchmeta.preflight`).

1. **HNSW persistence and deletes:** save/load the graph in snapshots; tombstone deletes in HNSW
   search (Phase 4).
2. **Parallel index build** on the thread pool (Phase 3).
3. **Filtered HNSW search:** filter during graph traversal, plus automatic strategy selection from
   the measured crossover (Phase 6).
4. **gRPC server and sharding**, built on the Oracle ARM machine, not the Mac (Phase 7).
5. **One AWS session (x86):** final benchmarks (SIFT1M, GloVe-100, recall-QPS vs. FAISS and
   hnswlib), the 10M-vector run, thread scaling, multi-machine sharding scaling, and `perf`
   profiling (Phase 9).
6. **Final ARM numbers on the Oracle machine** (Phase 9).
7. **README, design doc, and making the repo public** (Phase 9).

**Optional** (only if time allows, never blocking the above): PQ inside HNSW, WAL group commit,
extra BEIR datasets (FiQA, NFCorpus, and full-corpus hybrid + rerank), full-corpus HotpotQA
answer groundedness.

---

## Ground rules

- **Understand HNSW yourself.** Read the paper (Malkov & Yashunin, *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs*). The core is
  AI-implemented at my request (changed 2026-09-27), in reviewed stages, with
  `docs/explainers/hnsw.md` so I can explain every line. Interviewers will ask.
- **AI handles scaffolding:** build files, harnesses, bindings, tests, scripts.
- **Every result comes from a script** that saves raw data and generates its table or chart.
- **Always compare against a baseline** (brute force, hnswlib, FAISS, scalar vs. SIMD, with vs. without).
- **Long benchmarks run on AWS (x86) or the Oracle machine (ARM).** The MacBook Air is fanless,
  throttles, and has shut down under sustained load; on it, heavy jobs run one at a time behind
  the thermal checks, on subsets (e.g. 200k SIFT vectors) rather than full SIFT1M.
- **Commit after every working step.**

---

## Hardware

| Machine | Specs | Role |
|---|---|---|
| MacBook Air | M2 (ARM), 8 GB, fanless | Development, small datasets and subsets, NEON; development results only |
| Oracle ARM machine | (record specs in the first session there) | gRPC server and sharding (Phase 7); final ARM numbers |
| AWS (one session) | x86 with AVX2; instance type to be chosen and recorded | Final x86 benchmarks, 10M run, thread scaling, multi-machine scaling, `perf` |
| Kaggle (GPU notebook) | T4 | Embeddings and reranking for the RAG layer (done; see `kaggle/`) |
| IdeaPad | Ryzen 7 5800H, Ubuntu 22.04 | Linux bring-up on 2026-09-25 (`docs/reports/linux-bringup.md`); not used in the finish order (disk full) |

Code moves between machines via GitHub.

---

## Datasets

| Dataset | Size | Used for |
|---|---|---|
| SIFT10K (siftsmall) | 10K × 128-d | Fast iteration and correctness tests (Mac) |
| SIFT1M | 1M × 128-d, ~500 MB | Main benchmark (Mac subsets for dev, AWS/Oracle for results) |
| GloVe-100 | ~1.2M × 100-d | Harder benchmark, angular distance |
| 10M-vector set (e.g., BIGANN/SIFT subset) | ~5 GB raw | Large-scale run in the AWS session (use PQ if memory is tight) |
| BEIR: SciFact, FiQA, NFCorpus | Small corpora with relevance labels | Retrieval quality (nDCG@10) |
| HotpotQA (subset) | QA pairs with supporting docs | Answer quality and groundedness |

ann-benchmarks provides SIFT1M and GloVe-100 as HDF5 with precomputed ground-truth neighbors.
Convert them to raw binary with a Python script so the C++ code needs no HDF5 dependency.

---

## Tech stack

- **C++20**, CMake + Ninja, **vcpkg** (same dependency versions on macOS and Ubuntu)
- GoogleTest, Google Benchmark
- clang-format, clang-tidy, AddressSanitizer + UndefinedBehaviorSanitizer build modes
- gRPC + Protobuf (server phase)
- nanobind or pybind11 (Python bindings)
- Python: numpy, hnswlib, faiss-cpu, matplotlib, a small open embedding model, a small cross-encoder
- Claude Haiku (`claude-haiku-4-5`) for answer generation

---

## Phase 0 — Setup (1–2 sessions, Mac)

- [x] Install Xcode Command Line Tools: `xcode-select --install`
- [x] Install Homebrew, then `brew install cmake ninja`
- [x] Set Git identity on the Mac to your own details
- [x] Create `~/Strata`, `git init`, private GitHub repo
- [x] Write `CLAUDE.md` (project summary, phases, conventions, the HNSW ground rule)
- [x] CMake project with vcpkg manifest, GoogleTest, Google Benchmark
- [x] Sanitizer build presets (ASan + UBSan)
- [x] clang-format / clang-tidy configs
- [x] Python script: download ann-benchmarks HDF5 → raw binary (vectors, queries, ground truth)
- [x] Download SIFT10K

**Done when:** an empty test and benchmark compile and run; SIFT10K files exist.

---

## Phase 1 — Baseline (≈1 week, Mac)

- [x] Scalar distance functions: L2, inner product, cosine
- [x] Brute-force exact search (the ground truth)
- [x] Recall@k calculator
- [x] Benchmark harness: QPS and latency percentiles, raw results saved to files
- [x] Unit tests for distances and recall

**Measure:** brute-force QPS on SIFT10K (and SIFT1M).

---

## Phase 2 — HNSW from scratch (1–2 weeks, Mac) ★ core

- [x] Random level assignment
- [x] Layered graph insertion
- [x] Neighbor-selection heuristic from the paper (not just "closest M")
- [x] Search: greedy descent through upper layers, `efSearch`-bounded search at layer 0
- [x] Parameters: `M`, `efConstruction`, `efSearch`
- [x] Tests: recall vs. brute force; edge cases (empty index, one vector, duplicates) — spec in `tests/hnsw_test.cpp`
- [x] Python comparison scripts for hnswlib and FAISS on the same data

**Measure:** recall vs. QPS curve against hnswlib and FAISS on SIFT10K and SIFT1M.
First version will likely be slower than hnswlib. That's expected.
Done on the M2 as development results (`results/hnsw/hnsw_vs_reference.md`): recall matches
hnswlib at every ef_search; QPS is indicative only, and final speed comparisons wait for Phase 9.

---

## Phase 3 — Performance (Mac; final numbers on AWS)

- [x] SIMD distance kernels: NEON (Mac), AVX2 (verified on the IdeaPad), scalar fallback
- [x] Test: all kernels return identical results to scalar
- [x] Contiguous vector storage, compact neighbor lists, prefetching in `search_layer` (~2x on the 200k subset)
- [x] Prefetching in the upper-layer walk: tried, no gain distinguishable from noise on the M2, not kept; re-test on x86
- [x] Thread pool: parallel batch queries
- [ ] Parallel index build — finish order 2
- [x] Profile hotspots on the Mac (`sample`; `results/profiles/`)
- [ ] Profile on Linux with `perf`, including `search_layer`'s own loop — deferred to the AWS session
- [x] Record the effect of each optimization separately (`results/ab/`, SIMD and thread results)

**Measure:** SIMD speedup vs. scalar; QPS scaling across cores; updated recall-QPS curve.

---

## Phase 4 — Persistence & crash recovery (≈1 week, Mac)

- [x] Save and load index to/from disk (snapshot of vectors + tombstones)
- [ ] Save and load the HNSW graph in snapshots — finish order 1
- [x] Write-ahead log: log every insert before applying it
- [x] Deletes via tombstones (brute force)
- [ ] HNSW deletes (tombstones skipped during search) — finish order 1
- [ ] *Optional:* WAL group commit (durable inserts are capped at ~330/s by one fsync each)
- [x] Crash tests: kill mid-insert, restart, verify no acknowledged write is lost or corrupted

**Measure:** save/load time; recovery correctness across repeated crash tests.

---

## Phase 5 — Product quantization (≈1 week, Mac)

- [x] Split vectors into sub-vectors; k-means codebook per sub-space
- [x] Asymmetric distance computation with precomputed lookup tables
- [x] Re-rank top candidates with full-precision vectors
- [ ] *Optional:* combine PQ with HNSW

**Measure:** memory reduction and recall at several compression levels (memory vs. recall chart).

---

## Phase 6 — Filtered search (≈1 week, Mac)

- [x] Metadata attributes per vector (e.g., year, category)
- [x] Strategy A: pre-filter + brute force (best for very selective filters)
- [x] Strategy B: filter during graph traversal (best for broad filters) — finish order 3
- [x] Automatic strategy selection based on estimated selectivity (`estimate_selectivity` exists) — finish order 3

**Measure:** recall and QPS at 1%, 10%, 50% filter selectivity for each strategy (crossover chart).
Done on the 200k subset (Mac). In the AWS session the crossover is re-measured at 1M and 10M, and
those sweeps add 1-3% selectivity (near the threshold) to measure how often auto's fallback fires;
that sweep is skipped on the Mac.

---

## Phase 7 — Server & sharding (1–2 weeks, Oracle ARM machine; scaling on AWS)

- [ ] gRPC API: insert, search, delete
- [ ] Sharding: vectors partitioned across shards
- [ ] Coordinator: parallel scatter to all shards, gather and merge top-k
- [ ] Docker Compose running multiple shards on the Oracle machine
- [ ] Multi-machine scaling run in the AWS session

**Measure:** end-to-end gRPC query latency; throughput scaling from 1 to N machines.

---

## Phase 8 — Python bindings & RAG layer (Mac + Kaggle GPU)

- [x] Python bindings (nanobind or pybind11) — nanobind; HNSW bound, including `selection`
- [x] BM25 inverted index in C++ (Lucene/Anserini-exact; SciFact nDCG@10 0.6789 = published)
- [x] Hybrid retrieval with reciprocal rank fusion
- [x] Cross-encoder reranking (HotpotQA BEIR subset; SciFact on Kaggle: no gain; `docs/rag-results.md`)
- [x] Cited answer generation with Claude Haiku (live runs done; `docs/rag-results.md`)
- [x] Embeddings on GPU for SciFact and full BEIR HotpotQA (5.2M passages, Kaggle T4)
- [ ] *Optional:* extra BEIR datasets (FiQA, NFCorpus) and full-corpus hybrid + rerank
- [ ] *Optional:* HotpotQA answer groundedness on the full 5.2M corpus (Kaggle stage2b, behind a cost estimate)
- [x] Evaluation scripts for BEIR and HotpotQA

**Measure:** nDCG@10 for keyword-only, vector-only, hybrid, hybrid + rerank (ablation table);
answer groundedness on HotpotQA.

---

## Phase 9 — Final results & polish (AWS session, then the Oracle machine)

- [x] Linux build and tests on GCC 13, AVX2 verified natively (IdeaPad, 2026-09-25; predates HNSW)
- [ ] Rebuild and test on Linux with HNSW and everything since (first thing on the Oracle machine)
- [ ] Full runs on SIFT1M and GloVe-100, averaged over multiple runs — AWS
- [ ] 10M-vector run — AWS
- [ ] Filtered-search crossover at 1M and 10M, including 1-3% selectivity (fallback rate) — AWS
- [ ] Thread scaling 1 → N cores, and multi-machine sharding scaling — AWS
- [ ] x86 AVX2 results (AWS) + final ARM NEON results (Oracle machine)
- [ ] README: one-line summary, recall-QPS chart vs. FAISS/hnswlib at the top, architecture diagram,
      results tables with hardware noted
- [ ] Design doc: graph parameters, compression, filtering strategies, sharding trade-offs
- [ ] Fill in resume bullets with real numbers
- [ ] Make the repo public

---

## Metrics tracker

| Metric | How measured | Phase | Resume? | Result |
|---|---|---|---|---|
| Recall@10 vs. QPS (vs. FAISS, hnswlib) | ann-benchmarks methodology, SIFT1M & GloVe-100 | 2, 9 | Yes | |
| p99 query latency | Benchmark harness | 2, 9 | README | |
| SIMD speedup (AVX2 and NEON) | Kernel benchmark vs. scalar | 3 | Yes | |
| Multithreaded QPS scaling | 1 → 16 threads | 3 | README | |
| Index build time | Timed per dataset | 3, 9 | README | |
| Crash recovery correctness | Repeated kill/restart tests | 4 | README | |
| Memory reduction with PQ, recall retained | Compression sweep | 5 | Yes | |
| Filtered search recall/QPS by selectivity | 1% / 10% / 50% | 6 | README | |
| Sharded throughput scaling | 1 → N VMs | 7 | Maybe | |
| nDCG@10 (BEIR) | Keyword / vector / hybrid / hybrid+rerank | 8 | Yes | |
| Answer groundedness | HotpotQA subset | 8 | Yes | |

---

