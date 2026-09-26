# Strata — Build Plan

**Strata** is a distributed vector search engine written from scratch in C++20, with a RAG layer on top.
Given millions of vectors, it finds the most similar ones in milliseconds, and uses that to answer
questions from documents with cited sources.

**What it proves:** algorithms (from-scratch HNSW), systems depth (SIMD, concurrency, persistence,
sharding), and applied AI (hybrid retrieval, reranking, RAG), benchmarked against FAISS and hnswlib.

---

## Ground rules

- **Understand HNSW yourself.** Read the paper (Malkov & Yashunin, *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs*). Write the core insertion
  and search logic yourself, or be able to explain every line. Interviewers will ask.
- **AI handles scaffolding:** build files, harnesses, bindings, tests, scripts.
- **Every result comes from a script** that saves raw data and generates its table or chart.
- **Always compare against a baseline** (brute force, hnswlib, FAISS, scalar vs. SIMD, with vs. without).
- **Long benchmarks run on the IdeaPad only.** The MacBook Air is fanless and throttles under sustained load.
- **Commit after every working step.**

---

## Hardware

| Machine | Specs | Role |
|---|---|---|
| MacBook Air | M2 (ARM), 8 GB | Phases 0–6 development, small datasets, ARM NEON SIMD, ARM benchmark results |
| IdeaPad | Ryzen 7 5800H (8C/16T, AVX2), ~19 GiB usable RAM, RTX 3050 4 GB, Ubuntu 22.04 | Docker, gRPC server, embeddings on GPU, x86 AVX2 SIMD, all final large-scale benchmarks |
| Cloud VMs (brief) | 3–5 small instances, a few hours | Multi-machine sharding scaling numbers |

Code moves between machines via GitHub: push from the Mac, pull on the IdeaPad.
Note for results: the IdeaPad's RAM is mixed 8 GB + 16 GB (partly single-channel). State this in hardware notes.

---

## Datasets

| Dataset | Size | Used for |
|---|---|---|
| SIFT10K (siftsmall) | 10K × 128-d | Fast iteration and correctness tests (Mac) |
| SIFT1M | 1M × 128-d, ~500 MB | Main benchmark (Mac for dev, IdeaPad for results) |
| GloVe-100 | ~1.2M × 100-d | Harder benchmark, angular distance |
| 10M-vector set (e.g., BIGANN/SIFT subset) | ~5 GB raw | Large-scale run on IdeaPad (use PQ if memory is tight) |
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

- [ ] Random level assignment
- [ ] Layered graph insertion
- [ ] Neighbor-selection heuristic from the paper (not just "closest M")
- [ ] Search: greedy descent through upper layers, `efSearch`-bounded search at layer 0
- [ ] Parameters: `M`, `efConstruction`, `efSearch`
- [x] Tests: recall vs. brute force; edge cases (empty index, one vector, duplicates) — spec in `tests/hnsw_test.cpp`
- [x] Python comparison scripts for hnswlib and FAISS on the same data

**Measure:** recall vs. QPS curve against hnswlib and FAISS on SIFT10K and SIFT1M.
First version will likely be slower than hnswlib. That's expected.

---

## Phase 3 — Performance (1–2 weeks, Mac → IdeaPad)

- [x] SIMD distance kernels: NEON (Mac), AVX2 (IdeaPad), scalar fallback
- [x] Test: all kernels return identical results to scalar
- [ ] Contiguous vector storage, compact neighbor lists, prefetching
- [x] Thread pool: parallel batch queries (parallel index build waits for HNSW)
- [ ] Profile hotspots (Instruments on Mac, `perf` on Linux)
- [ ] Record the effect of each optimization separately

**Measure:** SIMD speedup vs. scalar; QPS scaling across cores; updated recall-QPS curve.

---

## Phase 4 — Persistence & crash recovery (≈1 week, Mac)

- [x] Save and load index to/from disk (snapshot of vectors + tombstones; HNSW graph pending)
- [x] Write-ahead log: log every insert before applying it
- [x] Deletes via tombstones (brute force; HNSW pending)
- [x] Crash tests: kill mid-insert, restart, verify no acknowledged write is lost or corrupted

**Measure:** save/load time; recovery correctness across repeated crash tests.

---

## Phase 5 — Product quantization (≈1 week, Mac)

- [ ] Split vectors into sub-vectors; k-means codebook per sub-space
- [ ] Asymmetric distance computation with precomputed lookup tables
- [ ] Re-rank top candidates with full-precision vectors
- [ ] Combine PQ with HNSW

**Measure:** memory reduction and recall at several compression levels (memory vs. recall chart).

---

## Phase 6 — Filtered search (≈1 week, Mac)

- [ ] Metadata attributes per vector (e.g., year, category)
- [ ] Strategy A: pre-filter + brute force (best for very selective filters)
- [ ] Strategy B: filter during graph traversal (best for broad filters)
- [ ] Automatic strategy selection based on estimated selectivity

**Measure:** recall and QPS at 1%, 10%, 50% filter selectivity for each strategy (crossover chart).

---

## Phase 7 — Server & sharding (1–2 weeks, IdeaPad)

- [ ] gRPC API: insert, search, delete
- [ ] Sharding: vectors partitioned across shards
- [ ] Coordinator: parallel scatter to all shards, gather and merge top-k
- [ ] Docker Compose running multiple shards on the IdeaPad
- [ ] Brief cloud run on 3–5 VMs for real multi-machine scaling

**Measure:** end-to-end gRPC query latency; throughput scaling from 1 to N machines.

---

## Phase 8 — Python bindings & RAG layer (1–2 weeks, IdeaPad)

- [ ] Python bindings (nanobind or pybind11)
- [ ] BM25 inverted index in C++
- [ ] Hybrid retrieval with reciprocal rank fusion
- [ ] Cross-encoder reranking
- [ ] Cited answer generation with Claude Haiku
- [ ] Generate embeddings for BEIR corpora on the IdeaPad GPU
- [ ] Evaluation scripts for BEIR and HotpotQA

**Measure:** nDCG@10 for keyword-only, vector-only, hybrid, hybrid + rerank (ablation table);
answer groundedness on HotpotQA.

---

## Phase 9 — Final results & polish (≈1 week, IdeaPad)

- [ ] Full runs on SIFT1M and GloVe-100, averaged over multiple runs
- [ ] 10M-vector run
- [ ] x86 AVX2 results (IdeaPad) + ARM NEON results (Mac, 1M scale)
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

