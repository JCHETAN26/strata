# Strata — Checklist

Where each item of `buildplan.md` stands, and which machine it needs.

- **Mac:** MacBook Air M2, 8 GB, fanless. Fine for code, tests, and small datasets (SIFT10K).
- **IdeaPad:** Ryzen 7 5800H (AVX2), ~19 GiB RAM, RTX 3050, Ubuntu 22.04. Needed for Docker,
  GPU work, large datasets, and every number that goes in the README.

Status as of 2026-09-25: **29 of 59 plan items done.** 125 tests pass under debug, ASan + UBSan,
and TSan on the Mac.

---

## ✅ Completed

### Phase 0 — Setup
- [x] Toolchain: Xcode CLT, Homebrew, CMake, Ninja, vcpkg (`~/vcpkg`)
- [x] Git identity, GitHub repo (github.com/JCHETAN26/strata), `CLAUDE.md`
- [x] CMake + vcpkg manifest, GoogleTest, Google Benchmark
- [x] Presets: `debug`, `release`, `asan`, `tsan`, `rosetta-avx2`
- [x] clang-format / clang-tidy configs
- [x] Dataset script (`scripts/prepare_datasets.py`): HDF5 / fvecs → `.fbin` / `.ibin`
- [x] SIFT10K and SIFT1M downloaded and converted

### Phase 1 — Baseline
- [x] Scalar distances: L2, inner product, cosine
- [x] Brute-force exact search
- [x] Recall@k, both by id and tie-aware (the ann-benchmarks definition)
- [x] Benchmark harness: QPS, latency percentiles, raw JSON with commit and hardware recorded
- [x] Unit tests for distances and recall

### Phase 2 — HNSW (only the parts around the core)
- [x] Public API: `include/strata/hnsw.hpp`
- [x] Test suite as spec: `tests/hnsw_test.cpp` (edge cases, graph invariants, recall), with
      thresholds checked against hnswlib
- [x] hnswlib and FAISS comparison scripts, recall-vs-QPS plot

### Phase 3 — Performance (non-HNSW parts)
- [x] NEON and AVX2 + FMA kernels, compile-time dispatch, scalar fallback
- [x] Tests: every SIMD kernel matches scalar within a documented tolerance, and exactly on
      integer data
- [x] Thread pool, parallel batch search, `--threads` in the harness

### Phase 4 — Persistence & crash recovery
- [x] Snapshots: save/load, checksummed, written atomically
- [x] Write-ahead log with checksums (CRC32C) and sequence numbers; torn writes vs. corruption
      handled
- [x] Tombstone deletes (brute force)
- [x] Crash tests: 49 rounds of kill-and-restart, no acknowledged write lost

### Phase 5 — Product quantization
- [x] k-means, product quantizer, lookup-table (ADC) distances
- [x] Exact reranking of the top candidates
- [x] Memory-vs-recall chart

### Phase 6 — Filtered search
- [x] Metadata attributes and filter expressions
- [x] Strategy A: pre-filter + brute force, with a benchmark and a crossover chart

---

## 🧠 Needs you: HNSW core (Mac is fine)

The core algorithm is hand-written (see `CLAUDE.md`). Creating `src/index/hnsw.cpp` switches on
the test spec and `--index hnsw` in every benchmark.

- [ ] Random level assignment
- [ ] Layered graph insertion
- [ ] Neighbor-selection heuristic from the paper
- [ ] Search: greedy descent through upper layers, efSearch-bounded search at layer 0
- [ ] Parameters `M`, `efConstruction`, `efSearch` wired through

After the core works (I can build these, on the Mac):
- [ ] Contiguous vector storage, compact neighbor lists, prefetching (you review)
- [ ] Profile hotspots with Instruments
- [ ] Record the effect of each optimization separately
- [ ] Parallel index build (the thread pool is ready)
- [ ] Combine PQ with HNSW
- [ ] Strategy B: filter during graph traversal
- [ ] Automatic filter-strategy selection (crossover measured on SIFT10K now, re-measured on the
      IdeaPad)
- [ ] HNSW deletes (tombstones) and saving the graph in snapshots

---

## 💻 Can be built on the Mac now (no heavy compute)

- [x] Python bindings (nanobind): brute force, PQ, filtered search, distances; HNSW switches on with the core
- [x] BM25 inverted index in C++, with bindings — reproduces Anserini's SciFact BM25 exactly
- [x] Hybrid retrieval with reciprocal rank fusion (+ train-tuned weighted fusion); SciFact nDCG@10 0.7273 RRF / 0.7316 weighted vs 0.6789 BM25, 0.7127 dense
- [x] Cited answer generation with Claude Haiku (sentence-level native citations; live run needs ANTHROPIC_API_KEY)
- [x] BEIR and HotpotQA evaluation scripts (BEIR runner with significance; HotpotQA answer + groundedness eval, retrieval measured, generation awaiting API key)
- [ ] Cross-encoder reranking (runs on the CPU for small subsets; the full evaluation goes on the
      IdeaPad)
- [ ] Design doc drafts: graph parameters, compression, filtering strategies
- [ ] Group commit for the WAL (optional; durable inserts are capped at ~330/s by one fsync each)

---

## 🖥️ Needs the IdeaPad (more compute, Linux, x86, or GPU)

### First, before anything else
- [ ] **Build and run all tests on Linux / GCC.** Nothing has been built on Linux yet, and
      `CLAUDE.md` requires both platforms. Run the `debug`, `asan`, and `tsan` presets.
- [ ] **Check the AVX2 kernels on real hardware.** So far they have only been run through Rosetta
      2 on the Mac, which is enough for correctness but not for timing.

### Phase 7 — Server & sharding
- [ ] gRPC API: insert, search, delete (gRPC is a very heavy vcpkg build for 8 GB)
- [ ] Sharding: vectors partitioned across shards
- [ ] Coordinator: parallel scatter to all shards, gather and merge top-k
- [ ] Docker Compose running multiple shards
- [ ] Brief cloud run on 3–5 VMs for multi-machine scaling (cloud, not the IdeaPad)

### Phase 8 — Compute-heavy parts
- [ ] Embeddings for the BEIR corpora (SciFact, FiQA, NFCorpus) on the RTX 3050
- [ ] Full BEIR nDCG@10 runs: keyword-only, vector-only, hybrid, hybrid + rerank
- [ ] HotpotQA answer-groundedness evaluation

### Phase 9 — Final results (all reported numbers)
- [ ] SIFT1M and GloVe-100 full runs, several runs each, with variance
      (the Mac is too noisy: runs varied 30–60% under background load)
- [ ] 10M-vector run (~5 GB raw; use PQ if memory is tight)
- [ ] x86 AVX2 results on the IdeaPad
- [ ] Thread-scaling curve from 1 to 16 threads (the Mac has only 4 performance cores)
- [ ] Storage benchmark with a larger dataset (so WAL vs. snapshot recovery actually differ)
- [ ] Filtered-search crossover chart, including HNSW in-graph filtering
- [ ] Linux profiling with `perf`

### Last, on the Mac
- [ ] ARM NEON results at 1M scale: the one long benchmark allowed on the Mac. Close other apps
      first.

---

## 📝 After results exist (any machine)

- [ ] README: recall-QPS chart vs. FAISS/hnswlib at the top, architecture diagram, results tables
      with hardware
- [ ] Final design doc
- [ ] Resume bullets with real numbers
- [ ] Make the repo public

---

## Setup still pending on the Mac
- [ ] Add `export VCPKG_ROOT=~/vcpkg` to `~/.zshrc`
- [ ] `brew install llvm` (for clang-tidy)
