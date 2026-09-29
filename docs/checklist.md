# Strata — Checklist

Where each item of `buildplan.md` stands, and which machine it needs.

**Status as of 2026-09-28:** the HNSW core is built and measured (AI-implemented at my request in
reviewed stages; see `CLAUDE.md` and `docs/explainers/hnsw.md`). What remains is the finish order
below. Tests: 188 C++ (debug, ASan + UBSan, TSan) and 135 Python, passing on the Mac. Every number
so far is a Mac development result: recall is final, speed is indicative.

Machines:
- **Mac:** MacBook Air M2, 8 GB, fanless; it shut down under sustained load once. Code, tests,
  small datasets, and the 200k SIFT subset only; one heavy job at a time behind the checks in
  `bench/benchmeta.py` (`preflight`).
- **Oracle ARM machine:** gRPC server and sharding, final ARM numbers.
- **AWS (one session, x86):** final benchmarks, the 10M run, thread and multi-machine scaling, `perf`.
- **Kaggle T4:** GPU work for the RAG layer (done).
- **IdeaPad:** Linux bring-up on 2026-09-25; not in the finish order (disk full).

---

## ✅ Completed

### Phase 0 — Setup
- [x] Toolchain: Xcode CLT, Homebrew, CMake, Ninja, vcpkg (`~/vcpkg`)
- [x] Git identity, GitHub repo (github.com/JCHETAN26/strata), `CLAUDE.md`
- [x] CMake + vcpkg manifest, GoogleTest, Google Benchmark
- [x] Presets: `debug`, `release`, `asan`, `tsan`, `rosetta-avx2`, plus `linux-*` (GCC 13)
- [x] clang-format / clang-tidy configs
- [x] Dataset scripts: `prepare_datasets.py` (HDF5 / fvecs → `.fbin` / `.ibin`), `make_subset.py`,
      `make_clustered.py`
- [x] SIFT10K, SIFT1M, the 200k SIFT subset, and a synthetic clustered set

### Phase 1 — Baseline
- [x] Scalar distances, brute-force search, recall@k (by id and tie-aware)
- [x] Benchmark harness: QPS, latency percentiles, raw JSON with commit and hardware recorded

### Phase 2 — HNSW
- [x] Core (`src/index/hnsw.cpp`): level assignment, layer search, insertion, the paper's
      neighbor-selection heuristic (switchable against closest-M), M / efConstruction / efSearch
- [x] Spec tests (edge cases, graph invariants, recall vs. brute force), concurrent-search (TSan)
      and self-aliasing (ASan) tests
- [x] Python bindings with exact agreement with C++ on SIFT10K
- [x] Heuristic vs. closest-M comparison (`results/selection/`): +0.15–0.19 recall on clustered data
- [x] Recall-QPS vs. hnswlib and FAISS on SIFT10K, SIFT1M, and the 200k subset (`results/hnsw/`);
      recall matches hnswlib; speed indicative (M2; hnswlib has no NEON path)
- [x] Explainer: `docs/explainers/hnsw.md`

### Phase 3 — Performance
- [x] NEON and AVX2 + FMA kernels, compile-time dispatch, scalar fallback, tolerance tests
- [x] Thread pool, parallel batch search
- [x] Contiguous vector storage, compact neighbor lists
- [x] Profiling on the Mac with `sample` (`bench/profile_hnsw_search.sh`, `results/profiles/`):
      refuted the per-query-allocation hypothesis (~1.5% of search time)
- [x] Prefetching in `search_layer`: ~2x QPS on the 200k subset, results bit-identical
      (`results/ab/prefetch-sift1m-200k.md`); no measurable change on cache-resident SIFT10K
- [x] Interleaved A/B tool for recording each optimization's effect (`bench/run_ab_search.py`)
- [x] Prefetching in the upper-layer walk (`greedy_search`): no gain distinguishable from noise on
      the M2 (`results/ab/prefetch-greedy-sift1m-200k.md`), so not kept; patch saved there

### Phase 4 — Persistence & crash recovery
- [x] Snapshots (vectors + tombstones): save/load, checksummed, written atomically
- [x] Write-ahead log with CRC32C and sequence numbers; torn writes vs. corruption handled
- [x] Tombstone deletes (brute force)
- [x] Crash tests: 49 rounds of kill-and-restart, no acknowledged write lost

### Phase 5 — Product quantization
- [x] k-means, product quantizer, ADC lookup tables, exact reranking, memory-vs-recall chart

### Phase 6 — Filtered search
- [x] Metadata attributes and filter expressions
- [x] Strategy A: pre-filter + brute force, with a benchmark and a crossover chart

### Phase 8 — Python bindings & RAG
- [x] Bindings (nanobind): brute force, PQ, filtered search, distances, BM25, HNSW
- [x] BM25 in C++, reproducing Anserini's SciFact BM25 exactly
- [x] Hybrid retrieval (RRF + train-tuned weighted fusion)
- [x] Cross-encoder reranking (HotpotQA BEIR subset; SciFact on Kaggle, no gain)
- [x] Cited answer generation with Claude Haiku, live runs done (`docs/rag-results.md`)
- [x] Two-hop retrieval with joint reranking, end to end (answer F1 0.517 → 0.669)
- [x] GPU embeddings: SciFact and full BEIR HotpotQA (5.2M passages; nDCG@10 matches published)

### Linux
- [x] Build and all tests on GCC 13, AVX2 verified natively (IdeaPad, 2026-09-25,
      `docs/reports/linux-bringup.md`). **Gap:** this predates the HNSW core; see finish order 4.

---

## 🏁 Finish order (adopted 2026-09-28)

1. **HNSW persistence and deletes** (Mac)
   - [ ] Save/load the HNSW graph in snapshots
   - [ ] HNSW tombstone deletes (skipped during search, still used for navigation)
2. **Parallel index build** (Mac for correctness; scaling numbers on AWS)
   - [ ] Concurrent insertion with per-node locks on the thread pool
3. **Filtered HNSW search** (Mac on subsets; final crossover on AWS)
   - [ ] Strategy B: filter during graph traversal
   - [ ] Automatic strategy selection from the measured crossover (`estimate_selectivity` exists)
4. **gRPC server and sharding** (Oracle ARM machine)
   - [ ] First: rebuild and run all tests there (Linux, ARM, with HNSW)
   - [ ] gRPC API: insert, search, delete
   - [ ] Shards + coordinator (parallel scatter, gather and merge top-k)
   - [ ] Docker Compose with multiple shards
5. **One AWS session (x86)**
   - [ ] SIFT1M and GloVe-100 recall-QPS vs. FAISS and hnswlib, several runs each, with variance
   - [ ] 10M-vector run (use PQ if memory is tight)
   - [ ] Thread scaling 1 → N cores
   - [ ] Multi-machine sharding scaling
   - [ ] `perf` profiling of `search_layer`'s own loop (visited marks, heap, redundant prefetch
         hints), deferred from the Mac
   - [ ] Re-measure prefetching's gain on x86 (Apple's hardware prefetchers may differ), and
         re-test the saved `greedy_search` prefetch patch there
   - [ ] Storage benchmark at a size where WAL vs. snapshot recovery differ
6. **Final ARM numbers on the Oracle machine**
   - [ ] Same benchmark set as AWS where it applies
7. **Publish**
   - [ ] README: recall-QPS chart at the top, architecture diagram, results tables with hardware
   - [ ] Design doc: graph parameters, compression, filtering strategies, sharding trade-offs
   - [ ] Resume bullets with real numbers
   - [ ] Make the repo public

---

## Optional (only if time allows)

- [ ] PQ inside HNSW
- [ ] WAL group commit (durable inserts are capped at ~330/s by one fsync each)
- [ ] Extra BEIR datasets (FiQA, NFCorpus) and full-corpus hybrid + rerank

## Not placed yet (needs a decision)

- [ ] HotpotQA answer groundedness on the full 5.2M-passage corpus (Kaggle stage2b, behind a cost
      estimate and `--enable-api`)

## Setup still pending on the Mac
- [ ] Add `export VCPKG_ROOT=~/vcpkg` to `~/.zshrc` (non-interactive shells don't see it; the Python
      module rebuild needs it)
- [ ] `brew install llvm` (for command-line clang-tidy)
