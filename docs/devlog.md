# Dev log

## 2026-09-25 — Phase 0 setup

**Done**
- Repo created at github.com/JCHETAN26/strata.
- CMake + Ninja + vcpkg (manifest mode, baseline pinned in `vcpkg.json`), presets `debug`,
  `release`, `asan`. Smoke test and smoke benchmark build and pass under all three presets.
- `.clang-format` (Google style, 100 cols) and `.clang-tidy`.
- `scripts/prepare_datasets.py`: downloads SIFT10K (TEXMEX `.fvecs`) and ann-benchmarks HDF5
  (SIFT1M, GloVe-100) and converts to `.fbin`/`.ibin`. SIFT10K converted; ground truth checked
  against a numpy brute-force search (100% top-10 agreement).

**Decisions**
- **Binary format:** big-ann-benchmarks `.fbin`/`.ibin` (8-byte `n, d` header + raw values).
  Simple, mmap-friendly, and a known format, so other tools can read our files.
- **Ground truth source:** use the dataset's published neighbors rather than recomputing, and
  record the download's SHA-256 in `meta.json` for reproducibility.
- **Sanitizer preset** doesn't force `detect_leaks`: LeakSanitizer isn't supported by Apple clang
  on arm64 macOS, and it's on by default on Linux, where it matters.
- **Warnings:** `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion` on all Strata
  targets via the `strata_options` interface library. Not `-Werror` yet.
- **Python module name:** `prepare_datasets`, not `datasets`, to avoid shadowing Hugging Face
  `datasets` when the RAG layer arrives.

**Problems**
- `BUILDPLAN.md` vs. `buildplan.md`: macOS's case-insensitive filesystem hid the mismatch with the
  reference in `CLAUDE.md`. Renamed to lowercase.
- GitHub push failed on an expired `gh` token; fixed by re-running `gh auth login`.

**Open**
- `clang-tidy` isn't installed on the Mac (Apple's toolchain doesn't ship it): `brew install llvm`.

## 2026-09-25 — Phase 1 baseline

**Done**
- Scalar distance kernels (L2², negated inner product, cosine distance), dispatched through a
  function pointer looked up once per search, not per distance.
- `BruteForceIndex`: contiguous storage, dense ids, bounded max-heap top-k, results sorted by
  `(distance, id)`. `Matrix<T>`, `.fbin`/`.ibin` readers and writers, `recall_at_k`.
- Harness `strata_search` (C++) + `bench/run_search_bench.py` (metadata, repeated runs, mean ±
  stdev) + `bench/make_tables.py`. Google Benchmark microbenchmarks for the distance kernels.
- 38 C++ tests (including SIFT10K end-to-end recall = 1.0), clean under ASan + UBSan.

**Decisions**
- **Error type:** `tl::expected` behind `strata::Expected<T>`, so moving to `std::expected` later
  is an alias change.
- **Distance convention:** every metric is a distance (lower = closer). Inner product returns
  `-dot` rather than hnswlib's `1 - dot`; ordering is identical and it has no magic constant.
  Cosine with a zero vector is defined as 1 (orthogonal) rather than NaN, so it never poisons a heap.
- **Scalar reference accumulates in float, left to right.** Clang won't auto-vectorize this
  (reassociation changes the result), which is what makes it a stable reference for SIMD tests.
- **Recall is by id.** Ties at the k-th distance can undercount; noted in `recall.hpp`.
- **Harness methodology:** one untimed warmup pass per run (first run was ~25% slower without
  it), queries timed one at a time, single thread. Driver refuses non-Release builds.
- **Layout:** added `src/eval/` for recall (not in the original layout in `CLAUDE.md`).

**Measured (Apple M2, single thread, scalar L2; `results/tables.md`)**
- SIFT10K brute force: ~1,600 QPS, p50 ≈ 620 µs, recall@10 = 1.0.
- SIFT1M brute force (200 queries): ~14 QPS, p50 ≈ 65 ms. p99 is noisy on the laptop (±60 ms).
- Scalar L2 at d=128: ~63 ns/op, about 2 GFLOP/s: the baseline SIMD has to beat in Phase 3.

**Problems**
- **Benchmark noise on the M2.** Runs occasionally dropped 30–40% mid-run. Two causes: background
  load (load average 4–6 during the first attempt) and macOS moving the process to efficiency
  cores. The harness now sets `QOS_CLASS_USER_INTERACTIVE`, and the driver records load average
  and median. The laptop is still not a quiet machine; final numbers come from the IdeaPad.
- **SIFT1M brute force scored recall 0.9995 by id.** Query 170 has two vectors tied at the 10th
  distance (40644, exact: SIFT features are integers) and the ground truth picked the other one.
  Added tie-aware recall (ann-benchmarks definition) as the headline and kept id recall alongside.
- **ann-benchmarks.com returns 403** to Python's default User-Agent; the download sets one.

## 2026-09-25 — Phase 2 scaffolding (HNSW core is hand-written; see CLAUDE.md)

**Done**
- `include/strata/hnsw.hpp`: public API plus introspection (`entry_point`, `max_level`, `level`,
  `neighbors`) so tests can check graph invariants. Private section left for the hand-written
  implementation.
- `tests/hnsw_test.cpp`: spec: edge cases (empty, single, k=0, k>size, ef<k, dimension
  mismatch, 100 identical vectors), result contract (sorted, distinct, exact distances), graph
  invariants (degree bounds, no self-loops or duplicate edges, neighbors live on the layer, entry
  point on top layer, level distribution vs mL = 1/ln M, layer-0 reachability), and recall vs
  brute force (all metrics, ef monotonicity, incremental inserts, SIFT10K >= 0.98).
- CMake compiles HNSW, its tests, and `--index hnsw` in the harness only when
  `src/index/hnsw.cpp` exists.
- Harness restructured: build once, sweep `ef_search`. Shared record format (`bench/records.py`)
  for Strata, hnswlib, and FAISS; `bench/run_reference_bench.py`; `bench/plot_recall_qps.py`.

**Decisions**
- **Spec thresholds checked against hnswlib** on the same data shapes: recall >= 0.996 where the
  tests require 0.95, 0.998 on SIFT10K where they require 0.98. hnswlib also passes the duplicates
  case. So a failure means a bug, not a harsh test.
- **Reference QPS from one batched call** (no per-query Python overhead). Their latency
  percentiles are per-call and include Python overhead; stated in the table notes.

**Observed**
- On the M2, FAISS HNSW is ~3.5x faster than hnswlib at the same M/ef (200k vs 57k QPS at
  ef=10). Checked it's single-threaded (wall = CPU time with `omp_set_num_threads(1)`). Likely
  cause: the FAISS wheel is built with NEON, while hnswlib's SIMD paths are SSE/AVX only, so on ARM
  it runs scalar distances. Expect hnswlib to look much stronger on the Ryzen.
- FAISS flat (BLAS) brute force: ~10.5k QPS vs Strata scalar brute force ~1.6k on SIFT10K.

## 2026-09-25 — Phase 3: SIMD and threads

**Done**
- NEON (arm64) and AVX2+FMA (x86_64) kernels for L2, inner product, cosine. Compile-time
  selection; `KernelSet::kScalar` keeps the reference selectable, and `--kernel scalar` gives the
  "without SIMD" line for every comparison.
- `ThreadPool` (dynamic chunking over an atomic counter, caller participates),
  `BruteForceIndex::search_batch`, harness `--threads N`.
- Presets: `tsan` (ThreadSanitizer) and `rosetta-avx2` (x86_64 AVX2 build on macOS, run under
  Rosetta 2, so AVX2 kernels are tested before they reach the Ryzen).

**Measured (Apple M2; `results/tables.md`)**
- Kernels at d=128: L2 66 → 7.9 ns (8.4x), IP 55 → 6.9 ns (~8x), cosine 81 → 13.6 ns (~6x).
- SIFT10K brute force, 1 thread: 1,583 → 8,321 QPS (5.3x end to end; heap updates and loop
  overhead don't vectorize).
- Thread scaling (NEON): 1 → 2 → 4 → 8 threads: 8.3k → 17.1k → 26.2k → 41.9k QPS. Past 4 threads
  the work lands on efficiency cores; the curve is noisy because of background load on the laptop.

**Decisions**
- **4 accumulators on NEON, 2 on AVX2.** FMA latency is ~4 cycles on both, so one accumulator
  stalls every iteration; independent accumulators let consecutive FMAs overlap. Each loop
  iteration covers 16 floats either way.
- **Tolerance:** `|simd - scalar| <= 1e-5 * sum|term_i|`. Worst observed: 23% of the bound
  (NEON L2) over dims 0..130 and 255..4096. Integer inputs (SIFT) match exactly.
- **Rosetta 2 executes AVX2/FMA** (CPUID doesn't advertise them). Good enough for correctness
  tests; timings under translation are meaningless and never recorded.

**Problems**
- **ThreadSanitizer caught a real bug** in the first thread pool: `workers_` (the `jthread`s) was
  declared first, so member destruction joined the threads *last*, after the mutex and condition
  variables they were waiting on were gone. The destructor now joins explicitly.
- **Result files overwrote each other**: three fast runs finished within one second and shared a
  filename. Filenames and timestamps now carry microseconds, and writes refuse to overwrite.
- **SIFT1M on the laptop is unreliable** (runs varied 30–61 QPS within one set; a Chrome tab at
  ~50% CPU, a VM, and a system service were competing). Not used for any claim; the Ryzen will
  produce the SIFT1M numbers.

**Left for the HNSW implementation (hand-written):** compact neighbor lists, prefetching, parallel
build (the `ThreadPool` is ready for it), and profiling the graph search.

## 2026-09-25 — Phase 4: persistence and crash recovery

**Done**
- `WriteAheadLog` (CRC32C + LSN per record), `Snapshot` (checksummed, atomic write),
  tombstone deletes in `BruteForceIndex`, and `Collection` (WAL-first writes, checkpoint =
  snapshot + WAL reset, shared/exclusive locking).
- Crash tests: a forked writer is SIGKILLed at random points across 49 rounds (WAL only,
  frequent/rare checkpoints, fsync); every acknowledged write must survive intact. Plus
  byte-level tests: every snapshot byte flip and truncation, WAL torn tail at every offset,
  mid-log corruption, crash between snapshot and WAL reset.
- `strata_storage_bench`: SIFT10K on the M2: ~290k inserts/s without sync, ~330/s with
  `F_FULLFSYNC` (p99 ~4.9 ms), checkpoint 31 ms, recovery ~15 ms.

**Decisions**
- **Torn tail vs corruption.** A bad record is treated as a torn write (and truncated) only if
  it reaches EOF *and* the remaining bytes fit in one record. Anything else fails the open
  with `kCorruptData`: silently truncating would drop acknowledged writes.
- **Acknowledgment = WAL append returned.** With `kNone` that survives a process crash; with
  `kFsync` it also survives power loss. On macOS `fsync` alone only reaches the drive cache, so
  `kFsync` uses `F_FULLFSYNC`.
- **Snapshots are index-independent** (vectors + tombstones + LSN). Brute force rebuilds
  trivially; the HNSW graph can be added to the snapshot once it exists (rebuilding from vectors
  is the fallback).
- **Checkpoint ordering.** Snapshot first (atomic rename), then WAL reset (also an atomic
  rename). A crash between them leaves WAL records the snapshot already holds; recovery skips
  records with LSN <= the snapshot's LSN.
- **No group commit yet.** One fsync per insert caps durable inserts at ~330/s here. Batching
  concurrent writers behind one fsync is the obvious next step if write throughput matters.

**Problems**
- **Review found a hole the first tests missed:** a corrupted *length* field in the middle of the
  log made the record appear to run past EOF, so recovery classified it as a torn tail and
  would have truncated every later (acknowledged) record. Added the one-record size bound and a
  test that fails without it (checked by temporarily reverting the fix).
- **The first crash test was wrong, not the code:** it demanded that a vector be present when its
  delete had become durable but the delete's ack hadn't reached the parent before SIGKILL.
  Unacknowledged writes may or may not survive; the test now allows exactly that.
- **ThreadPool-style destruction order** was not an issue here: `Collection` holds its
  `shared_mutex` behind a `unique_ptr` so the class stays movable.
