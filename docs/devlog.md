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

## 2026-09-25 — Phase 5: product quantization

**Done**
- `kmeans` (Lloyd + k-means++, parallel assignment, empty-cluster splitting),
  `ProductQuantizer` (8-bit codes, ADC tables for L2 / IP / cosine), flat `PqIndex` with optional
  exact reranking. Harness `--index pq` sweeps rerank depth and reports memory.
- Harness now judges recall on exact distances recomputed for the returned ids, so indexes that
  return estimated distances (PQ without rerank) are scored correctly.

**Measured (SIFT10K, Apple M2, 1 thread; `results/plots/pq_memory_siftsmall.png`)**

| m (bytes/vector) | compression | recall@10 ADC only | recall@10 rerank 100 |
|---|---|---|---|
| 4 | 128x | 0.42 | 0.92 |
| 8 | 64x | 0.60 | 0.99 |
| 16 | 32x | 0.73 | 1.00 |
| 32 | 16x | 0.83 | 1.00 |
| 64 | 8x | 0.94 | 1.00 |

(Numbers from `results/tables.md`; this table is a summary for the log.)

**Decisions**
- **Rerank memory is reported separately.** Reranking reads full-precision vectors; the chart's
  x-axis counts codes only, as if originals were on disk. Stated on the chart.
- **Four accumulators in the ADC sum.** Measured: m=16 11.8k → 17.6k QPS, m=64 2.6k → 5.4k QPS.

**Weaknesses (honest)**
- On SIFT10K everything fits in cache, so NEON brute force (8.3k QPS) beats ADC at m=64
  (5.4k QPS): 64 table lookups cost more than one 128-d SIMD distance. PQ's win is memory, and
  speed only once the raw vectors no longer fit in cache/RAM. FAISS's 4-bit "fast scan" does ADC
  with SIMD shuffles; Strata's ADC is scalar.
- "Combine PQ with HNSW" waits for the HNSW implementation.

## 2026-09-25 — Phase 6: filtered search (non-HNSW parts)

**Done**
- `AttributeTable` (columnar int + interned category columns), `Filter` expressions
  (equals / range / in, all_of / any_of / negate), `CompiledFilter` (per-id check, `Bitset`
  evaluation, sampled selectivity estimate).
- Strategy A: `BruteForceIndex::search_filtered` (iterate set bits of the pre-filter bitset) and
  `search_predicate` (scan and check) as its baseline.
- `strata_filter_bench`: synthetic `bucket = hash(id) % 1000` attribute for exact selectivity,
  exact filtered ground truth, filter evaluation timed per query. `bench/plot_filter.py`.

**Measured (SIFT10K, Apple M2; noisy, see below)**
- At 1% selectivity pre-filtering is ~1.6x faster than scan-and-check (42.8k vs 26.5k QPS); at
  10–50% they converge because distance computations dominate.
- At 100% selectivity, median latency is ~140 µs vs ~120 µs unfiltered: filter evaluation costs
  ~2 ns per row.

**Problems**
- The laptop was under background load again (load average ~5.7): individual runs of the same
  configuration varied 3.0k–6.9k QPS. Medians are stable; means and the chart's error bars are
  not. Final filtered-search numbers come from the Ryzen.

**Waits for HNSW**
- Strategy B (filter during graph traversal) and automatic strategy selection. The selection
  threshold should come from the measured crossover between pre-filter + brute force and
  in-graph filtering, not a guess; `estimate_selectivity` is ready for it.

## 2026-09-25 — Python bindings

**Done**
- nanobind module `strata._core`, built by scikit-build-core (`pip install -e .`): distances,
  `BruteForceIndex` (incl. `search_filtered` with a compiled filter, bool mask, or id array),
  `ProductQuantizer` / `PqIndex`, `AttributeTable` / `Filter` / `CompiledFilter`, `build_info()`.
  `HnswIndex` is bound when `STRATA_HAS_HNSW` is set and otherwise raises `NotImplementedError`.
- `MatrixView<const T>` (non-owning) for read-only batch APIs, including
  `HnswIndex::add_batch` in the header. `Matrix` converts implicitly, so C++ callers are unchanged.
- `tests/strata_reference` writes C++ results on SIFT10K; `tests/python/test_bindings.py`
  (37 tests) checks that the bindings agree with them, plus conversions, padding, errors, locking,
  filters, PQ.

**Decisions**
- **Results:** padded `(ids, distances)`, FAISS style (`-1` / `inf`).
- **Input conversion:** float32 C-contiguous is a zero-copy view; anything else is converted
  with one copy (nanobind implicit conversion). Python lists are rejected with `TypeError`.
- **Locking:** each bound index has a `shared_mutex`. Searches share it; `add`/`remove` take it
  exclusively, so Python inserts are serialized. The GIL is always released *before* taking the
  lock (the opposite order can deadlock). Filter bitsets are evaluated with the GIL held,
  because `AttributeTable.append` also runs under the GIL. Appending after `compile()`
  invalidates the compiled filter (it raises instead of reading reallocated columns).
- **Floating point is now pinned:** `-ffp-contract=off -fno-fast-math` on every target
  (PUBLIC on `strata`). Clang contracts `a*b + c` into FMA by default and GCC doesn't in
  `-std=c++20` mode, so the scalar "reference" kernel used to compute different bits depending
  on the compiler. Explicit FMA intrinsics in NEON/AVX2 kernels are unaffected. The flags,
  compiler, kernel, and target go into a generated `build_info.hpp`.
- **Python-vs-C++ comparison:** ids and PQ codes must match exactly. Distances must be
  bit-identical when kernel, compiler, FP flags, and target all match (they do when both are
  built from this CMakeLists on one machine; a test asserts it). Otherwise they must be within
  a relative 1e-5 (`DISTANCE_RTOL`, the SIMD tolerance scale). Both branches are tested.
- **vcpkg:** gtest/benchmark moved behind a `tests` feature (on in the presets), so the Python
  build only fetches `tl-expected`.
- **HNSW switch-on:** CMake now re-checks for `src/index/hnsw.cpp` whenever `src/index/`
  changes (the directory is a configure dependency), so creating the file turns HNSW on at the
  next build in both the C++ and Python builds. Verified with a probe file (nothing was written
  under `src/index/hnsw*`). The HNSW bindings and `tests/hnsw_test.cpp` were syntax-checked
  with `STRATA_HAS_HNSW=1`.

- **Effect of `-ffp-contract=off` on speed:** checked at d=128 (median of 5, M2): scalar L2
  61 ns (was 66), IP 55 (59), cosine 90 (86); NEON unchanged (its FMAs are intrinsics). Within
  the laptop's noise, so the recorded scalar baselines still stand.
- All presets pass after the change: debug, asan, tsan, release, rosetta-avx2 (125 C++ tests).

**Problems**
- `ProductQuantizer.codebooks` failed: properties default to `reference_internal`, which can't
  apply to an array that already owns its buffer. Fixed with `rv_policy::move`.
- First draft evaluated filter bitsets with the GIL released, racing with
  `AttributeTable.append` on another Python thread. Caught in review before running.
- With build isolation, the persistent build dir caches the path of pip's temporary Python, so
  `cmake --build build/python/...` fails after the install. Rebuild with `pip install -e .`.

## 2026-09-25 — BM25 (C++, with bindings)

**Done**
- Text analysis (`include/strata/text.hpp`): UAX #29 word tokenizer (Lucene StandardTokenizer
  behaviour, utf8proc for Unicode classes, emoji tokens), plain tokenizer, English possessive
  filter, lowercase, Lucene's 33 English stopwords, Porter stemmer ported from Lucene.
  `AnalyzerConfig::anserini_english()` = Anserini's DefaultEnglishAnalyzer; `plain()` kept.
- `Bm25Index`: inverted index, Lucene BM25 (`idf = ln(1 + (N - df + 0.5)/(df + 0.5))`, k1=0.9,
  b=0.4), Lucene's float arithmetic and 1-byte length encoding (or exact lengths), query-term
  counts as boosts (Anserini), tombstone deletes, dense ids shared with the vector indexes.
- Bindings: `strata.Analyzer`, `strata.porter_stem`, `strata.Bm25Index` (GIL released,
  shared/exclusive lock like the vector indexes).
- `scripts/prepare_beir.py`, `bench/ir_eval.py` (trec_eval semantics), and
  `bench/validate_bm25_beir.py`.

**Result: SciFact reproduces Anserini's published BM25 flat numbers exactly**

| Metric | Strata | Anserini (published) |
|---|---|---|
| nDCG@10 | 0.6789 | 0.6789 |
| R@100 | 0.9253 | 0.9253 |
| R@1000 | 0.9767 | 0.9767 |
| Total terms in index | 838,127 | 838,128 |

Pass criterion (set before the run): each metric within 0.002 absolute, total terms within 0.1%.
Indexing 5,183 docs takes ~0.5 s; 300 queries at k=1000 run at ~34–44k QPS on all cores (M2).

**Reference pinned:** Anserini 2.3.0 (tag `anserini-2.3.0`, Lucene 10.5.0), configuration
`beir-v1.0.0-scifact.flat` (not multifield), with links to the published numbers at that tag, in
`results/bm25/ANSERINI_REFERENCE.md` and `anserini_reference.json` (the validation script reads
the JSON, so the reference lives in one place).

**How the setup was pinned down (sources, not memory)**
- Anserini's regression config (`beir-v1.0.0-scifact.flat.yaml`): BeirFlatCollection
  (title + "\n" + text), `-bm25 -removeQuery -hits 1000`, trec_eval `-c`, published values,
  and index stats (5,183 docs, 838,128 terms).
- `DefaultEnglishAnalyzer`: StandardTokenizer → EnglishPossessiveFilter → LowerCaseFilter →
  StopFilter(ENGLISH_STOP_WORDS_SET) → PorterStemFilter.
- Lucene `BM25Similarity` (idf, avgdl = sumTotalTermFreq / docCount, 256-entry norm cache,
  `w - w / (1 + tf * normInverse)`), `SmallFloat` (length encoding), `PorterStemmer`.
- Anserini `BagOfWordsQueryGenerator`: repeated query terms become a boost equal to the count.
- `RunOutputWriter`: scores written with `%f` (6 decimals; creates ties). trec_eval breaks
  ties by docno *descending*; confirmed with pytrec_eval on a toy run.

**Validation beyond SciFact**
- Scores match bm25s (`method="lucene"`, exact lengths) on 2,000 SciFact docs × 100 queries,
  rtol 1e-5, given the same tokens.
- Porter stems match Martin Porter's official output for all 23,531 test words, and NLTK's
  `MARTIN_EXTENSIONS` mode on the SciFact vocabulary.
- `bench/ir_eval.py` matches pytrec_eval per query on random runs with many ties.

**Problems**
- First SciFact index had 45 fewer terms than Anserini (838,083). Counting Emoji-property
  characters in the corpus found ® (27), © (11), ™ (4), ↔ (2) = 44: Lucene emits emoji as tokens,
  which the first tokenizer skipped. Added emoji tokens from Unicode's emoji-data.txt
  (`scripts/gen_emoji_table.py`, Unicode 18.0). One term (0.0001%) is still unaccounted for;
  finding it needs a Lucene run, and it doesn't move any metric.
- Two of my own test expectations were wrong (Porter keeps "runner"; two docs tied exactly).
  The code was right both times.

**Decisions**
- Deleted documents keep counting in N, df, avgdl (Lucene's behaviour before merges); an exact
  update would need a forward index. Documented on the class.
- BM25 returns `Neighbor` with the negated score in `distance`, so every index sorts the same
  way; Python gets positive scores and `-inf` padding (FAISS's convention for similarities).
- Unicode Word_Break classes come from general categories plus UAX #29's explicit punctuation
  lists, not the full Word_Break property table. The one-term gap on SciFact says this is close;
  other corpora (non-Latin scripts) may show larger differences.

## 2026-09-25 — Hybrid retrieval

**Done**
- C++ fusion (`include/strata/fusion.hpp`): reciprocal rank fusion (k = 60) and weighted
  fusion of min-max-normalized scores; bound as `strata.fuse_rrf` / `strata.fuse_weighted`
  (batched, GIL released).
- `strata.HybridIndex` (Python): one vector index + one `Bm25Index`, ids aligned by
  construction (validate, then add to both under one lock), external doc ids, removal from both.
- `scripts/embed_beir.py`: pinned models (Hugging Face commit hashes) with each model's
  required input formatting; L2-normalized; model, revision, formatting, pooling, device, and
  library versions recorded in `meta.json`.
- `bench/eval_hybrid_beir.py`: BM25, dense, RRF, weighted on the test split, fusion weight tuned
  on the train split; baselines checked against published references.

**Result (SciFact test, 300 queries; bge-small-en-v1.5 @ 5c38ec7c405e, CPU)**

| Method | nDCG@10 | R@100 | Reference |
|---|---|---|---|
| BM25 | 0.6789 | 0.9253 | Anserini 2.3.0 flat: 0.6789 / 0.9253 |
| Dense (bge-small) | 0.7127 | 0.9417 | MTEB (model card): 0.71275 / 0.94167 |
| RRF (k = 60, not tuned) | 0.7273 | 0.9683 | |
| Weighted (dense 0.65, tuned on train) | 0.7316 | 0.9667 | |

Both fusions beat either retriever alone; RRF gets the best recall, weighted the best nDCG@10.

**Protocol decisions (fixed before looking at test numbers)**
- **RRF k = 60** from the paper, not tuned.
- **Weighted fusion weight** tuned on SciFact's *train* split (809 queries; grid 0.00..1.00 in
  steps of 0.05, best nDCG@10, ties to the smaller weight) and applied unchanged to test. The
  script refuses to run if train and test query ids overlap. Train curve saved with the result.
- **Candidates:** top 100 from each retriever.
- **Documents:** BM25 indexes title + "\n" + text (Anserini flat); the embedding model sees
  title + " " + text (BEIR's dense convention, and what MTEB used for the reference).
- **Embeddings:** bge's retrieval instruction is prepended to queries only (model card); e5
  (available, not yet run) uses "query: " / "passage: ". Embeddings are L2-normalized and
  searched with negated inner product, which ranks exactly like cosine. The revision is pinned
  in `MODELS`, so the IdeaPad run loads identical weights; the device is recorded because CPU,
  CUDA, and MPS arithmetic differ slightly.
- **Identical query ids removed** from results for every method (Anserini `-removeQuery`,
  BEIR `ignore_identical_ids`).

**Notes**
- Embedding SciFact (5,183 docs + 1,109 queries) with bge-small took 243 s on the M2 CPU.
- The dense pipeline reproduces MTEB's published SciFact numbers to 4 decimals, which checks
  the prefix, normalization, pooling (CLS), and revision together.
