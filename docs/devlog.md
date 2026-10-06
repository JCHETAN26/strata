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

On means, both fusions beat either retriever alone (RRF best recall, weighted best nDCG@10).
**Correction after significance testing (next entry):** the gains over BM25 are significant,
the gains over dense retrieval are not once the six comparisons are Holm-corrected.

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

## 2026-09-25 — BEIR runner: weight rule and significance tests

**Pre-declared weight rule** (in `bench/eval_hybrid_beir.py`, `WEIGHT_RULE`): tune the
weighted-fusion weight on the dataset's dev split if it exists, else train, else a fixed 0.5.
The split used is recorded per dataset as `protocol.weight_source`. SciFact has no dev split,
so it uses train (weight 0.65, unchanged).

**Significance** (`bench/significance.py`): for every pair of methods, the mean per-query
difference with a 95% paired bootstrap CI (10,000 resamples) and a two-sided paired
randomization (sign-flip) test, Holm-adjusted across the 6 pairs. Per-query nDCG@10 and R@100
are saved with each result so the tests can be rerun. The test is calibrated: under no true
difference it rejects at p < 0.05 in under 10% of 200 simulated comparisons (unit test).

**SciFact (300 test queries), nDCG@10 differences**

| A − B | Mean diff | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| RRF − BM25 | +0.048 | [+0.026, +0.070] | 0.0001 | 0.0006 |
| weighted − BM25 | +0.053 | [+0.027, +0.079] | 0.0004 | 0.002 |
| dense − BM25 | +0.034 | [+0.002, +0.066] | 0.042 | 0.13 |
| weighted − dense | +0.019 | [+0.002, +0.036] | 0.029 | 0.12 |
| RRF − dense | +0.015 | [−0.008, +0.037] | 0.21 | 0.41 |
| weighted − RRF | +0.004 | [−0.011, +0.019] | 0.58 | 0.58 |

(Exact values in `results/tables.md`, generated from the saved result.)

**What this changes:** the earlier claim that both fusions "beat either retriever alone" held
on means only. Both fusions are reliably better than BM25; against dense retrieval, the
weighted fusion's uncorrected CI excludes zero but it does not survive Holm correction, and
RRF's does not exclude zero. RRF and weighted fusion are indistinguishable. 300 queries is a
small test set; more datasets (NFCorpus, FiQA on the IdeaPad) will say more than one.

## 2026-09-25 — Cited answer generation and HotpotQA evaluation

**Done**
- `rag/answer.py`: `claude-haiku-4-5` (named in CLAUDE.md), temperature 0, prompt version
  `cited-answer-v1`. Passages go in as *custom content* documents whose blocks are sentences,
  with citations enabled; citations come back as `content_block_location` (sentence ranges), which
  map to (title, sentence index). The reply ends with `Answer: <short answer>` or
  `Answer: unknown` (citations and structured outputs can't be combined, so the short answer is a
  marked line). Raw responses are cached by SHA-256 of the full request.
- `rag/hotpot_metrics.py`: the official `hotpot_evaluate_v1.py` metrics, ported and checked
  against the official script's own outputs (including its quirks).
- `scripts/prepare_hotpotqa.py`: distractor dev set from Hugging Face `hotpotqa/hotpot_qa`,
  pinned revision `1908d6afbbea…`, checksum recorded; seeded subset (n=100, seed 0: 81 bridge,
  19 comparison).
- `bench/eval_hotpotqa.py`: two conditions on the same questions: *distractor* (each
  question's 10 paragraphs, the standard setting) and *retrieved* (HybridIndex RRF top-5 from a
  pooled corpus of the subset's 991 paragraphs). Answer EM/F1, supporting-fact P/R/F1/EM of the
  cited sentences, joint metrics, answer-in-citations rate, citation coverage, abstention,
  cost, and paired significance tests between conditions.

**Groundedness is measured, not judged.** The model's citations are sentence indices, and
HotpotQA's gold supporting facts are sentence indices, so the official supporting-fact metrics
score whether answers cite the right evidence. No LLM judge is involved.

**Status: generation has not run yet.** There is no `ANTHROPIC_API_KEY` on this machine (no
env var, no `.env`). Everything up to the API call runs and is tested:
- Retrieval (measured): both gold paragraphs in the top 5 for 71% of questions, at least one for
  100%.
- The full evaluation path, driven by a scripted oracle generator in the tests, scores every
  official metric at exactly 1.0 in the distractor setting.
- Estimated cost of the full run: 200 calls at roughly 1.5–3k input tokens each, about $0.5–1.

**Decisions**
- HotpotQA from the original distractor release (via Hugging Face), not BEIR's HotpotQA: BEIR
  keeps only retrieval qrels, and answer evaluation needs the answers and supporting facts.
- Two conditions separate retrieval errors from generation errors.
- `answer-in-citations` is a simple string check (normalized short answer inside normalized cited
  text); it is reported next to the official metrics, not instead of them.

## 2026-09-26 — Readable parameterized test names; isolated Python test runs

**Problem:** on the Mac, ctest listed every value-parameterized test with a raw hex dump
(`Kernels/SimdKernel.MatchesScalarWithinTolerance/48-byte object <6E-65 6F-6E ...>`), and
so did `BruteForceMetric`, `PqAdc`, and `CrashRecovery`. CMake's `gtest_discover_tests` names
ctest tests from gtest's *printed parameter value*, not from the name generator, and gtest
prints types it has no printer for as bytes. For `KernelCase` and `CrashCase` those bytes include
function pointers and heap addresses, so the names could also change between builds.

**Fix:** a `PrintTo` for every parameter type (`Metric`, `KernelCase`, `CrashCase`), and one
explicit name generator, `test::PrintedName`, that returns the `PrintTo` string. The gtest name
and the printed value are the same string, so ctest names are readable and have the same format
on every platform. Verified after a clean rebuild: `ctest -N` shows `.../neon_cosine` on arm64
and `.../avx2_cosine` on the x86_64 build, with no hex names left in either.

**Python tests** now run through `make test-python`:
`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest`. Checked the effect: plain
`pytest` autoloaded the `anyio` plugin (pulled in by the Anthropic SDK) and put an inherited
`PYTHONPATH` entry on `sys.path`; under `make test-python` neither happens, and the suite still
passes (94 passed, 2 skipped).

## 2026-09-26 — HotpotQA in the BEIR setting (subset), with cost estimates

**Status of cited-answer generation:** built and tested (26 Sep), never run against the API:
there is still no `ANTHROPIC_API_KEY` on the Mac. Everything up to the API call runs.

**Done**
- `scripts/prepare_hotpotqa_beir.py`: streams the official BEIR `hotpotqa.zip` (654 MB,
  SHA-256 pinned; 5,233,329 passages) without extracting it and writes a subset: 100 test
  queries (seed 0) with their 2 gold passages, their 8 HotpotQA distractor passages (TF-IDF hard
  negatives, matched by title), and a hash-seeded uniform background sample (19,915 passages);
  20,906 passages total. Answers and supporting facts are joined from the pinned HotpotQA dev set.
- `bench/eval_hotpotqa_beir.py`: HybridIndex retrieval over the subset, then cited answers in
  two conditions: *retrieved* (top 5) and *gold* (the 2 qrels passages). It **estimates cost by
  default**; `--run` requires a cap and refuses if the worst case exceeds it.

**Checks on the join (all passed, over the 200 gold / 791 distractor passages used):**
- BEIR's HotpotQA test split is exactly HotpotQA's dev set (7,405 / 7,405 ids).
- Every query has 2 qrels passages, and their titles equal the supporting-fact titles.
- Every gold/distractor passage's BEIR text equals HotpotQA's joined sentences after
  whitespace/NFKC normalization (39 of 600 differ only in non-breaking spaces), so HotpotQA's
  official sentence splits are used and cited sentence indices map exactly to supporting facts.

**Retrieval on the subset** (RRF of BM25 + bge-small; *not comparable to full-corpus BEIR*):
nDCG@10 0.842, R@5 0.815, R@100 0.975, both gold passages in the top 5 for 65% of queries.

**Cost estimate before any run** (claude-haiku-4-5, $1 / $5 per MTok; offline token estimate
at 3.5 chars/token + 25% for citation overhead, replaced by the free count_tokens endpoint when
a key is present):

| Condition | Requests | Input tokens | Expected | Worst case (max_tokens) |
|---|---|---|---|---|
| retrieved | 100 | 111,258 | $0.19 | $0.62 |
| gold | 100 | 52,079 | $0.13 | $0.56 |
| total | 200 | | $0.31 | $1.19 |

The default cap ($1.00) is below the worst case on purpose: a run needs an explicit cap.

**Problems**
- The first end-to-end test expected the oracle's cited-sentence precision to be 1.0 in the
  retrieved condition; it was 0.99 because one question had no gold passage retrieved, and the
  official metric scores an empty prediction as precision 0. The test was wrong; it now checks
  that every citation is a gold fact.

## 2026-09-26 — First live run: HotpotQA (BEIR subset) with cited answers

Run: `results/rag/hotpotqa-subset-n100-seed0-bg20000-20260926-121603-008304.json`
(claude-haiku-4-5, temperature 0, max_tokens 512, prompt `cited-answer-v1`; 100 BEIR test queries
over a 20,906-passage subset corpus).

**Tokens and cost**

| Condition | Offline estimate (old) | Exact (count_tokens) | Billed input | Output | Actual cost |
|---|---|---|---|---|---|
| retrieved (top 5) | 111,258 | 173,263 | 173,263 | 9,643 | $0.2215 |
| gold (2 passages) | 52,079 | 102,350 | 102,350 | 9,622 | $0.1505 |
| total | 163,337 | 275,613 | 275,613 | 19,265 | **$0.372** |

Pre-run estimate with exact counts: expected $0.426, worst case $0.788 (cap $1.25). All 200
responses ended with `end_turn`: none hit max_tokens (longest answer 191 tokens, mean 96), so
nothing was flagged and every answer is scored.

**Results**

| Condition | EM | F1 | Cited-sentence precision | Cited-sentence recall | SP F1 | Joint F1 | Abstained |
|---|---|---|---|---|---|---|---|
| retrieved | 0.380 | 0.517 | 0.852 | 0.644 | 0.703 | 0.411 | 28% |
| gold | 0.550 | 0.739 | 0.941 | 0.803 | 0.844 | 0.634 | 3% |

Gap (retrieved − gold, 100 questions, 95% paired bootstrap CI, randomization p):
answer F1 −0.221 [−0.298, −0.148], p = 0.0001 (retrieved better on 0, worse on 27, tied on 73);
SP F1 −0.140 [−0.190, −0.093], p = 0.0001; joint F1 −0.223 [−0.297, −0.151], p = 0.0001.

**Where the gap comes from:** retrieval. On the 65 questions where both gold passages were in the
top 5, answer F1 is 0.736 retrieved vs 0.754 gold. On the other 35 it is 0.111 vs 0.710, and the
model abstained ("Answer: unknown") on 71% of them rather than guessing. Every answered question
carried citations (coverage 1.00).

**Bug found by the first live call:** anthropic SDK 1.x removed `temperature` from
`messages.create()`'s signature (TypeError), though the API still accepts it for Haiku 4.5. It now
goes in `extra_body` (the SDK upgrade guide's documented path for models that honour it). The
fake client in the offline tests accepted any keyword, so only the live test caught it.

**Estimator calibration** (`bench/calibrate_token_estimate.py`): least squares on the 200 exact
counts gives tokens ≈ 481 + 0.392 × characters (about 2.5 characters per token plus a fixed
~481-token citation overhead per request). Mean absolute error: old estimator 43% (41% low in
total), new 4.0% in-sample (max 14%); cross-condition (fit on one condition, predict the other)
10–19%. Adding a per-sentence-block term only reached 3.2% in-sample and 8–17% cross-condition,
so the two-term model was kept. Exact counts equalled billed usage for all 200 requests.

## 2026-09-26 — Follow-ups: SDK-checked fake client, answered-only metrics, retrieval failures

**Fake Anthropic client** (`tests/python/fake_anthropic.py`): every `messages.create` /
`count_tokens` call is bound against the installed SDK's real signature
(`inspect.signature(...).bind`), so arguments the SDK rejects raise the same TypeError offline.
Checked by reintroducing `temperature=` as a keyword: three offline tests failed, including a
TypeError from the signature check.

**Answered-only quality** (abstentions excluded; re-scored from cached responses, $0):

| Condition | Abstained | Answered | EM (answered) | F1 (answered) |
|---|---|---|---|---|
| retrieved | 28% | 72 | 0.528 | 0.718 |
| gold | 3% | 97 | 0.567 | 0.761 |

**Retrieval failures** (`bench/analyze_hotpotqa_retrieval.py`): all 35 questions that miss a gold
passage in the top 5 are *bridge* questions (all 19 comparison questions have both). 36 gold
passages are missing; none is named in its question (typical second-hop passages), and for 34 of
them the question's other gold passage is already in the top 5.

Rank of each missing passage in the fused (RRF) top 100:

| Rank | 6–10 | 11–20 | 21–50 | 51–100 | not in top 100 |
|---|---|---|---|---|---|
| Missing passages (all bridge) | 19 | 6 | 3 | 3 | 5 |

Ceiling for a reranker (questions with both gold passages inside the candidate pool):

| Pool | N=5 | N=10 | N=20 | N=50 | N=100 |
|---|---|---|---|---|---|
| fused top N | 65 | 83 | 89 | 92 | 95 |
| union of BM25 top N and dense top N (mean size) | 76 (7) | 83 (15) | 90 (32) | 95 (87) | 97 (177) |

RRF can bury a passage that only one retriever finds (e.g. dense rank 6, fused rank 41), so a
reranking pool should be the union of the retrievers' lists, not the fused list. Three questions
stay out of reach even with the top-100 union: those need multi-hop retrieval.

## 2026-09-26 — Cross-encoder reranking (HotpotQA BEIR subset)

Run: `results/rerank/hotpotqa-subset-n100-seed0-bg20000-20260926-131257-741129.json`, code at
`42ac70c`. The record is flagged dirty only because `bench/eval_hotpotqa_beir.py` was being edited
during the run; `eval_rerank.py` does not import it, and every module it does use is unchanged
since `42ac70c`. The quality numbers are identical to an earlier dry run (deterministic).

**Pool depth, chosen on the dev-split subset** (100 BEIR dev queries, built like the test subset),
R@5 by union-pool depth N:

| Model | N=10 | N=20 | N=50 | Chosen |
|---|---|---|---|---|
| bge-reranker-base | 0.930 | **0.950** | 0.945 | 20 |
| ms-marco-MiniLM-L6-v2 | **0.835** | 0.825 | 0.825 | 10 |

**Test subset (100 queries)**

| Method | Passages | R@5 | Both gold in set | nDCG@10 | Latency / query (CPU) |
|---|---|---|---|---|---|
| fused top 5 (current) | 5 | 0.820 | 0.65 | 0.837 | — |
| union top 5, no model | 7.3 | 0.870 | 0.76 | — | — |
| MiniLM-L6, N=10 (pool 15) | 5 | 0.840 | 0.69 | 0.847 | 231 ms (p95 464) |
| bge-reranker-base, N=20 (pool 32) | 5 | **0.915** | **0.83** | **0.907** | 3,029 ms (p95 4,887) |
| ceiling (bge pool) | 32 | | 0.90 | | |

Paired differences in both-gold (Holm over 6 pairs): bge vs fused +0.18 [+0.10, +0.26],
p_holm 0.0006; bge vs union-no-model +0.07 [0.00, +0.14], p_holm 0.19 (not significant);
union-no-model vs fused +0.11 [+0.05, +0.18], p_holm 0.015; MiniLM vs fused +0.04, p_holm 0.48;
bge vs MiniLM +0.14 [+0.06, +0.22], p_holm 0.0045.

**Reading:** bge-reranker-base recovers 18 of the 25 questions its pool makes reachable (0.65 →
0.83 of a 0.90 ceiling) with 5 passages, and is the only reranker that beats the fused baseline.
Against the no-model union baseline (7.3 passages, zero model cost) its gain is not significant on
100 queries. MiniLM-L6 is not distinguishable from doing nothing. Latency on the M2 CPU (4 torch
threads) is 13x higher for bge (3.0 s vs 0.23 s per query); this belongs on the GPU.

**Deferred:** SciFact reranking (planned second dataset). Tuning on 300 train queries with pools
up to N=50 of ~300-token passages would take bge-reranker-base an estimated 2+ hours of sustained
CPU on the fanless Mac; it runs on the IdeaPad's GPU instead (checklist).

## 2026-09-26 — End-to-end: reranked and union passages to the generator

Run: `results/rag/hotpotqa-subset-n100-seed0-bg20000-20260926-132037-509138.json` (clean tree,
code at `b742c7b`; passages from the rerank run
`hotpotqa-subset-n100-seed0-bg20000-20260926-131257-741129.json`). New spend $0.497
(reranked_bge $0.221, union_top5 $0.277; retrieved and gold from cache). Estimate beforehand:
expected $0.550, worst case $0.910, cap $1.00. All 400 responses ended with `end_turn`.

| Condition | Passages | EM | F1 | Answered EM | Answered F1 | Abstained | Cite P | Cite R | SP F1 | Joint F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieved (fused top 5) | 5 | 0.380 | 0.517 | 0.528 | 0.718 | 28% | 0.852 | 0.644 | 0.703 | 0.411 |
| union top 5, no model | 7.3 | 0.420 | 0.572 | 0.553 | 0.752 | 24% | 0.848 | 0.675 | 0.727 | 0.465 |
| reranked (bge, N=20) | 5 | 0.460 | 0.614 | 0.561 | 0.749 | 18% | 0.887 | 0.692 | 0.750 | 0.495 |
| gold passages | 2 | 0.550 | 0.739 | 0.567 | 0.761 | 3% | 0.941 | 0.803 | 0.844 | 0.634 |

Planned comparisons (one Holm family of 3 per metric; 95% paired bootstrap CI):

| Metric | bge − retrieved | union − retrieved | bge − union |
|---|---|---|---|
| EM | +0.080 [+0.02, +0.15], p_holm 0.12 | +0.040, p_holm 0.56 | +0.040, p_holm 0.56 |
| F1 | **+0.097 [+0.037, +0.161], p_holm 0.007** | +0.055 [+0.006, +0.106], p_holm 0.074 | +0.042 [−0.021, +0.107], p_holm 0.20 |
| SP F1 | **+0.046 [+0.016, +0.081], p_holm 0.011** | +0.024, p_holm 0.44 | +0.023, p_holm 0.44 |
| Joint F1 | **+0.084 [+0.030, +0.142], p_holm 0.009** | **+0.054 [+0.011, +0.102], p_holm 0.039** | +0.030, p_holm 0.30 |

**Reading:** reranking with bge significantly improves answer F1, citation quality, and joint F1
over the current pipeline. The zero-cost union baseline improves joint F1 significantly but not
answer F1 after correction, and costs 33% more input tokens (7.3 passages). bge and the union
are not distinguishable on 100 questions. Answered-only F1 is nearly the same everywhere
(0.718–0.761): the gains come from answering more questions (abstention 28% → 18%) because the
right passages are present, not from better answers when the model does answer.

**Display bug:** the run's printout labelled every condition "no new spend", including the two
that were generated fresh; the label was unconditional. The saved record's cache counts were
right. Fixed after the run.

## 2026-09-26: Two-hop retrieval for bridge questions

`rag/multihop.py`, `bench/eval_multihop.py`. Hop 1 is the fused ranking. The top m hop-1 passages
are each expanded into a hop-2 query (question + passage title / title + first sentence / full
text), which goes to BM25 and dense retrieval. The final top 5 is the first `keep` hop-1
passages, then the hop-2 candidates, ordered round-robin (no model) or by bge-reranker-base
reading the hop-2 query. All parameters are tuned on the dev subset only. Stage 1 (no model,
108 configurations) chose m=1, full text, depth 5, keep 3 (dev R@5 0.885). Stage 2 (bge;
m/expansion fixed from stage 1 because bge costs ~90 ms per pair here) chose depth 10, keep 2
(dev R@5 0.940). The code was committed (`434a0d5`) and run on a clean tree. There is no API
spend. Result: `results/multihop/hotpotqa-subset-n100-seed0-bg20000-20260926-133541-408924.json`.

Test subset (100 queries: 81 bridge, 19 comparison):

| Method | R@5 | R@5 bridge | R@5 comparison | All-relevant@5 | Passages | ms/query (p50 / p95) | Pool ceiling |
|---|---|---|---|---|---|---|---|
| fused top 5 | 0.820 | 0.778 | 1.000 | 0.65 | 5 | — | — |
| union top 5, no model | 0.870 | 0.840 | 1.000 | 0.76 | 7.3 | — | — |
| bge rerank (single hop, N=20) | **0.915** | 0.895 | 1.000 | **0.83** | 5 | 3029 (2863 / 4887)* | 0.90 |
| two-hop, no model | 0.885 | 0.858 | 1.000 | 0.78 | 5 | **35 (28 / 60)** | 0.90 |
| two-hop, bge | 0.905 | 0.889 | 0.974 | 0.82 | 5 | 2222 (2080 / 3704) | **0.98** |

\*Reranker scoring time only, from the rerank run. For two-hop, latency is end to end per query
(the original question's embedding is precomputed and excluded). No-model two-hop is dominated
by embedding the hop-2 query (32 ms); bge two-hop by scoring (2138 ms, ~15 pairs).

Planned comparisons (Holm over 4 per metric): two-hop bge − fused R@5 +0.085 [+0.030, +0.145],
p_holm 0.040; all-relevant +0.170, p_holm 0.025. Two-hop no-model − fused R@5 +0.065 [+0.020,
+0.115], p_holm 0.045; all-relevant +0.130, p_holm 0.045. Two-hop bge − single-hop bge −0.010,
p_holm 1.0. Two-hop no-model − union +0.015, p_holm 1.0.

**Comparison-question check: two-hop bge FAILS (pre-declared: mean R@5 difference from fused
≥ 0).** One comparison question (5a75eb12…, "Which Muslim scholar was born first…") lost its
second gold passage: it was fused rank 3, and keep=2 let two hop-2 candidates push it out. Every
comparison question's gold passages are already in the fused top 5, so any keep < 5 can only hurt
them. The no-model variant (keep=3) passes with no change on any comparison question.

**Reading:** two-hop retrieval works for bridge questions and gets the pool ceiling to 0.98, but
on this subset it does not beat single-hop bge reranking at top 5. The practical result is the
no-model variant. It is significantly better than fused, as good as the no-model union
(+0.015, n.s.) with 5 passages instead of 7.3, costs ~35 ms per query instead of ~3 s, and does
not hurt comparison questions. Tuning 108 configurations on 100 dev queries risks overfitting.
The dev-to-test drops (0.885 → 0.885 and 0.940 → 0.905) are consistent with that for the bge
variant. Obvious next steps, not done: rerank the union of the single-hop and hop-2 pools
together (the 0.98 ceiling is there to be had), or skip hop 2 for questions classified as
comparisons.

## 2026-09-26: Joint bge reranking of single-hop and hop-2 candidates

`bench/eval_multihop_joint.py`, `rag/multihop.py` `joint_rank`. The keep rule is gone. The
single-hop pool (BM25 top N ∪ dense top N, scored against the question) and the hop-2 pools
(fused top m, full-text expansion, BM25/dense top `depth`, each scored against its hop-2 query)
are merged. Each candidate gets its best bge score. The grid was kept to 8 settings to limit
overfitting (m ∈ {1,2} × depth ∈ {5,10} × N ∈ {10,20}), dev only. Chosen: m=2, depth 5, N=10
(dev R@5 0.945; every setting was 0.92–0.945). Code `d86cb53`, clean tree. No API spend. Result:
`results/multihop/hotpotqa-subset-n100-seed0-bg20000-joint-20260926-140113-567318.json`.

| Method | R@5 | bridge | comparison | All-relevant@5 |
|---|---|---|---|---|
| fused top 5 | 0.820 | 0.778 | 1.000 | 0.65 |
| bge rerank (single hop) | 0.915 | 0.895 | 1.000 | 0.83 |
| two-hop, no model | 0.885 | 0.858 | 1.000 | 0.78 |
| two-hop, bge (keep 2) | 0.905 | 0.889 | 0.974 | 0.82 |
| **joint bge** | **0.935** | **0.926** | 0.974 | **0.87** |

Joint: pool 18.4 passages, pool ceiling 0.95, 3333 ms/query end to end (p50 3174, p95 5299;
nearly all bge scoring). Planned comparisons (Holm over 3): vs fused R@5 +0.115 [+0.070, +0.160],
p_holm 0.0003 (all-relevant +0.22, p_holm 0.0003); vs single-hop bge +0.020 [−0.020, +0.060],
p_holm 0.46; vs two-hop no-model +0.050 [+0.010, +0.090], p_holm 0.064.

**The comparison-question check still FAILS**, on the same question as before (5a75eb12…, "Which
Muslim scholar was born first, Kamāl al-Dīn al-Fārisī or M. A. Muqtedar Khan?"). I expected
joint scoring to fix it, and it didn't. Diagnosis from bge's scores on that question: the hop-2 query built
from the al-Fārisī passage makes bge rate *other* medieval-Muslim-scholar passages (Al-Raghib
al-Isfahani 0.99996, …) above the second gold passage (M. A. Muqtedar Khan, best score 0.99958).
The expanded query contains the passage's own text, so passages *similar to the hop-1 passage*
score as relevant, and a max over pools lets that bias through. This is a structural weakness of
max-over-queries scoring, not a keep-rule artifact. I did not tune further for it (one question,
and further tuning on 100 dev questions would be overfitting).

Side note: `Reranker` returns bge's sigmoid-activated scores (the sentence-transformers default for
a one-logit model). Ranking is unaffected (sigmoid is monotone, and so is max after it), but
confident scores bunch near 1.0 (here 0.9989–0.99997), so float32 resolution could create ties.
Returning raw logits would be safer; left as is so results stay comparable.

## 2026-09-26: End-to-end with two-hop and joint-reranked passages (six conditions)

`uv run python bench/eval_hotpotqa_beir.py --run --max-cost-usd 1.00 --rerank-result <rerank>
<multihop> <joint>`, code `37ba873`, clean tree. Result:
`results/rag/hotpotqa-subset-n100-seed0-bg20000-20260926-141835-620789.json`. Estimate beforehand:
expected $0.462, worst case $0.801 (count_tokens), cap $1.00. New spend **$0.412** (twohop
$0.199 for 91 uncached, joint_bge $0.213 for 96 uncached; the other four conditions came entirely
from the cache). All 600 responses ended with `end_turn`; none flagged.

| Condition | Passages | EM | F1 | Answered EM | Answered F1 | Abstained | Cite P | Cite R | SP F1 | Joint F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieved (fused top 5) | 5 | 0.380 | 0.517 | 0.528 | 0.718 | 28% | 0.852 | 0.644 | 0.703 | 0.411 |
| union top 5, no model | 7.3 | 0.420 | 0.572 | 0.553 | 0.752 | 24% | 0.848 | 0.675 | 0.727 | 0.465 |
| reranked (bge, single hop) | 5 | 0.460 | 0.614 | 0.561 | 0.749 | 18% | 0.887 | 0.692 | 0.750 | 0.495 |
| two-hop, no model | 5 | 0.450 | 0.605 | 0.570 | 0.766 | 21% | 0.845 | 0.674 | 0.722 | 0.498 |
| **joint bge (two-hop)** | 5 | **0.480** | **0.669** | 0.545 | 0.761 | **12%** | 0.877 | **0.739** | **0.778** | **0.564** |
| gold passages | 2 | 0.550 | 0.739 | 0.567 | 0.761 | 3% | 0.941 | 0.803 | 0.844 | 0.634 |

Planned comparisons: one Holm family of 6 per metric, pre-declared in `PLANNED` before the run.
Bold means p_holm < 0.05.

| Metric | bge − retr. | union − retr. | bge − union | two-hop − retr. | joint − retr. | joint − bge |
|---|---|---|---|---|---|---|
| EM | +0.080, 0.20 | +0.040, 0.85 | +0.040, 0.85 | +0.070, 0.47 | **+0.100, 0.008** | +0.020, 0.85 |
| F1 | **+0.097, 0.012** | +0.055, 0.11 | +0.042, 0.20 | +0.088, 0.096 | **+0.152 [+0.094, +0.216], 0.0006** | +0.055 [−0.000, +0.114], 0.11 |
| SP F1 | **+0.046, 0.019** | +0.024, 0.65 | +0.023, 0.65 | +0.019, 0.65 | **+0.074, 0.005** | +0.028, 0.57 |
| Joint F1 | **+0.084, 0.016** | +0.054, 0.054 | +0.030, 0.30 | +0.087, 0.054 | **+0.153, 0.0006** | +0.069 [+0.014, +0.127], 0.054 |

(Entries are mean difference and p_holm. The larger family makes earlier pairs' p_holm larger
than in the 3-pair run; for example, union − retrieved joint F1 goes from 0.039 to 0.054.)

By question type (81 bridge / 19 comparison), answer F1: retrieved 0.481 / 0.674, bge 0.600 /
0.674, two-hop 0.575 / 0.735, joint 0.663 / 0.695, gold 0.754 / 0.674. End to end, comparison
questions do **not** regress under joint reranking. The one question whose second gold passage it
drops is answered wrong by *every* condition, including gold (F1 0). Under joint it abstains and
cites only one supporting fact (SP F1 1.0 → 0.67). The retrieval-level check still fails, and
that stands as recorded.

**Reading:** joint two-hop reranking is the best condition on every headline metric. It closes
about 68% of the retrieved-to-gold F1 gap (0.517 → 0.669 of 0.739) and nearly halves abstention
(28% → 12%) for the same input tokens as fused top 5. Against single-hop bge its gains are
consistent but not significant after correction on 100 questions (F1 +0.055, p_holm 0.11; joint F1
+0.069, p_holm 0.054). As before, answered-only F1 is flat (0.72–0.77). Better retrieval
works by letting the model answer more questions, not by improving the answers it would already give.

### Future work (recorded, not done)

1. **Score against the original question as well as the expanded query.** The joint reranker's
   comparison failure comes from the expanded query rewarding passages similar to the hop-1
   passage. Candidates could be scored against both the question and the hop-2 query and
   combined (e.g. the mean, or the question score with a hop-2 bonus), so a passage must also be
   relevant to the question itself. Costs one extra bge pass over the hop-2 candidates
   (~+10 pairs, ~+1 s per query on the M2 CPU).
2. **Use raw reranker logits instead of sigmoid scores.** `Reranker` returns sentence-
   transformers' default sigmoid for one-logit models; confident scores bunch at 0.9989–0.99997,
   where float32 resolution (~6e-8) can create ties that fall back to pool order. Passing
   `activation_fn=torch.nn.Identity()` would keep the same ranking with more headroom. It would
   also make score combinations such as idea 1 better behaved: logits add, probabilities near 1 do
   not.

RAG experiments on the Mac are closed. The write-up is in `docs/rag-results.md`.

## Kaggle GPU runner (IdeaPad disk full)

The IdeaPad's root filesystem is at 99% (2.6 GiB free), and the only reclaimable space belongs to
other projects in the owner's home directory. That is not enough for CUDA PyTorch (~6–8 GiB
installed) let alone the full-corpus HotpotQA embeddings (~7.5 GiB). So Stage 1 (embed group +
`uv sync`) was **cancelled** and the GPU stages moved to a Kaggle notebook. `uv.lock` already
pins `torch==2.14.0` to its CUDA (cu13) build on Linux, so no dependency change was needed.

New runner in `kaggle/` (`setup.sh`, `run_stages.py`, `notebook.ipynb`, `README.md`): clones the
repo with a token from Kaggle Secrets (never printed), builds the bindings (vcpkg pinned baseline
+ `uv sync`), installs the embed/rag deps, and runs full-corpus BEIR HotpotQA retrieval (Stage 2),
SciFact reranking (Stage 3), and GPU reranking latency (Stage 4). Only result JSONs are downloaded;
embeddings and indexes stay on Kaggle. Answer generation (Stage 2b) stays behind a cost estimate
and an explicit `--enable-api` flag.

Supporting changes:

- **`bench/benchmeta.py`:** `accelerator_info()` records GPU (via `nvidia-smi`), torch's CUDA view
  (only if torch imports), and Kaggle env vars (never the data-proxy token). Added to `metadata()`,
  so every result now records the exact GPU and environment.
- **`scripts/embed_beir.py`:** sharded, **resumable** embedding. Each shard is appended to
  `corpus.fbin` with a progress sidecar; a session killed at Kaggle's time limit resumes from the
  last completed shard, and a finished file is skipped. Verified locally (stubbed encoder): a run
  killed after 2 of 4 shards resumes to a byte-identical `corpus.fbin`. Peak encode RAM is bounded
  by one shard (default 200k × 384 × 4 ≈ 0.3 GiB) instead of the whole 7.5 GiB.
- **`scripts/prepare_hotpotqa_beir.py`:** `--full` builds the entire 5,233,329-passage BEIR
  HotpotQA corpus (test + dev) as `data/beir/hotpotqa/`, streamed, with `source_sha256`.
- **`bench/eval_rerank.py`:** `--device` / `--batch-size` so the cross-encoder runs on the GPU;
  device and batch size are recorded in the latency block for direct CPU-vs-GPU comparison.
- **Published references (`results/bm25/`, `results/hybrid/`):** added BEIR HotpotQA references so
  Stage 2 checks against them — BM25 nDCG@10 0.6330 / R@100 0.7957 (Anserini
  `beir-v1.0.0-hotpotqa.flat`, 2.3.0) and bge-small-en-v1.5 nDCG@10 0.69935 / R@100 0.84862 (model
  card MTEB at the pinned revision). A gap vs MTEB is expected (CPU-vs-GPU float, tokenizer
  truncation), so the runner reports the comparison rather than treating a miss as a hard failure.
- **Stage 2 memory guard:** the runner estimates corpus-embedding disk (~7.5 GiB) and exact-dense
  retrieval RAM (~15 GiB) up front and refuses to embed if it will not fit, proposing a higher-RAM
  accelerator, `--embed-only`, or `--allow-large-memory`.

Local tests: full Python suite passes (105 passed, 15 skipped — the torch-only ones). The full
GPU run happens on Kaggle.

### Full-corpus retrieval eval: weighted-tuning grid and silent output (fixed)

First Kaggle run: embedding finished cleanly (5,233,329 passages in 10,832 s on the RTX 3050 via
the sharded path), then `eval_hybrid_beir.py` printed nothing for a long time. Two causes, both in
that script's original small-dataset design:

1. Every `print` was at the *end* of `run_dataset`, so a run over 5.2M passages showed no output
   for its entire (long) duration — indistinguishable from a hang.
2. The weighted-fusion weight tuning re-runs the full search once per grid point (21 points) over
   the tuning split. On SciFact (5,183 docs) that is nothing; over 5.2M passages and ~5,447 dev
   queries it is ~21 full-corpus brute-force passes before any result prints — realistically
   longer than a Kaggle session, and it produces nothing the published-reference comparison uses
   (references are BM25 and dense only; RRF is fixed k=60).

Fix: `eval_hybrid_beir.py` gained `--methods` (evaluate a subset of bm25/dense/rrf/weighted); the
weighted grid runs only when `weighted` is selected; and results now print incrementally (each
tuning point and each method as it completes, flushed). The Kaggle `stage2` defaults to
`--methods bm25 dense rrf`, skipping the tuning bomb; `--methods ... weighted` opts back in.
Because `corpus.fbin` persists, killing the eval and rerunning reuses the embeddings.

## 2026-09-27: Kaggle GPU results: full-corpus HotpotQA, SciFact reranking, GPU latency

Ran `kaggle/run_stages.py` (env, stage2, stage3, stage4) at code `43ca3ce` on a Kaggle T4 x2
session (one GPU used; 4-vCPU Xeon host, 31 GiB RAM; `results/kaggle/environment.json`). No API
spend (stage2b not run). Of the 41 downloaded files, 36 were byte-identical copies of committed
results. Only the 5 new ones were copied in, none overwriting:
`results/hybrid/hotpotqa-bge-small-en-v1.5-20260927-094841-995184.json`,
`results/rerank/scifact-20260927-101744-604023.json`,
`results/rerank/hotpotqa-subset-n100-seed0-bg20000-20260927-105611-634422.json`,
`results/rerank/rerank_latency_cpu_vs_gpu.json`, `results/kaggle/environment.json`.

- **Full-corpus HotpotQA** (7,405 queries, 5,233,329 passages): BM25 nDCG@10 0.6329 / R@100
  0.7957 (Anserini 0.633 / 0.7957); dense 0.6993 / 0.8487 (published 0.69935 / 0.84862);
  RRF 0.7297 / 0.8722. `baselines_match_published: true`. RRF − dense +0.030 [+0.026, +0.035],
  p_holm 0.0003; larger than on SciFact (+0.015).
- **SciFact reranking** (N tuned on train: 10 for both): fused nDCG@10 0.7269, bge 0.7266
  (p_holm 0.98), MiniLM 0.6991 (p_holm 0.076). Reranking doesn't help. The likely causes are
  domain mismatch (web-trained cross-encoders, scientific claims) and little headroom (fused
  all-relevant@5 0.77 vs pool ceiling 0.87). These are hypotheses and have not been tested.
- **GPU latency** on the HotpotQA subset: bge 3029 → 688 ms, MiniLM 231 → 47 ms per query (M2 CPU
  vs T4). These are different machines, and the GPU is underused: one partial batch of 15–32
  pairs per query. The GPU rerank produced top-5 lists identical to the CPU run for all
  100 queries and both models.
- Small unexplained difference: SciFact fused nDCG@10 0.7269 in the rerank run vs 0.7273 in the
  committed hybrid run.

Placement of `environment.json`: `kaggle/README.md` only said "commit into `results/`". I used
`results/kaggle/environment.json` so it's labeled as the Kaggle environment, and documented
that in the README.

## 2026-09-27: HNSW stage (a): levels, layer search, insertion with simple selection

The HNSW core is now AI-implemented at my request (ground rule in CLAUDE.md updated), built in
reviewed stages, with `docs/explainers/hnsw.md` as the line-by-line explainer.

**Done**
- `src/index/hnsw.cpp`: level assignment (mL = 1/ln M), SEARCH-LAYER (Algorithm 2), greedy
  upper-layer descent + ef-bounded layer-0 search (Algorithm 5), INSERT (Algorithm 1) with
  back-links and overflow re-selection, using SELECT-NEIGHBORS-SIMPLE (Algorithm 3) for now.
- Layout: contiguous vectors; layer-0 links at a fixed stride of 2M+1 ids per node (count first);
  upper layers in small per-node blocks.
- New tests: `ConcurrentSearchesMatchSerial` (4 threads, for TSan) and
  `AddingItsOwnVectorsIsSafe` (for ASan).
- All 185 C++ tests pass under debug, asan, and tsan; Python tests pass, including the HNSW
  binding smoke test.

**Decisions**
- **Level RNG built from raw mt19937_64 bits**, not `std::uniform_real_distribution`: the engine
  sequence is standardized, the distributions are not, so libc++ and libstdc++ would build
  different graphs from the same seed.
- **Ties broken by (distance, id)** everywhere: deterministic traversals, and duplicates stay
  well-defined.
- **Entry points for the next layer during insert = all of W** (paper), not only the closest
  (hnswlib).
- **Visited set is a thread_local epoch array**: O(1) reset, and concurrent const searches share
  no scratch state. Inserts are single-writer.
- **Aliasing guard:** `add(index.vector(i))` would read from storage that the append may
  reallocate; such inputs are copied first.

**Observed (quick check, one run, not a result)**
- SIFT10K, M=16, efC=200, simple selection: recall@10 0.913 / 0.957 / 0.996 at ef 10 / 20 / 40;
  build 0.6 s. The scripted before/after comparison against the heuristic comes in stage (b).

**Went wrong**
- The Python module didn't pick up HNSW until reinstalled, and `uv sync --reinstall-package`
  needs `VCPKG_ROOT` (unset in the non-interactive shell; vcpkg is at `~/vcpkg`). Plain
  `uv sync` also **removes the `embed` group** (sentence-transformers, torch), since it isn't a
  default group. Restored with `uv sync --group embed`. Rebuild the module with
  `uv sync --group embed --reinstall-package strata`.
- `test_live_cited_answer` fails with an API usage-limit error (resets 2026-10-01). External,
  and unrelated to HNSW.

## 2026-09-27: HNSW stage (b): the neighbor-selection heuristic

**Done**
- `NeighborSelection { kSimple, kHeuristic }` in `HnswParams` (default: heuristic).
  `select_neighbors` implements Algorithm 4 (extendCandidates and keepPrunedConnections off) for
  both new links and overflow re-selection.
- Tests: `HeuristicPrefersDiverseNeighbors` (a hand-worked 5-point example where the two modes
  must choose different neighbors), and `BothModes/HnswSelection` (recall and layer-0
  reachability for each mode).
- Harness: `--selection heuristic|simple`, recorded in `build_params`; a new `graph` block (max
  level, nodes per layer, mean layer-0 degree, layer-0 reachable fraction).
- `scripts/make_clustered.py` (synthetic Gaussian clusters, exact float64 ground truth);
  `bench/run_selection_comparison.py` writes `results/selection/selection_comparison.md` and
  `results/plots/hnsw_selection.png`.

**Result** (M=16, efC=200; recall deterministic, QPS indicative on the M2)
- Clustered (100 x 1000, d16): closest-M levels off at recall 0.853 even at ef 320; 6.9% of
  nodes are unreachable from the entry point on layer 0. The heuristic reaches 1.000 at ef 80
  and 100% reachability with mean degree 16.5 vs 24.2.
- SIFT10K: +0.022 / +0.024 at ef 10 / 20, then parity; slightly behind at ef 80 (0.998 vs
  1.000). Both graphs are fully reachable.

**Decisions**
- **Ties keep the candidate** (reject only if d(e, r) < d(e, b)), as hnswlib does. Otherwise
  exact duplicates collapse to one link each.
- **Heuristic applied even with <= m candidates** (paper); hnswlib skips it there.
- The trial runs used to pick the clustered configuration (d16/c100, d8/c1000, d4/c100, all showing
  the effect) went to the scratchpad, not `results/`. The committed comparison uses the first.

## 2026-09-27: HNSW stage (c): Python bindings and curves against hnswlib and FAISS

**Done**
- Bindings: `HnswIndex(..., selection="heuristic"|"simple")`, plus read-only `M`,
  `ef_construction`, and `selection`. The stub test was replaced by real tests: recall vs brute
  force (3 metrics x 2 modes), exact distances, padding, errors, introspection, the worked
  selection example, threads, and concurrent adds. `strata_reference` now writes HNSW results,
  and the bindings match C++ exactly on SIFT10K (both modes, ef 16 and 64).
- `bench/run_hnsw_curves.py`: Strata vs hnswlib vs FAISS HNSW plus a brute-force baseline, Mac
  protocol (3 runs, one configuration, 60 s cool-down between builds). Charts and table are
  titled "Mac development results, not final"; QPS is labeled indicative.

**Result** (M2, M=16, efC=200, single thread; recall final, QPS/build indicative)
- SIFT1M recall@10 matches hnswlib at every ef to within 0.001 (e.g. 0.9939 at ef 160 for both).
  FAISS is +0.005 at ef 10, converging by ef 160.
- Speed: Strata runs between the two references, ~1.6-1.8x hnswlib (no NEON path in hnswlib) and
  ~0.7x FAISS at ef 10, closing to ~0.97x at ef 320. HNSW at recall 0.994 is ~39x SIMD brute
  force on SIFT1M.
- SIFT1M build, single thread: Strata 316 s, FAISS 406 s, hnswlib 578 s.

**Went wrong**
- **OpenMP double-load aborted the FAISS runs (OMP Error #15)**, twice, for different reasons:
  1. The curves script imported hnswlib's and FAISS's drivers into one process. Fixed by running
     every step as its own subprocess, which also frees each SIFT1M index before the next build.
  2. Even alone, the FAISS process aborted when saving its record: `benchmeta.accelerator_info()`
     imported torch (installed by the embed group), which loads its own libomp. That code came
     with the Kaggle commit, after the last FAISS run, so it hadn't shown up before. Fixed by
     probing torch in a subprocess; the same fields are recorded. I did not use the
     `KMP_DUPLICATE_LIB_OK` workaround, which the OpenMP runtime itself documents as unsafe.
  Partial records from both failed runs were deleted, so every committed curve comes from the
  same commit (`72c148e`).
- The first report included an old pre-SIMD brute-force record (no `kernel` field). The filter
  now requires a recorded non-scalar kernel.

**Open**
- FAISS leads at low ef. Hypothesis: per-call allocations in `search_layer` (heaps, output
  vector) and no prefetching. To be profiled in Phase 3, not assumed.
- M and ef_construction are unmeasured beyond the single configuration; that sweep belongs on the
  Ryzen.

## 2026-09-28: Profiling HNSW search: the allocation hypothesis is refuted

The M2 overheated and shut down during the previous session. On resuming, nothing from stage (c)
had been committed. The results, labeling, report fix, explainer section 6, and the ARM caveat
were all present on disk, and were committed as `1b1f9ec` and `8ce0868`.

**Thermal safeguards for this step**
- A 200k-vector subset instead of full SIFT1M: `scripts/make_subset.py` builds
  `sift1m-200k-q1000` (the first 200k base vectors and first 1000 queries, exact ground truth
  recomputed; brute force gets recall 1.0 against it, which validates it). Build takes ~40 s on
  one core, instead of ~5 min for SIFT1M.
- One heavy job at a time, checked with `ps` first; `pmset -g therm` before and after every run
  (`bench/profile_hnsw_search.sh` refuses to start on a warning); a 2-minute cool-down between the
  sweep and the profile; ground-truth BLAS pinned to one thread. No thermal or performance
  warning was recorded at any point.

**Profile** (`bench/profile_hnsw_search.sh sift1m-200k-q1000 40 6` at `9a5abdd`; macOS `sample`
for 6 s during the search phase only; raw report in
`results/profiles/sift1m-200k-q1000-ef40-9a5abdd.sample.txt`)

| Self time | Samples | Share |
|---|---:|---:|
| `search_layer` (inlined heap pushes, visited checks, neighbor-list loads) | 2176 | 47.9% |
| `neon::l2_squared` | 2060 | 45.3% |
| `std::__pop_heap` | 210 | 4.6% |
| Allocator (`_xzm_free`, `_free`, `_platform_memset`, and the `mach_absolute_time` calls `_xzm_free` makes) | 66 | 1.5% |
| `greedy_search` (upper layers) | 26 | 0.6% |

An earlier run of the same commands from a scratch script, before they were moved into `bench/`,
gave the same picture (allocator ~1.7% of samples under `HnswIndex::search`).

**Decision: no allocation fix.** Removing every per-query allocation could gain at most ~1.5-2%
QPS. That is the same size as the run-to-run QPS noise in the baseline sweep below (stdev
1-2%), and nowhere near the 10-30% gap to FAISS at low ef. Per the instruction ("fix it if it's
confirmed"), the core was not changed, so there is no before/after pair: the sweep below is the
"before" for the next optimization.

**Baseline on the subset** (`bench/run_search_bench.py --dataset sift1m-200k-q1000 --index
hnsw --runs 3` at `9a5abdd`; M=16, efC=200, single thread; Mac development results: recall final,
QPS indicative). Recall@10 / QPS: ef 10 0.766 / 39.1k; ef 40 0.956 / 14.7k; ef 80 0.988 / 8.3k;
ef 160 0.998 / 4.6k; ef 320 0.9995 / 2.6k. Raw record in
`results/search/sift1m-200k-q1000/`; table in `results/tables.md`.

**What the profile does point at.** Search time splits between the distance kernel and
`search_layer`'s own loop. Both are dominated by first touches of scattered memory: a 512-byte
vector per candidate, a 132-byte neighbor list per expanded node, and a 4-byte visited mark per
neighbor. `sample` cannot separate stall cycles from arithmetic. Candidates, each to be measured
on this subset before it is kept:
1. Prefetch the next neighbors' vectors (and visited marks) while computing the current distance,
   as hnswlib and FAISS do.
2. Compute distances for neighbors in batches of four (FAISS's `distances_batch_4`), so several
   cache misses are in flight at once.
3. A smaller visited set (1-byte epochs or a bitset) to cut its cache footprint (800 KB for 200k
   nodes today).

**Went wrong**
- The hypothesis I stated in stage (c) (per-call allocation explains the FAISS gap) was wrong. It
  was labeled as a hypothesis, and the explainer now says it was tested and refuted.

## 2026-09-28: Prefetching in search_layer: ~2x QPS on a 200k subset, results unchanged

**Change** (`src/index/hnsw.cpp`, `search_layer`). Each expansion now makes two passes over the
node's neighbor list. Pass 1 marks the unvisited neighbors, remembers them in a per-thread scratch
list, and issues `__builtin_prefetch` for each one's whole vector (one hint per 64 bytes). Pass 2
computes distances and updates the heaps in list order. Before either pass, it prefetches the
neighbor list of the next likely node to expand (the new top of C). The idea is that the memory
loads the profile pointed at overlap instead of queueing.

**Results cannot change, and did not.** A prefetch only affects timing. Marking a whole
(duplicate-free) neighbor list visited before computing distances selects the same neighbors in
the same order. Checked three ways:
1. All 19 `strata_reference` output files on SIFT10K (brute force, PQ, filtered, and HNSW in both
   selection modes, including the graphs they build) are bit-identical before and after.
2. Identical recall at every ef_search in all 20 A/B runs.
3. Identical graph statistics in all A/B runs.

**Measured** with `bench/run_ab_search.py`, a new script: the before binary (`7232d6e`) and the
after binary (`7232d6e` + `results/ab/*/b.patch`) alternate, in 5 pairs whose order flips each
pair. Each run is a fresh process with 5 timed passes per ef. M2, so QPS is indicative.

`results/ab/prefetch-sift1m-200k.md` (200k vectors, ~100 MB, far larger than cache):

| ef | recall (both) | before QPS | after QPS | after / before: mean (range) |
|---:|---:|---:|---:|---:|
| 10 | 0.7656 | 38,882 ± 4.9% | 82,165 ± 2.3% | 2.12 (1.99-2.36) |
| 20 | 0.8846 | 24,073 ± 6.0% | 49,408 ± 1.6% | 2.06 (1.98-2.27) |
| 40 | 0.9564 | 14,375 ± 3.2% | 28,619 ± 4.0% | 1.99 (1.86-2.17) |
| 80 | 0.9882 | 8,190 ± 3.0% | 16,005 ± 6.1% | 1.96 (1.73-2.08) |
| 160 | 0.9979 | 4,577 ± 3.2% | 8,723 ± 10.5% | 1.91 (1.56-2.05) |
| 320 | 0.9995 | 2,591 ± 3.9% | 4,666 ± 14.1% | 1.80 (1.34-2.04) |

(± is the stdev across the 5 runs. The table is copied from the generated file for the log;
the file is the source.) The worst single pair (1.34x) is still far above the run-to-run
noise. The wider after-spread at high ef comes from one slow run (3.5k QPS at ef 320). High ef is
the last part of each sweep, so it is most likely throttling late in that run; it lowers the mean
without changing the conclusion. Build: 40.4 s -> 25.3 s (0.63x), since insertion runs the same
search.

`results/ab/prefetch-siftsmall.md` (SIFT10K, 5 MB, cache-resident): no measurable difference
(mean 0.96-1.00x, every range straddles 1; build 0.6 -> 0.7 s). A two-pair smoke test had
suggested a 0.83-0.89x slowdown there; the full protocol shows that was noise. Lesson: don't read
a result off two runs.

**Caveat: Apple's hardware prefetchers.** The M2 has aggressive hardware prefetchers, and x86
cores behave differently (smaller 64-byte lines, different prefetchers and miss handling).
The gain may be smaller or larger on the Ryzen, so it is re-measured there in Phase 9 before it
is quoted as a general result. The SIFT1M curves in `results/hnsw/` predate this change and were
not re-run on the Mac.

**Thermal safeguards** as before: one heavy job at a time, `pmset -g therm` checked before and
after every run inside the script (it stops at the first warning), and 60 s cool-downs (20 s on
SIFT10K, where each run takes seconds). No warning was recorded.

**Decision: keep.** Next candidate: batched distance computation.

## 2026-09-28: Re-profile after prefetching; updated comparison on the 200k subset

**Went wrong first: a profile under contention.** I started the re-profile even though my own
check showed another project's test suite (`~/Lumen`, `pytest`) using ~3 cores. That broke the
one-heavy-job rule, and it showed in the data: the build took 42.6 s instead of ~25 s. That
profile was deleted, not used. To stop it from happening again, `benchmeta.preflight()` and
`bench/profile_hnsw_search.sh` now refuse to start a heavy step while another process uses more
than half a core (`4a7bf38`). The first version then aborted the comparison run on a single 75%
spike from the editor, so preflight now waits for three quiet checks 5 s apart and gives up
after 5 minutes of sustained load (`bc33dea`). A thermal warning still stops everything
immediately. The partial records from the aborted run were deleted, and the comparison was rerun
in full.

**Profile with prefetching** (`results/profiles/sift1m-200k-q1000-ef40-4a7bf38.sample.txt`, quiet
machine, build 26.4 s). Self time, compared with before prefetching (`...-9a5abdd`):

| | before prefetch | with prefetch |
|---|---:|---:|
| `search_layer` own loop | 47.9% | 59.5% |
| `neon::l2_squared` | 45.3% | 32.8% |
| `std::__pop_heap` | 4.6% | 3.0% |
| `greedy_search` self | 0.6% | 1.6% |
| allocator | ~1.5% | ~2.5% |

In the call tree under `HnswIndex::search` (first half of the samples): the `search_layer` loop
is 65%, distances called from it 28%, and the upper-layer `greedy_search` 12%. Most of that 12%
is its own distance calls, which are not prefetched, and its share grew because layer 0 got
faster. The distance kernel now mostly finds its vectors in cache, so what remains there is
closer to pure arithmetic. `sample` cannot split the 60-65% `search_layer` self time further,
because the release build has no line-level debug info and the loop is inlined.

**Updated comparison on the same subset** (`results/hnsw/hnsw_vs_reference.md`, section
`sift1m-200k-q1000`, generated by `bench/run_hnsw_curves.py` at `bc33dea`; M=16, efC=200, single
thread, 3 runs; M2, so QPS is indicative). Recall@10 / QPS:

| ef | Strata | FAISS | hnswlib |
|---:|---|---|---|
| 10 | 0.766 / 66.2k ± 27% | 0.774 / 47.5k ± 5% | 0.768 / 21.2k ± 1% |
| 40 | 0.956 / 25.0k ± 3% | 0.960 / 15.9k ± 9% | 0.957 / 7.7k ± 12% |
| 80 | 0.988 / 13.7k ± 9% | 0.990 / 7.6k ± 25% | 0.988 / 4.6k ± 6% |
| 160 | 0.998 / 8.0k ± 26% | 0.997 / 4.7k ± 3% | 0.998 / 2.7k ± 2% |
| 320 | 0.9995 / 4.6k ± 5% | 0.9994 / 2.4k ± 1% | 0.9995 / 1.5k ± 11% |

(Copied from the generated file for the log; the file is the source.)
- With prefetching, Strata leads FAISS at every ef on this subset, by roughly 1.4-1.9x. That
  reverses the pre-prefetch SIFT1M picture (0.7-0.97x). But the two runs use different
  datasets (200k vs 1M) and different commits, so they are not directly comparable. The SIFT1M
  curves have not been re-run on the Mac.
- This run is noisy. Individual Strata passes dropped (46k at ef 10 and 5.6k at ef 160, against
  ~75k and ~9k in the other runs), consistent with throttling. Strata's ef 10 mean (66k) is below
  the 82k from the 5-pair A/B. The direction is consistent at every ef; the size of the margin
  is not established.
- FAISS still gets slightly higher recall at low ef (0.774 vs 0.766 at ef 10).
- hnswlib: ~3x behind, but it has no NEON path on ARM, so this comparison favors Strata. Fair
  comparisons against both libraries come from the x86 runs in Phase 9.

**Recommendation: not batched distances next.** Batched distances target the distance kernel,
now ~33% of search time. Their main benefit in FAISS, keeping several vector loads in flight, is
largely what prefetching already provides; what remains is arithmetic (reusing the query across
four candidates). Even a 25% faster kernel would be ~8% overall. The largest cost is now
`search_layer`'s own loop (~60%), and we cannot yet see what inside it is expensive. Next, in
order:
1. **Attribute `search_layer`'s self time at line level**: one profiling run with a
   release-with-debug-info build. The suspects are the visited-mark loads (4 bytes per neighbor,
   not prefetched, scattered over 800 KB), heap pushes, and the prefetch hints themselves (8 per
   vector, half of them redundant with the M2's 128-byte lines).
2. **Prefetch in `greedy_search`**, the upper-layer walk: the same change, ~12% of search time,
   low risk.
3. Then choose between the item that step 1 shows is largest and batched distances, each
   measured with `bench/run_ab_search.py` as before.

## 2026-09-28: Prefetching in greedy_search: no measurable gain, not kept; search_layer work deferred

**Tried:** prefetching every neighbor's vector at the start of each step of `greedy_search`, the
upper-layer walk, before computing any distance (the same idea as in `search_layer`). The profile
had put ~12% of search time in this walk, most of it in its distance calls.

**Results unchanged, as required.** All 19 `strata_reference` files were bit-identical to HEAD
(`840f88b`), and recall and graph statistics were identical in all 10 A/B runs.

**Measured** with `bench/run_ab_search.py` on `sift1m-200k-q1000` (5 interleaved pairs, 5 timed
passes per ef; M2, so QPS is indicative; no thermal warnings;
`results/ab/prefetch-greedy-sift1m-200k.md`): mean after/before ratios 1.14, 1.06, 1.05, 1.03,
0.99, 1.05 at ef 10-320. Every range straddles 1 except ef 320 (1.01-1.09). Run-to-run spread was
4-11%. Build time was unchanged (1.003x).

**Decision: not kept.** The gain is not distinguishable from the noise, which is the bar
prefetching in `search_layer` cleared easily (worst pair 1.34x). That fits the setting: upper
layers hold ~1/M of the nodes, and every query walks down from the same entry region, so those
vectors are probably already cached. At most ~12% of search time is in this walk, so even a real
gain would be a few percent. The core stays at `840f88b`. The exact patch is saved in
`results/ab/prefetch-greedy-sift1m-200k/b.patch`, to re-test on x86, where the hardware
prefetchers differ.

**Deferred to Phase 9 (AWS session, x86, `perf`):** the `search_layer` investigation. That covers
line-level attribution of its ~60% self time, the visited-mark loads, heap operations, and the
prefetch hints (8 per vector, half redundant on the M2's 128-byte lines). `sample` on the Mac
cannot attribute inlined code without a debug-info build, and `perf` with hardware counters on
x86 can show cache misses and stalls directly. Batched distances are also set aside for now: the
distance kernel is ~33% of search time and mostly finds its vectors cached, so the expected gain
is small.

**Housekeeping in the same commit:** `buildplan.md` and `docs/checklist.md` reconciled with the
repo (status dated 2026-09-28). They adopt the finish order: HNSW persistence and deletes;
parallel build; filtered HNSW with automatic strategy selection; gRPC and sharding on the Oracle
ARM machine; one AWS session for the x86 finals, the 10M run, and thread and multi-machine scaling;
final ARM numbers on Oracle; then README, design doc, and going public. PQ-in-HNSW, WAL group
commit, and extra BEIR datasets are marked optional. One item was not placed and needs a
decision: HotpotQA answer groundedness on the full 5.2M corpus. The Linux build (2026-09-25)
predates the HNSW core, so rebuilding and testing on Linux is the first step on the Oracle machine.

## 2026-09-28: HNSW Stage A: tombstone deletes and snapshot save/load

**Done**
- **Snapshot format version 2** (`include/strata/snapshot.hpp`): a byte-order mark
  (`0x01020304`, checked on load; the opposite order is named in the error), an index-kind field,
  and an index section, still under one CRC32C and one atomic write. Version 1 files still read, as
  flat snapshots, and unknown versions are rejected. Host byte order and IEEE floats are
  `static_assert`ed. Sizes read from the file are guarded against overflow.
- **HNSW deletes** (`remove`, `is_deleted`, `live_size`): tombstones only. Deleted nodes are still
  traversed, and inserts still link to them, so the graph depends only on the inserts. The
  trade-offs (no memory reclaimed, searches slow under heavy deletion, rebuild as the remedy) are
  documented in `hnsw.hpp` and in explainer section 9.
- **Heavy deletion (your addition 1): the search widens automatically.** In the tombstone-aware
  `search_layer`, deleted nodes are expanded but never enter W, and the stopping rule applies only
  once W holds ef live nodes. With 95% deleted and ef = k = 10, every query still returns 10 live
  results (recall at least 0.9 on the live set). With 7 live nodes, all 7 come back in brute-force
  order. With everything deleted, a search returns immediately. Without deletes the plain
  instantiation keeps the paper's rule exactly: all 19 `strata_reference` files are bit-identical
  to `840f88b`.
- **HNSW persistence** (`to_snapshot`/`from_snapshot`, `save`/`load`): parameters, levels, every
  list, entry point, tombstones, and the level generator's state (the standard-specified text form
  of `std::mt19937_64`, classic locale). Loading validates structure (bounds-checked reads;
  parameters; max level and entry point; list counts within capacity; neighbor ids exist on their
  layer), so a damaged file fails to load rather than crashing a later search.
- **Python:** `HnswIndex.remove`, `is_deleted`, `live_count`, `save`, `HnswIndex.load` (str or
  `pathlib.Path`).

**Tests** (26 new C++, 4 new Python)
- Save then load is bit-identical: graph and search results (ids and float bits) for L2, IP, and
  cosine, and for both selection modes.
- Save, load, then add equals never saving: same checks. A deliberate mutation (parsing but not
  restoring the generator state) made all 4 cases fail, so the test catches the thing it is for.
- Corrupt files: a flipped byte, an unknown snapshot version, the opposite byte order, an unknown
  graph version, an entry point out of range, a neighbor id out of range, a count over capacity,
  and a truncated or padded section. The structural cases recompute the CRC so they reach the
  structural checks.
- Deletes: errors and counts, never returned, recall on the live set at 30% deleted, a deleted
  entry point, 95% deleted, fewer live nodes than k, everything deleted, and deletes not changing
  the graph.
- **Cross-machine (your addition 2):** `tests/golden/hnsw_v2.snap` (49 KB) and its expected results
  were written on the Mac. `HnswGolden.LoadsBitIdenticallyOnThisMachine` must reproduce them bit for
  bit on every machine, and rebuild the identical graph from the same vectors and seed. The
  vectors are small integers, so all SIMD kernels give identical exact distances. Regenerate with
  `STRATA_WRITE_GOLDEN=1` only after a deliberate format change. It runs on Linux for the first
  time on the Oracle machine.

**Next (Stage B):** `Collection` backed by HNSW (index kind and HNSW parameters stored and
checked on reopen, per your addition 3), WAL replay into HNSW, crash tests for HNSW, and a
save/load vs. rebuild measurement on the 200k subset.

## 2026-09-29 — First Linux build (Oracle ARM, GCC 13, branch `linux-arm-check`)

Oracle machine: aarch64, 4 cores, 23 GB, Ubuntu 22.04, GCC 13.4, presets `linux-debug`,
`linux-asan`, `linux-tsan` (no preset changes needed: AVX2 is off on aarch64, NEON is baseline).
All 216 C++ tests pass under all three presets except the golden test, with no ASan, UBSan, or
TSan reports. Python: 114 passed, 26 skipped (datasets and the API key absent).

**What went wrong**
- **Golden snapshot does not load on Linux ("bad generator state"). Not fixed; needs a decision.**
  libc++ writes `std::mt19937_64`'s state as the standard's 312 words; libstdc++ writes its raw
  312-word array plus its internal index (313 numbers) and requires that index when reading. So
  snapshots do not cross between macOS and Linux in either direction, and the comment in
  `include/strata/hnsw.hpp` that the standard makes this format portable is wrong in practice.
  Everything else is portable: a scratch rebuild of the golden index with GCC gives byte-identical
  levels, neighbor lists, vectors, and tombstones, and all 200 search results match the Mac's
  ids and float bits. Only the generator text, its length fields, and the CRC differ.
- **UBSan: `memcpy` with a null pointer in `SectionReader::read_all`** (empty span, e.g. an empty
  index or a level-0 node's upper lists). Undefined even for zero bytes; glibc declares `memcpy`
  nonnull, so UBSan reports it on Linux, and macOS's libc does not. Fixed with an empty-span guard
  (the same guard `snapshot.cpp` already has). No behavior change.
- GCC-only warnings (`-Wmissing-field-initializers` on designated initializers in `filter.cpp`,
  `-Wsign-conversion` on `x += c ? 1 : 0`, `-Wcomment` on a `\` in two usage comments): benign,
  left as they are.
- The Python extension is built by the default compiler (GCC 11.4 here), not the presets' GCC 13.

**Follow-up (same branch)**
- **One byte-copy helper.** Every `memcpy` in src and tests now goes through
  `util::copy_bytes` (`src/util/bytes.hpp`), which skips zero-length copies; there were no
  `memmove` calls. Audit: 12 sites. The fixed-size ones (`sizeof(T)` into or out of a local) were
  already safe, and the check folds away for a constant length. The ones that could see an empty
  buffer were the two already guarded by hand (`snapshot.cpp`, `hnsw.cpp`), which now use the
  helper, the WAL replay copy (safe today only because `dim > 0` is validated), and a test reading
  a possibly empty file. `tests/bytes_test.cpp` covers the helper; with its guard removed, the
  asan preset reports it. The test uses a runtime length because GCC deletes a `memcpy` whose
  length is the constant 0 before UBSan sees it.
- **Python extension built with GCC 13 on Linux.** `CMakeLists.txt` picks `/usr/bin/g++-13` for
  `STRATA_BUILD_PYTHON` builds on Linux when it exists and no `CMAKE_CXX_COMPILER` was given, so
  Kaggle (no GCC 13) keeps its default and macOS is untouched. `CXX` can't be the opt-out: uv's
  build environment sets `CXX=c++` from Python's sysconfig, which made a first version (that
  deferred to `CXX`) silently keep GCC 11. Verified: `strata.build_info()` reports GNU 13.4.0.
  The only GCC 11 code left in the module is vcpkg's `libutf8proc.a` (a C library vcpkg builds
  with the system compiler), the same as in the preset builds.

## 2026-09-29: Merge of linux-arm-check; snapshot format version 3 (portable generator position)

**Merged** `origin/linux-arm-check` (`d2faf40`, `f00405f`): `util::copy_bytes` for every byte copy
(zero-length copies with null pointers are undefined, and glibc's UBSan reports them), and GCC 13
for the Python extension on Linux. First built on the Mac here: clang-format made no changes, no
raw `memcpy` remains outside `src/util/bytes.hpp`, and all 218 C++ tests pass under debug, asan,
and tsan, with 140 Python tests (the live API test ran and passed; the usage cap has reset).

**Fixed: HNSW snapshots did not cross between macOS and Linux.** The first Linux run found it
(entry above): format version 2 saved `std::mt19937_64` as stream text, and libc++ writes 312 words
while libstdc++ writes 313 (its state plus an internal index) and requires the extra one. Everything
else in the file was already portable.
- **Format version 3:** the index section (now section version 2) stores the number of generator
  draws since seeding instead of the text. Every engine call goes through `HnswIndex::draw()`,
  which counts calls, not adds. Loading re-seeds and calls `discard(count)`, which the standard
  defines identically for every library.
- **Version 2 files** load only on the standard library that wrote them. There the draw count is
  recovered (re-seed and check that one draw per node reproduces the saved state), so the next
  save writes version 3. Elsewhere they fail with an error naming the libc++/libstdc++ mismatch and
  saying to re-save on the machine that wrote them. The snapshot file version bump also makes
  pre-version-3 readers reject new files cleanly.
- **Fixtures:** `tests/golden/hnsw_v3.snap` (new, written on the Mac) is the cross-machine test,
  now also checking that inserts after the load stay identical to a fresh build. The old fixture
  is kept as `hnsw_v2_libcxx.snap`: it must load and upgrade under libc++ and be rejected clearly
  under libstdc++. A new test rewrites its generator text into the other library's shape, so the
  rejection runs on the Mac too. Both fixtures give byte-identical expected results.
- **Mutation check:** without the `discard`, 6 tests fail (both golden tests and all 4 save, load,
  then add cases).
- The `hnsw.hpp` and `snapshot.hpp` format comments and explainer section 9 are corrected. They
  had said the standard's text format made the state portable.

**Not yet confirmed:** the version 3 fixture on Linux. The next Oracle run should show the golden
test passing, and the version 2 fixture rejected with the new message.

## 2026-09-29: HNSW Stage B: Collection on HNSW, WAL replay, crash tests, measurements

**Done**
- **`Collection` backed by either index.** `CollectionOptions{.index = kFlat | kHnsw,
  .hnsw = HnswParams}` is held internally as a `std::variant` (no virtual calls in search).
  `search` gains `ef_search`, plus `index_kind()` and `hnsw()` for introspection.
- **Settings are fixed at creation (your addition 3).** Reopening with a different index kind, M,
  ef_construction, selection, or seed fails with `kInvalidArgument`, naming each field with both
  values (e.g. "M: on disk 8, requested 16"). **A new collection writes an initial empty
  snapshot before its WAL,** so the settings are on disk from the start and are checked even
  before the first checkpoint. Directories from before this rule (a WAL, no snapshot) can only be
  flat and open as flat.
- **WAL replay into HNSW:** inserts and deletes after the snapshot's LSN replay in order, and the
  recovered graph is exactly the uninterrupted one. `Collection::open` was split into helpers
  (`load_or_create`, `apply_record`) to keep clang-tidy's complexity limit.
- **HNSW is unconditional in CMake** (`Collection` depends on it). `STRATA_HAS_HNSW` stays
  defined for existing `#ifdef`s.

**Tests**
- `HnswCollectionTest`: recovery rebuilds the same graph, from WAL only and from checkpoint + WAL
  (compared with an index that never touched disk). Reopening with each different setting fails
  with the right message. Settings are checked before the first checkpoint. Flat collections,
  including pre-rule WAL-only directories, refuse HNSW.
- **Crash tests extended to HNSW:** 4 new modes (WAL only, frequent and rare checkpoints, fsync),
  forked writer SIGKILLed at random points. After every round the recovered graph must equal the
  graph the recovered writes build without a crash, and no search may return a deleted id. All
  pass, alongside the 4 flat modes.
- The shared graph comparison (`tests/hnsw_test_util.hpp`) has a test showing it can fail
  (different seeds, an extra tombstone).

**Measured** (Mac development results; recall final, speed indicative; one heavy job at a time
behind `preflight`, cool-downs, no thermal warnings; recorded at `0f25f11`, the committed bench
tooling, with the uncommitted `Collection` work stashed so the records are clean. The measured
HNSW core is unchanged since Stage A.)
- **Save/load vs. rebuild**, 200k subset (`results/storage/hnsw_persist_sift1m-200k-q1000.md`):
  build 23.5 s, save 0.51 s, load 0.36 s, 124 MiB. Loading is ~66x faster than rebuilding, with
  identical answers in every run.
- **Search under deletion** (`results/hnsw_deletes/deletes_sift1m-200k-q1000.md`): QPS at matched
  recall, tombstoned vs. rebuilt over the live vectors: 25% deleted 0.83x / 0.79x (recall 0.95 /
  0.99), 50% 0.69x / 0.64x, 90% >= 0.23x / 0.24x. Rebuild times: 17.3 / 10.1 / 1.5 s. Recall rises
  with deletes (the search widens); speed is what suffers. **Guideline** (explainer section 9):
  rebuild once a quarter to a half of the index is deleted, if search speed matters.
- The sweep compares at matched recall, not matched ef_search: under deletion the same ef gives
  higher recall, so matching ef would flatter tombstones. Where even ef = k already exceeds the
  target recall, the report marks the value as a lower bound and omits ratios it would make
  meaningless.

**Went wrong along the way**
- The first version of the delete report printed a ratio of two lower bounds as if it were
  exact (0.17x). Now a lower bound in the denominator gives "—".
- **Provenance after the version 3 change:** these measurements were taken at `0f25f11`, before
  snapshot format version 3. Search code is unaffected. Save and load now write and read a draw
  count instead of the generator text, and loading re-seeds and discards, so the load time may
  shift slightly; it has not been re-measured. Stage B was rebased onto the merge and version 3
  by stash and pop; the only conflict was this devlog.

**Not done / next**
- No rebuild or compaction operation exists (a rebuild assigns new ids; there is no mapping).
- Python has no `Collection` binding (it had none before either).

## 2026-09-29: Save/load re-measured with snapshot format version 3

`bench/run_hnsw_persist_bench.py` at `397bd38` (200k subset, 3 runs, M2, no thermal warnings):
build 25.63 ± 1.89 s, save 0.525 ± 0.107 s, load 0.393 ± 0.056 s, 124 MiB; loading ~65x faster
than rebuilding, with identical answers in every run. Version 2 (`0f25f11`) measured build 23.48 s,
save 0.513 s, load 0.357 s. The difference is within noise: run 2 was 15–25% slower in every step,
including the build, which the format cannot affect, so it was the machine. The draw-count restore
(re-seed and discard) has no visible cost. Explainer section 9 now quotes the version 3 numbers.

## 2026-09-29: Parallel HNSW build

**Design (approved):** `HnswIndex::add_batch(vectors, ThreadPool&)`; the sequential build stays the
deterministic default; Python `HnswIndex.add(vectors, threads=1)` releases the GIL.
- **Prepare first:** store all vectors, draw all levels in id order (the same draws as sequential),
  and allocate every list at its final size. No array moves during the parallel phase.
- **Link in parallel,** one id at a time (`parallel_for`, chunk 1).
- **Locks:** a striped table of 65,536 `std::mutex` for neighbor lists (option A; about 4 MiB,
  fixed) and one `top` mutex for the entry point and max level. A promoting insert holds `top`
  until done. No thread holds two node locks at once, so there is no deadlock.
- **Zero cost when sequential:** locking is behind `if constexpr`. The C++ reference outputs are
  bit-identical, and a one-thread pool builds exactly the sequential graph.
- **`Collection` stays sequential:** WAL replay must reproduce the graph.

**Went wrong, and fixed (found by the new tests)**
1. **Lost back-links.** A back-link added to a node before it wrote its own list was overwritten.
   Layer-0 reachability on random 6000 x 32: 1.0 sequential vs 0.997 (4 threads) and 0.994 (8).
   Fixed with `merge_links` (keep the early back-links; re-select only on overflow): 0.9993-1.0
   over repeated runs.
2. **Duplicate links** (two concurrent nodes selecting each other). Fixed by skipping existing
   back-links in concurrent mode.
3. **Self-loops** (a node reachable on a layer before it is linked there found itself). Fixed by
   removing the node from its own candidates in concurrent mode.
- **The stress test's reachability bar was wrong,** not the code. Its data (5 tight clusters,
  M = 4) gave 0.2-1.0 reachability over 20 *sequential* builds in shuffled orders, so it cannot
  separate a bug from insertion order. The stress test now checks structure and races only;
  reachability is checked on random data and SIFT10K.

**Tests** (`tests/hnsw_parallel_test.cpp`, 6 C++ tests plus 1 Python): one thread equals sequential
exactly; quality vs sequential at 4 threads on random data and SIFT10K (well-formed, same levels,
reachability >= 0.999, degree within 5%, recall within 0.01 at ef 10/40/160); high-contention
stress (M = 4, 8 threads, 3 rounds); save/load after a parallel build then sequential adds are
deterministic; a parallel batch over tombstones plus errors. They passed 10-15 repeats in debug,
3 repeats under TSan with no race reports, and under ASan. The SIFT10K quality test skips itself
under TSan (too slow there); it runs in debug and asan.

**Measured** (`bench/run_hnsw_build_scaling.py` at `97f4dda`, 200k subset, 3 interleaved runs per
thread count, 60 s cool-downs, no thermal warnings; M2, indicative):
build 24.55 ± 0.12 s (1 thread) / 13.58 ± 0.27 s (2, 1.81x) / 7.68 ± 0.12 s (4, 3.20x). Recall@10 at
ef 10/40/160 within 0.0004 of sequential, layer-0 reachability 1.0 and mean degree 20.3 at every
thread count. The full 1-16 thread curve comes from the AWS session.

## 2026-09-30: Filtered HNSW search with automatic strategy selection

**Design (approved):** `HnswIndex::search_filtered` for a `CompiledFilter` or a `Bitset`, with
strategies kGraph (filtered-out nodes traversed like tombstones: keep-policy templates on
`search_layer`, same widened stopping rule), kPreFilter (exact), and kAuto (estimate selectivity,
choose, fall back). Python: `HnswIndex.search_filtered(..., strategy=, ef_search=,
prefilter_below=, fallback_budget=)`, accepting a compiled filter, bool mask, or id array.
`Collection` filtering is future work.

**Your additions:**
- (a) **Two-stage sampling:** 1,000 ids, and 20,000 when the first estimate is within 3 standard
  errors of the threshold. **Runtime fallback:** a graph search past `(fallback_budget +
  estimated selectivity) * n` distances (about the pre-filter's own cost) switches that query to
  the pre-filter; `FilteredSearchStats` reports it and the benchmark counts it.
- (b) The benchmark report flags any graph recall ceiling (recall below 0.99 with under 0.005
  gained from ef 160 to 320).

**Went wrong, and fixed**
- **The first fallback budget was just `fallback_budget * n`.** On the 4,000-vector test index
  that is 400 distances, less than a plain search at ef 64 computes, so a 50% filter fell back.
  The budget now adds the estimated selectivity (the pre-filter's per-match distances), which is
  what the pre-filter would actually cost.
- **A dangling reference in a test helper:** `for (n : *run(...))` iterated a temporary
  `Expected` that was already destroyed. That produced recall 0 while the search itself was
  correct (checked against exact results directly).
- **Slow `in` filters:** int `in` was an OR of ranges (one test per value per id), and category
  `in` scanned its set. At 500 clusters the pre-filter slowed about 20x. Both are now sorted sets
  with binary search above 16 values; `Filter.LargeInSetsMatchMembership` covers it.

**Tests** (`tests/hnsw_filter_test.cpp`, 13, plus 1 filter test and 1 Python test):
- graph recall against exact filtered brute force at 1%, 10%, and 50%;
- the pre-filter is exact;
- no non-matching or deleted id is ever returned (5 selectivities, 4 option sets);
- filters matching nothing (empty) and everything (bit-identical to unfiltered);
- filters combined with deletes;
- fewer matches than k;
- auto picking each side of a per-call threshold;
- resampling near the threshold;
- fallback firing (and never for a forced graph);
- `Bitset` and `CompiledFilter` forms agreeing;
- errors;
- concurrent filtered searches under TSan.

249 C++ tests pass under debug, asan, and tsan, and 141 Python.

**Measured** (`bench/run_hnsw_filter_bench.py` at `08eb3cb`, 200k subset, 500 queries, 3 runs; M2,
indicative; no thermal warnings). The first attempt was refused by `preflight`: a Chrome tab held
60-140% of a core for 5 minutes. A background waiter started the run once the machine had been
quiet for a minute.
- **Crossover:** random 1.03% / 1.04% (recall 0.95 / 0.99), correlated 1.25% / 1.28%. The
  default `kDefaultPrefilterBelow` is now **1.3%** (the largest, rounded; it was provisionally 2%).
- **Auto** chose the pre-filter for 100% of queries at 0.1-1.1% selectivity (recall 1.0), and the
  graph for 100% at 10% and 50%.
- **Fallback fired 0%:** every measured selectivity is far from the threshold, so it was never
  needed. Tested, not yet measured; measuring it needs selectivities near 1-3%.
- **No correlated recall ceiling:** the graph reaches recall 1.0 by ef 40-320 everywhere.
  Queries whose own cluster matches have lower recall at low ef (0.89 vs 0.95 at 10%, ef 10),
  equal by ef 80. ACORN-style predicate-aware traversal is noted as future work in case larger
  scale shows a ceiling.
- **Weakness:** auto's per-query 1,000-id sample makes it slower than a forced graph search at
  high selectivity and low ef (random 50% ef 10: 32k vs 43k QPS; correlated: 14k vs 41k, noisy).
  Fix options: a `Bitset` filter (exact popcount), or estimate once per filter.
- The crossover depends on n; re-measured at 1M and 10M in the AWS session.

### 2026-09-30: selectivity estimated once per filter

- `CompiledFilter` caches its coarse (1,000-id) and precise (20,000-id) estimates behind
  `std::call_once`, in a `shared_ptr` so copies share them and the filter stays cheap to copy.
  `FilteredSearchOptions::selectivity` lets a caller pass a known value; the Python binding counts
  its bitset once per batch and passes it. Caching is safe because a compiled filter must not be
  used across appends to its table. Three new tests (override honored, cache shared by copies,
  concurrent first estimates under TSan); 252 C++ tests pass under debug, asan, and tsan, and 141
  Python.
- The driver gained `--label` / `--threshold`: a re-measurement into its own directory and report,
  with forced graph and auto in the same session.
- **Measured** at `58e6e30` (200k subset, 10% and 50%, threshold fixed at 1.3%; no thermal
  warnings): the gap is closed. At 50%, ef 10, auto / graph QPS is 1.10 (random, was 0.74) and
  1.05 (correlated, was 0.34); the median over all 24 points is 1.00, with identical recall.
  Outliers (0.47 at random 10%, ef 40; 0.83-0.89 at three others; 1.17 at one) each have a large
  standard deviation on one side, and auto does the same work as the graph once the estimate is
  cached, so they are noise on the fanless Mac.
- **Plan change:** the fallback-rate measurement at 1-3% selectivity is not run on the Mac. The AWS
  crossover sweeps at 1M and 10M include 1%, 1.5%, 2%, and 3% instead (buildplan.md,
  docs/checklist.md).

## 2026-10-01: Phase 7 begins: gRPC shard and coordinator (first commit)

**Written by another session** (its report, verified here):
- `proto/strata/v1/vector_service.proto`: one `VectorService` (Insert, InsertBatch, Search, Delete,
  Stats) served by both a shard and the coordinator, so clients cannot tell them apart.
- `server/shard_service`: gRPC in front of `Collection`, so it keeps WAL, snapshots, and crash
  recovery. `strata_shard`: `--dir --dim --metric --index flat|hnsw --port` and HNSW settings.
- `server/coordinator_service`: round-robin inserts, parallel search on every shard merged into
  one top-k, delete routed to the owning shard, summed stats. `strata_coordinator`.
- `server/id_codec.hpp`: global id = local * num_shards + shard_index (no lookup table; the
  order of the `--shard` flags must stay fixed across restarts). `status_util.hpp` maps Strata
  errors to gRPC status codes.
- Build: `STRATA_BUILD_SERVER` (off by default) and a `server` vcpkg feature for Linux.
- Tests: id codec round trips, and 3 real shards on localhost checked against exact brute
  force, including delete and batch insert.

**Verified here (Mac, Homebrew gRPC 1.84.0, protobuf 36.2, abseil 20260817)**
- The default build is unchanged: 252 tests pass, no warnings.
- The server, coordinator, and tests build with no warnings, and all 6 server tests pass.
- `Collection::remove` returns `kNotFound` for an unknown or already-deleted id, before logging
  anything, so a repeated Delete fails cleanly (the other session's unchecked assumption).
- **ASan reported a container-overflow, judged a false positive.** It fires inside protobuf's
  `RepeatedField<float>::data()` (`shard_service.cpp:14`), reading the field's own pointer inside
  an 87-byte gRPC arena block: in bounds. Protobuf annotates `RepeatedField` storage only when
  compiled with ASan; our generated code and inlined headers are, but Homebrew's libprotobuf and
  libgrpc (which parsed the request) are not. That is the mixed-instrumentation false positive the
  ASan docs describe. With `ASAN_OPTIONS=detect_container_overflow=0` (every other check still on),
  all 6 tests pass with no ASan or UBSan report. The real fix is gRPC/protobuf built with ASan
  (possible through vcpkg on Linux); until then server ASan runs use that option.

**Changes to this machine (done by the other session):** `brew install grpc protobuf` (prebuilt
bottles) with `yes` piped in, which also auto-approved installing abseil and re2 and upgrading
openssl@3, c-ares, and ca-certificates system-wide. This can affect other projects.

**Known gaps, not yet addressed**
- Insecure: no TLS or auth, and both servers listen on 0.0.0.0.
- `InsertBatch` is not atomic across shards.
- Global ids are 32-bit (about 4.29 billion vectors).
- Search starts one thread per shard per query.
- No presets or README steps for the server build.
- tsan not yet run.
- Not yet built on Linux.

## 2026-10-01: Phase 7 build fixes and sanitizer status

- **Presets.** `server` and `server-asan` (macOS, gRPC/protobuf/abseil from Homebrew via
  `CMAKE_PREFIX_PATH=/opt/homebrew`); `linux-server`, `linux-server-release`, `linux-server-asan`
  (vcpkg feature `server`, GCC 13). The README has a matching section. The Linux presets are not
  yet tried (no Linux build of the server so far).
- **ASan workaround scoped.** `detect_container_overflow=0` is now set only on the server tests
  (a `gtest_discover_tests` property under `STRATA_SANITIZE`), so the core ASan suite keeps that
  check.
- **TSan is not possible against stock protobuf.** A TSan build of the server crashed (SEGV in
  `ZeroFieldsBase::~ZeroFieldsBase`, destroying a `StatsRequest`). Cause: when the compiler
  defines `ABSL_HAVE_THREAD_SANITIZER`, protobuf's `port_def.inc` declares an extra
  `char _tsan_detect_race` member in every message, so our TSan-compiled generated code and
  Homebrew's non-TSan libprotobuf disagree on message layout. Not a Strata bug. Server TSan needs
  gRPC and protobuf built with TSan (a vcpkg sanitizer triplet, on Linux). CMake now warns when
  `STRATA_TSAN` and `STRATA_BUILD_SERVER` are both on; there is no server TSan preset.
- **Versions recorded.** `strata_shard` and `strata_coordinator` print the gRPC, protobuf, and
  abseil versions they were built against on their startup line, since Homebrew (Mac) and vcpkg
  (Linux) can differ. On this Mac: gRPC 1.84.0, protobuf C++ runtime 7.36.2 (protobuf 36.2),
  abseil 20260817. The startup line is now flushed: with stdout redirected to a log file it was
  block-buffered and never appeared while the server ran.
- **Cleanup.** `strata_server`'s include path is now `server/` only (was the whole repository
  root), and a CMake comment that wrongly credited `POSITION_INDEPENDENT_CODE` for the generated
  code having no strict warnings is corrected (it's because `strata_proto` doesn't link
  `strata_options`).
- Delete-twice was already covered (`sharding_test.cpp`): Delete is idempotent.

Remaining known gaps: no TLS/auth and listening on 0.0.0.0; non-atomic `InsertBatch`; 32-bit
global ids; one thread per shard per query; not yet built on Linux.

## 2026-10-01: Server hardening: localhost default, async fan-out, TLS + token, InsertBatch retries

**Exposure.** Both binaries listen on 127.0.0.1 unless `--listen` says otherwise. A non-loopback
listen needs TLS and a shared token, or `--insecure`; a token without TLS is refused (it would
travel in plain text, and gRPC won't attach call credentials to an insecure channel anyway). The
binaries check this before opening the collection, so a refused start does no work.

**Fan-out.** Search, Stats, and the per-shard InsertBatch calls now use gRPC's callback API: all
shard calls start at once and complete on gRPC's threads, so a query holds no thread per shard
(the old Search started one `std::async` thread per shard per query; Stats and InsertBatch were
sequential). I chose callbacks over a thread pool: a pool of blocking calls would cap in-flight
shard calls at its size and queue the rest, and its size would need tuning per shard count. The
configurable sizes are the gRPC server's thread cap (`--max-threads`) and the per-shard-call
deadline (`--shard-timeout-ms`, default 10 s; before, a hung shard hung the request forever).
The thread-per-shard Search is kept behind `SearchFanout::kThreadPerShard`, only as the baseline
the benchmark measures against.
- The countdown the callbacks share lives in a `shared_ptr` each callback holds. With a
  `std::latch` on the handler's stack, the waiter could return and destroy it while the last
  callback was still inside `count_down`.
- Found by a test: after a shard restarts, gRPC's default reconnect backoff (up to 120 s) keeps
  failing calls fast with the cached "connection refused". The coordinator's channels now back
  off 100 ms to 2 s.

**TLS + token.** `scripts/make_dev_certs.sh` makes a CA, a server certificate with the given
SANs, and a 64-hex-character token (works with OpenSSL 3 and macOS LibreSSL; the build runs it to
make the test certificates, so the script is tested too). Shards: `--tls-cert/--tls-key/
--token-file`; coordinator: `--shard-ca/--shard-token-file` plus the same server flags for its own
port. The token check is an `AuthMetadataProcessor` (constant-time compare; the token is consumed
so handlers never see it). Tests cover: valid token accepted; missing token and wrong token
UNAUTHENTICATED; plaintext client UNAVAILABLE; a client trusting a different CA refuses the
certificate (the token is never sent); the coordinator over TLS works and with a wrong token
surfaces UNAUTHENTICATED. Not done: mutual TLS (client certificates) and token rotation without
a restart.

**InsertBatch partial failure, now explicit.** Validation failures are the call's status and
insert nothing. Otherwise the call returns OK and `ids` lists every input's id or 4294967295,
with `error_code`/`error_message` for the first failure. A retry sends `ids` back: inputs with an
id are verified against the stored vector (bitwise; a since-deleted id is accepted and not
resurrected), all before inserting anything on that shard, and only the rest are inserted. The
retry is idempotent and repeatable. A shard stopped mid-batch, restarted, and retried until
complete ends with exactly the batch's vectors (test). Remaining gaps, documented in the proto and
README: a lost reply (shard committed, reply timed out) still makes a retry insert duplicates,
which needs a request key in the WAL to close; and two concurrent retries of one batch both insert
the missing inputs.

**32-bit ids.** Global ids share 32 bits across shards (each shard holds about 4.29 billion / N),
and 4294967295 is reserved. Before, `to_global` would silently wrap. Now the coordinator passes
each shard `id_limit = codec.local_limit(shard)`, and the shard refuses an insert that would reach
it with RESOURCE_EXHAUSTED. A concurrent insert that slips past the pre-check is undone with a
tombstone. Responses with out-of-range ids are rejected as INTERNAL.

**Also:** Python tests re-run after the Homebrew openssl/c-ares upgrade: 142 passed (the live API
test ran, the usage cap having reset). New `server-release` preset for the benchmark.

**Measured: async fan-out vs. thread per shard** (`results/server/coordinator_latency_sift1m-200k-q1000.md`,
200k SIFT vectors, HNSW shards, ef_search 64, M2 Mac: development, indicative; 3 runs x 5 rounds,
modes alternated within each round; recall@10 identical in both modes):

- Sequential median latency: 299 -> 275 us (-8%) with 2 shards, 362 -> 327 us (-10%) with 4
  (async better in 14/15 and 15/15 rounds). That matches the cost of starting 2 or 4 threads per
  query that async no longer pays.
- Concurrent, 8 clients: QPS +17% (2 shards) and +23% (4 shards); median latency -16% and -21%
  (async better in 14/15 and 15/15 rounds). Under load, thread creation competes with the searches,
  so the gain grows with the shard count, as expected.
- Tails are noisy here. With 4 shards the concurrent p99 *mean* is worse for async (+57%), but
  async had the lower p99 in 12/15 rounds. The mean comes from two rounds at 26.6 ms and 16.0 ms;
  thread per shard had one at 11.1 ms. That pattern looks like scheduling noise: 8 clients + 4 shard
  servers + 2 coordinators on 8 cores, 4 of them efficiency cores. I added an "async better in N of
  M rounds" column (generated) so the table shows this instead of a hand note. Tail latency needs
  re-measuring on dedicated hardware, with shards on separate machines (Phase 9).

## 2026-10-01: AWS session prepared (nothing created)

**Read from the account (all read-only, free):**
- On-demand prices from the Pricing API; c7i.8xlarge is $1.428/h in both us-east-1 and us-east-2.
- Seven days of spot history: c7i.8xlarge is $0.34–0.46/h in us-east-2, $0.49–0.57/h in
  us-east-1.
- **vCPU quota of 32** (on-demand and spot) in both regions. That decided the shape of the
  session: one c7i.8xlarge uses all 32, so the cluster (28 vCPU) runs as a second stage after
  teardown. `up.sh` checks the quota and refuses a stage that would exceed it.

**Plan:** `docs/aws-plan.md`. About $20 on-demand (about $6 on spot, not recommended: an interruption
mid-10M-build costs more than it saves).

**Infrastructure:** `aws/`. Terraform for a dedicated VPC, SSH from the operator's IP only, the
instances per stage, and 14 h auto-termination. Scripts to set up, run each part in tmux, collect,
tear down, and verify nothing remains.
- `terraform validate` and `plan` for all three stages ran against the account with scratch
  state; no apply.
- The plan caught a wrong Canonical owner ID in the AMI lookup (my typo).

**Bench changes so the session can run:**
- `--build-threads` for hnswlib, FAISS, and the curve runner (10M single-threaded builds would take
  hours per library).
- `--name` on the curve runner, so AWS tables don't overwrite the Mac ones.
- BIGANN-10M in `prepare_datasets.py`, via an HTTP range request into the 1B file. The published
  ground truth is spot-checked against exact search; the check was tested on a 20k-row sample,
  including that it catches a corrupted distance.
- `--snapshot` on the search harness, so perf and thread scaling measure search, not build.
- `bench/run_search_scaling.py`, which pins each run to N distinct physical cores.
- EC2 instance type, topology, and kernel in every record (IMDSv2).
- A `linux-profile` preset.
- `strata_shard --sync none` for bulk loads.
- `strata_load`, a standalone load client, with `bench/run_sharding_bench.py`.

**Found by the local dry run** (`LOCAL=1 aws/scripts/run_sharding.sh`, the whole cluster flow on
the Mac):
- Closed-loop throughput was computed from a start time 100 ms after the workers had started;
  short runs gave negative QPS.
- The first open-loop design (256 sleeping worker threads) measured the load generator: workers
  woke up to 5 ms late. Replaced by one dispatcher thread that issues async calls on schedule
  (sleep, then spin the last millisecond). Its lateness is now reported per row and warned about
  above 1 ms.

**Noticed:** the GitHub repo is already public, so instances clone it without credentials;
`buildplan.md` still listed that as open.

## 2026-10-02: Before the AWS session: secrets audit, ignore rules, IAM policy, SIMD labels

**Decisions recorded:** no bare-metal perf rerun, on-demand instances, no quota increase.

**Secrets audit of the public repo** (the full history: 94 commits on every branch; the remote has
only `main` and `linux-arm-check`, no other refs):
- gitleaks 8.30.1 (checksum-verified, run from the scratchpad, not installed): **no leaks**. It
  scans 93 commits; the 94th is a merge, whose changes are already in the commits it merged.
- Direct searches of every diff in history for AWS key IDs (AKIA/ASIA),
  `aws_secret_access_key`, Anthropic keys (`sk-ant-`), GitHub and Slack tokens, HuggingFace
  tokens, private-key blocks, Kaggle keys, and `ANTHROPIC_API_KEY=` with a value: **0 matches**.
- Every path ever committed was checked for `.env`, `.claude/`, settings files, keys, `.aws/`,
  tfstate, tfvars, `kaggle.json`, `.netrc`. The only hit is `.env.example`, whose only version has
  an empty `ANTHROPIC_API_KEY=`.
- `results/kaggle/environment.json` holds hardware, CUDA details, and three allowlisted
  `KAGGLE_*` values (run type, image digest, URL base): nothing secret.
- Commit messages: no matches.

**Ignore rules:**
- Terraform state, plans, tfvars, crash logs, overrides, CLI config, `.aws/`, private keys
  (`*.pem`, `*.key`, ...), `kaggle.json`, `.netrc`, and Claude Code local settings are now
  ignored everywhere in the repo, not just under `aws/terraform/`.
- `git check-ignore` confirmed 19 sample paths are ignored and the provider lock file,
  `.env.example`, and `*.tfvars.example` are not. No tracked file matches an ignore rule.

**Credentials can't flow into the repo from aws/:**
- The instances get no IAM role, so no AWS credentials exist on them.
- Keys, the cluster token, and state stay in ignored directories.
- `collect.sh` now scans everything copied back for private keys, AWS/Anthropic/GitHub token
  formats, IMDS tokens, and the session's cluster token. Any hit is moved to `aws/.quarantine/`
  (ignored) and the script fails. It was tested on the real `results/` (no false positives) and on
  planted secrets (both caught, the clean file left alone).

**IAM:** `aws/iam/strata-terraform-policy.json`, least privilege for Terraform and the scripts.
- Describe calls only in us-east-2.
- Creates only with `Project=strata-bench` at creation.
- Changes and deletes only on tagged resources.
- Launches only c7i.8xlarge/2xlarge/xlarge, on-demand, IMDSv2, Canonical images, gp3 ≤ 150 GB.
- The `Project` tag can't be removed.

Access Analyzer reports no findings, and `aws/iam/simulate_policy.sh` runs 37 allow and deny cases
through the IAM simulator, all as expected. Whether every provider call is listed can only be
confirmed by running as the user (plan first).

**SIMD labels:** c7i CPUs have AVX-512, and the three libraries won't use the same width there.
- Strata: AVX2 + FMA, compile time.
- FAISS 1.15.1: one wheel with runtime dispatch, reported by `faiss.SIMDConfig.get_level_name()`.
  I first assumed separate per-level builds; inspecting the wheel showed otherwise.
- hnswlib 0.8.0: sdist only, so it compiles on the instance with `-march=native` and dispatches
  to AVX-512.

`bench/simd_info.py` records what each one actually ran: its kernel or dispatched level, plus a
disassembly count of 512-bit and 256-bit register instructions, checked here on x86 binaries
(Strata's x86 build: 0 AVX-512 instructions; the FAISS wheel's libfaiss: AVX-512 present). The
comparison table now has a SIMD column and warns when widths differ.

## 2026-10-02: x86 comparison at AVX2 (primary) and AVX-512 (second); no AVX-512 kernel yet

**Decision:** no AVX-512 Strata kernel for now (noted as future work in `buildplan.md` and
`docs/design.md`). The x86 comparison runs twice:
- **Primary, like for like:** all three libraries at AVX2.
- **Second, labeled not like for like:** hnswlib and FAISS at AVX-512, against the same Strata runs.

**Holding FAISS to AVX2:**
- The optimization-level variable `FAISS_OPT_LEVEL` turned out not to work for faiss-cpu 1.15.1.
  It chooses among separate per-level builds, which that wheel no longer ships; its single build
  dispatches at runtime and would still pick AVX-512.
- What works is `FAISS_SIMD_LEVEL=AVX2` before import plus `SIMDConfig.set_level(AVX2)` after.
- Tested on the Mac: an unsupported level in `FAISS_SIMD_LEVEL` silently becomes NONE (scalar),
  and an empty value aborts the process, while `set_level` raises. So the level is always verified
  after it's set.

**Holding hnswlib to AVX2:** its runtime dispatch picks AVX-512 whenever it is compiled in, so the
like-for-like runs use a second env, `.venv-avx2`. There hnswlib is built from source with
`HNSWLIB_NO_NATIVE=1` and `-mavx2 -mfma` (Strata's flags), with uv's cache bypassed so the
`-march=native` wheel can't be reused.

**Verification (`bench/simd_info.py`):**
- `verify()` checks a library against the requested configuration:
  - AVX2: FAISS reports and dispatches AVX2; hnswlib has 256-bit code and zero AVX-512
    instructions.
  - Native: FAISS runs its auto-detected level; hnswlib has an AVX-512 path on an AVX-512 CPU.
- `verify_strata()` requires Strata's AVX2 kernels and no AVX-512 code on x86.
- Every run refuses to start if its check fails, and setup checks both configurations before any
  benchmark.
- On the Mac, the AVX2 requests are refused cleanly and the native checks pass. The hnswlib AVX2
  build itself can only be exercised on x86, where setup verifies it.

**Cost and schedule:** the second comparison reruns hnswlib and FAISS in parts A and B. The main
stage is now ~14.4 h with contingency (~$20.5) and the session about $24. The auto-termination
safety net went from 14 h to 18 h; at 14 h it would have cut the main stage off.

## 2026-10-02: First apply refused at RunInstances: the AMI condition was wrong

**What happened:** the main-stage apply created the network (VPC, subnet, internet gateway,
route table and association, security group, key pair: 7 resources, none billed), then failed
with UnauthorizedOperation for `ec2:RunInstances` on the Canonical AMI.

**Diagnosis:**
- An EC2 `--dry-run` of the same launch as the strata-terraform user reproduced the refusal
  without launching anything.
- `sts decode-authorization-message`, run with an admin profile, showed the context AWS evaluated
  for the image: `ec2:Owner = "amazon"`, with the owner account 099720109477 present only as the
  resource's account.
- For a verified provider's public AMI, `ec2:Owner` is the alias `amazon`, so my condition
  `ec2:Owner = 099720109477` could never match.
- My simulator case had passed because I supplied that wrong value as the context. The simulator
  evaluates whatever context it is given; it can't detect a wrong model of the context.

**Fix:** `ec2:Owner = amazon` AND `aws:ResourceAccount = 099720109477`. The first alone would
allow any Amazon or verified-provider image (Amazon Linux, Windows, ...); the second narrows it to
Canonical.
- Access Analyzer reports no findings.
- `simulate_policy.sh` now uses the real context: 39 of 39 pass, including that Amazon Linux stays
  denied.

**New check:** `aws/iam/dryrun_launch.sh` runs Terraform's exact launch as `--dry-run` with the
strata profile and decodes any refusal. It caught nothing new yet because the fix isn't on the
IAM user yet. Running it after updating the policy is the only proof that the remaining launch
conditions (instance type, market, IMDSv2, volume, tags) hold for real requests. The image was
simply the first resource AWS evaluated.

**Leftovers:** the 7 network resources are in Terraform's state and free. The rerun reuses them,
and teardown removes them. No cleanup is needed before rerunning.

## 2026-10-02: Second apply: launched, then refused ModifyInstanceAttribute; whole-session dry run

**What happened:** with the AMI fix on the user, the apply launched i-0d5e7907f0932b3a0
(c7i.8xlarge), then failed: `ec2:ModifyInstanceAttribute` was not allowed. Terraform marked the
instance tainted.

**Why:** in the pinned AWS provider (v5.100.0, `internal/service/ec2/ec2_instance.go`), create
runs the update function on the new instance. Its `instance_initiated_shutdown_behavior` block has
no `IsNewResource` guard, so it calls `ModifyInstanceAttribute` even though RunInstances had
already set `terminate` (the live instance confirms it). Reading the code also showed a worse gap:
**delete always calls `ModifyInstanceAttribute(DisableApiTermination=false)` before terminating**.
So `terraform destroy`, and with it `teardown.sh`, would also have failed, leaving a billing
instance Terraform couldn't remove.

**Fix:** decoded dry runs against the live instance showed AWS populates
`ec2:Attribute/<Name>` with the requested value. Two statements allow `ModifyInstanceAttribute` on
tagged instances only for shutdown behavior = `terminate` and termination protection = `false`.
Stop, termination protection on, instance type, and user data stay denied.
- Access Analyzer reports no findings.
- `simulate_policy.sh` passes 46 of 46.
- The policy is 5,595 of 6,144 bytes.

**`dryrun_launch.sh` now checks the whole session** through `--dry-run`, against real resources:
launches, post-launch calls, all reads, the whole teardown, and calls that must be refused. The
call list comes from the provider source. Run under the policy still attached to the user, it
gave 54 as expected and exactly the 2 problems the new statements fix, one on the create path
and one on the teardown path. The fix itself is proven once the user's policy is updated and the
script rerun while the instance still exists.

## 2026-10-02: AWS setup, second rerun: Python headers missing for hnswlib

The server, coordinator benchmark, and load client compiled with GCC 13 on the first try. Then
`uv pip install` failed building hnswlib 0.8.0 (no wheel, so it compiles on the machine):
`Python.h: No such file or directory`. The venv uses Ubuntu's system Python 3.12, whose headers
are in `python3-dev`, which setup didn't install. Fixed by installing it on main and client.

Testing the fix on the instance found the next failure before it happened: uv 0.12 refuses to
create a venv over an existing one (`A virtual environment already exists`), and the failed run
left `.venv` behind. Setup now recreates both venvs each run.

Both hnswlib builds were checked on the c7i with `bench/simd_info.py`:
- native (`-march=native`): AVX-512, 120 zmm instructions;
- AVX2-only (`HNSWLIB_NO_NATIVE=1`, `-mavx2 -mfma`): 0 zmm, 552 ymm.

Both pass `verify()`.

## 2026-10-03: AWS comparison tables mixed in Mac records

Parts A–C finished on the c7i, but their comparison tables were wrong in a way that would have
been easy to publish. Records are kept per machine, the committed Mac records were on the instance
too, and `run_hnsw_curves.py` selected records by library and parameters only. So the x86 tables
also listed Strata's NEON runs and the Mac's hnswlib and FAISS runs, and their **Hardware line read
"Apple M2 (development machine)"**, taken from the first record.

Fixed with `--machine`: by default only records from the machine running the report, or those
whose instance type and CPU contain a given string (e.g. `c7i.8xlarge`). The four AWS tables were
regenerated on the laptop from the collected records. `collect.sh` now regenerates them after
every main-stage collect, because the instance still writes the old versions (it runs the pinned
commit). The raw records were never affected.

## 2026-10-03: AWS main stage complete; results reviewed before committing

All six steps succeeded on the c7i.8xlarge (setup 22:53 UTC, perf 11:13 UTC). Nobody ran teardown
right away, so the instance idled about 5 h until its 18 h self-termination (about $7 extra; the
main stage cost about $26 against the planned $20.5). Teardown then removed the network.
`check_clean.sh` and a sweep of every region found no instances, volumes, or Elastic IPs.

**Review before committing:**
- **Comparison tables (A, B):** already fixed to one machine (`--machine`). Recall agrees across
  libraries, run-to-run stdev is ~1% or less, and Strata's brute force has recall 1.0.
- **Scaling (C):** search 14.4x at 16 cores (90%), 17.2x at 32 SMT threads. Build 15.3x at 16
  threads with identical recall and graph statistics.
- **Filter (D):** two report bugs fixed and both reports regenerated from the raw data. The Mac
  200k report regenerates byte-identical.
  - At 10M with random filters, the graph beat the pre-filter at every selectivity measured, but
    the crossover table printed the smallest measured point (0.100%) as if it were measured. It
    now says "< 0.100% (graph faster at every selectivity measured)", and the opposite case gets
    ">".
  - `--report-only` printed the query count from its own default (500) instead of the run's
    (200 at 10M). Reports now use the settings recorded in `meta.json`.
- **Perf (E):** hardware cycles were available (no `cpu-clock` fallback).
  - `perf stat` shows `cache-misses` as 0 and `LLC-load-misses` as unsupported. The 0 means
    "not counted on this VM", not zero misses; never quote it.
  - IPC 0.54 (1M) and 0.58 (10M), memory-bound.
  - The profile includes loading the index snapshot: about 4% of samples at 1M (`crc32c`, page
    faults) and about 24% at 10M (`crc32c` 16%, page faults, `from_snapshot`, kernel copies).
    Within search, `search_layer` (graph traversal, stalled on the prefetched loads) takes 54% and
    the AVX2 distance kernel 33% (1M). Next time, start recording after the load.
- gitleaks over `results/`: no leaks. The collect guard quarantined nothing.

**Headlines (like for like, all 256-bit, one thread, same M/ef_construction):**
- **SIFT1M:** recall equal. hnswlib about 6% faster than Strata (ef=80: 5,392 vs 5,067 QPS), FAISS
  19% slower (4,107). Strata builds fastest (333 s vs 364 s and 493 s).
- **GloVe-100:** Strata fastest (ef=80: 5,040 vs 4,643 and 4,394 QPS).
- **BIGANN-10M:** hnswlib about 7% faster than Strata (ef=80: 4,035 vs 3,767). FAISS reaches
  slightly higher recall per ef but about 43% lower QPS.
- **AVX-512 barely changes the references.** FAISS gains about 0–6%. hnswlib's `-march=native`
  build is slower than its AVX build on SIFT1M (4,888 vs 5,392) and at 10M (6,911 vs 7,032 at
  ef=40), and faster only on GloVe-100 (5,185 vs 4,643). With IPC ~0.55, search is bound by
  memory latency, not vector width. So the missing AVX-512 kernel is not why Strata trails
  hnswlib.

## 2026-10-03: Before the cluster stage: run_sharding.sh used bash 4's mapfile

`run_sharding.sh` read the shard addresses with `mapfile`, which the Mac's bash 3.2 doesn't have.
It would have failed right after `up.sh cluster --apply`, leaving 6 instances billing idle. The
`LOCAL=1` smoke test never reaches that branch. It now uses a bash 3.2 read loop (tested under
/bin/bash), which also fails loudly if the public and private address lists don't match. The server
binaries saved from the main stage (built at 8ae9bf4) still match the code: no changes under
`server/`, `proto/`, `src/`, or `include/` since then.

## 2026-10-03: Cluster stage: old coordinators were never stopped

The 2-shard run failed with TLS "unable to get local issuer certificate". The cause:
- `run_sharding.sh` stopped servers with `pkill -x strata_coordinator`, but `pkill -x` matches
  the process name, which Linux truncates to 15 characters (`strata_coordina`). Coordinators
  were therefore never stopped.
- gRPC binds with SO_REUSEPORT, so each new coordinator started on the same port, and the kernel
  spread connections between old and new ones.
- The new run's certificates (new CA) didn't match the old coordinator's, hence the handshake
  failure.

Earlier, when my tool's time limit killed the first orchestrator, its "2-shard" load (339 s, no
faster than 1 shard) had gone through the still-running 1-shard coordinator, so every vector
landed on shard 0. That run was discarded anyway.
- **The 1-shard results are valid:** no coordinator was running before them.
- **The fix:** `stop_all` kills by command line (`pkill -f "[b]in/strata_(shard|coordinator)"`),
  waits, escalates to SIGKILL, and fails unless nothing remains. Tested on a cluster node: the
  old match found 0 of a running dummy `strata_coordinator`; the new one stops both dummies.
- **Follow-up:** the servers should disable SO_REUSEPORT (`GRPC_ARG_ALLOW_REUSEPORT=0`) so a
  second server on a port fails to start instead of silently sharing it. That needs a rebuild,
  so not during this session.
- **Operations:** the remaining shard counts run from a detached session
  (`start_new_session`), because my tool's background time limit had killed the orchestrator.

## 2026-10-04: Cluster stage (part F) results

4 × c7i.xlarge shards, a c7i.xlarge coordinator, and a c7i.2xlarge client, in one placement group,
TLS and token on every hop, SIFT1M, ef_search=64. Results: `results/server/sharding_aws.md`.
- **Ingest scales:** 1M vectors through the coordinator in 345 s (1 shard), 174 s (2), and 72 s
  (4). The 4-shard load is faster than linear because each shard's graph is smaller.
- **Query capacity barely scales:** 8,035, 8,605, and 9,945 QPS (1.07x and 1.24x). Every query
  visits every shard, and an HNSW search over half the data costs nearly as much as over all of
  it. The coordinator (2 cores, TLS to the client plus one call per shard) or the single load
  client may also cap it. The planned diagnostic (CPU on client, coordinator, and shards during
  a long saturating load) didn't happen: my restart command broke on zsh array indexing as the
  session's usage ran out, and the cluster was torn down first so it would stop billing. The
  cause is **open**. Re-measure with the diagnostic (and with replicas instead of partitions)
  on the Oracle machine or a later session.
- **Latency:** about 2.5 ms per query one at a time, at every shard count; p99 1.4 ms at 50% of
  capacity and 1.7–2.3 ms at 25% and 75%. The open-loop client sent at most 3 µs late (p99), so
  these latencies are the servers'.
- **Recall rises with shard count** at a fixed ef_search (0.964, 0.979, 0.989), because each shard
  returns its own top-k. That is more work per query, and part of why capacity doesn't grow.
- Teardown: 14 resources destroyed, `check_clean.sh` clean.

## 2026-10-04: Build plan and design doc brought up to date

- **Build plan:**
  - Ticked items that were done but never checked off, each verified in the repo: parallel
    build, HNSW snapshots and deletes, the gRPC API, sharding and the coordinator, and the repo
    being public.
  - Filled the metrics tracker with the measured results.
  - Added two follow-ups: diagnose sharded query capacity, and a size-aware filter threshold.
- **Design doc:** x86 results replace the development numbers wherever they exist; *M2* labels
  mark the rest. New content:
  - the like-for-like comparison;
  - the memory-bound profile (search_layer 54% of cycles, AVX2 distance 33%, ~0.55 IPC), which
    explains why AVX-512 barely helps;
  - build and search scaling;
  - the crossover at 200k, 1M, and 10M;
  - the cross-machine sharding results.
- **New weakness found while writing it:** the auto-filter threshold is fixed at 1.3%, but at 10M
  the graph beats the pre-filter even at 0.1% for random filters (≥443 vs 26 QPS at 1%). With the
  default, auto would take the ~17x slower path there. The benchmark derived its own threshold
  (0.89%) and is unaffected.

## 2026-10-04: Size-aware auto-filter threshold

`FilteredSearchOptions::prefilter_below` is now optional. Unset, `search_filtered` uses
`default_prefilter_below(size())` = 1.48% × (n / 1M)^-0.22, clamped to [0.05%, 5%]: the power
law through the largest crossovers measured on x86 (1.48% at 1M, 0.89% at 10M). It is exposed in
Python as `strata.default_prefilter_below`. The bench's `--threshold` is optional too (the driver
always passes it).
- **Why:** the fixed 1.3% sent 10M-vector filters between 0.89% and 1.3% to the pre-filter, where
  the graph is faster.
- **A correction to the design doc:** I had written "~17x slower". That came from the crossover
  table's recall-0.95 operating point, which uses very low ef. At ef 40 the measured gap is about
  5x for random filters and 1.4x for correlated ones.
- **Remaining limit, documented in the explainer:** the threshold follows the correlated
  crossover, so at 10M random filters between ~0.2% and 0.9% still take the pre-filter, which is
  2–5x slower there, never less exact.
- **Tests:** 254/254 C++ (2 new), 142 Python. HNSW explainer section 11 updated in the same
  commit, per CLAUDE.md.

## 2026-10-04: GCC warnings, servers refuse a shared port, resume bullets

- **GCC warnings: 63 → 0** in Strata's sources (GCC 14 `-fsyntax-only` with the project's warning
  flags, run on the Mac; the Linux build on the Oracle machine is the final check, including
  `server/`).
  - `-Wmissing-field-initializers` (52): GCC, unlike clang, warns when a designated initializer
    skips a member without a default initializer. Fixed at the source with `{}` initializers on
    7 members (`FilteredSearchOptions`, `Filter::Node`, `CompiledFilter::Op`).
  - `-Wsign-conversion` (7): `count += cond ? 1 : 0` into a `size_t`; now unsigned literals.
  - `-Wcomment` (4): `\` at the end of `//` usage examples, which GCC reads as a line
    continuation.
- **Servers no longer share a port:** `start_server` sets `GRPC_ARG_ALLOW_REUSEPORT=0`, so a stale
  coordinator can't silently share its port with a new one (the cluster-stage bug). The new test
  `SecondServerOnTheSamePortFailsToStart` fails without the fix ("a second server shared port")
  and passes with it.
- **`docs/resume-bullets.md`:** short and long versions, every number linked to its results
  table. Checking them caught an inverted comparison: Strata is 15–77% faster than FAISS (FAISS
  answers 13–43% fewer QPS), not "13–43% faster". Sharded *query* capacity is deliberately not
  claimed.
- Tests: 254 core, 282 server, 142 Python.

## 2026-10-05: Oracle ARM session: Linux ARM check, final ARM numbers, Docker Compose

Oracle Cloud `VM.Standard.A1.Flex`, 4 Arm Neoverse-N1 cores, 24 GB, Ubuntu 22.04, shared with
Vigil.

**Linux ARM check:** `linux-release`, `linux-asan`, and `linux-tsan` each pass 254/254, and
`linux-server-release` passes 282/282, all with GCC 13. The first ARM server build found 7 GCC-only
warnings in `server/`; they were fixed (`98dd664`), and the rebuild has 0 warnings.

**Benchmarks, with Vigil frozen** (`docker pause` of its 6 containers, unpaused on exit, plus a 3 h
watchdog; your choice):
- **SIFT1M, single thread, ef_search 80:** Strata 4,455 QPS @ 0.975, FAISS 4,034 @ 0.978 (both
  NEON), hnswlib 1,971 (no NEON path, scalar). Builds: 431 s, 584 s, and 1,065 s.
- **Scaling on 4 cores:** search 3.4x (85%), build 3.3x, with the same recall.

**Bugs found on the way:**
- **The scaling runners silently reused another machine's runs.** They resume by skipping run
  files that exist, and the AWS runs committed for `sift1m` sat at the same paths, so part C
  "finished" in seconds. `--label` now keeps each machine's runs apart. Part C was rerun in a
  second quiet window.
- **The build-scaling report printed the invocation's cool-down,** not the recorded run's (the
  same bug as the filter report).
- **ARM CPUs were recorded as just `aarch64`:** `/proc/cpuinfo` has no model name there. The
  recorder now asks `lscpu` and Oracle's metadata service (shape, OCPUs, memory, availability
  domain). The ARM records already written keep their bare label; the report falls back to their
  recorded core count. Raw records are not edited after the fact.

**Docker Compose (`deploy/docker/`):** 3 shards and a coordinator, TLS and a token on every hop,
data in named volumes, the coordinator published on loopback only. Tested end to end on the ARM
machine:
- 10k vectors inserted through the coordinator, recall 1.0;
- a client without the token is refused;
- a restarted shard recovers from its volume, and recall is still 1.0.

Three design corrections came out of testing:
1. **The base image is Ubuntu 24.04.** GCC 13 binaries need `GLIBCXX_3.4.32`, which stock 22.04
   doesn't have.
2. **Containers run as the host user** (`STRATA_UID`/`STRATA_GID`) instead of a fixed uid, so the
   mounted key and token stay mode 600 and are still readable by a client on the host (an ACL
   approach needed `setfacl`, which isn't installed by default).
3. **Command flags are written as separate arguments,** because the servers don't parse
   `--flag=value`.

**Ops:**
- The Mac lost its Tailscale connection to the machine once, mid-build; the build itself finished
  unaffected.
- My `pkill -f "sleep 10800"` matched its own SSH command line. Use the `[s]leep` bracket form.

## 2026-10-06: Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) builds and tests on every push: `release` on Linux
x86, Linux ARM, and macOS ARM; `asan` on Linux x86 and macOS ARM; `tsan` on Linux x86; the
Python tests (including the bit-for-bit comparison with the C++ build on SIFT10K) and ruff; and
the gRPC server tests. vcpkg is pinned to the manifest's baseline and its binary cache is kept
between runs (the first gRPC build took 59 minutes).

What the first run found:
- **`std::jthread` doesn't exist in Xcode 16's libc++** (GitHub's macos-15 runner). My Mac's
  newer toolchain hid it, and the README claimed Apple clang 15+. The thread pool and two tests now use
  `std::thread` with explicit joins; the README states the toolchains CI actually tests.
- **GCC's ASan on the GitHub ARM runner is ~80x slower than on x86** for compute-heavy tests
  (KMeans 35 s vs 0.45 s; one PQ test took 72 minutes), while the same tests under clang's ASan on
  the Mac ARM are normal speed. The cause was not isolated. CI drops `linux-asan` on ARM: the
  NEON kernels still run under ASan on macOS ARM, and `linux-asan` on ARM passed on the Oracle
  machine.
