# Strata

Distributed vector search engine written from scratch in C++20 (HNSW, SIMD, product quantization,
filtered search, sharding), with a retrieval-augmented generation layer on top.

> Status: Phases 0–6 done except the hand-written HNSW core and the parts that build on it. See `docs/devlog.md`. See [`buildplan.md`](buildplan.md) for the roadmap.

## Build

Requirements: CMake ≥ 3.25, Ninja, a C++20 compiler (Apple clang 15+ or GCC 12+/Clang 16+), and
[vcpkg](https://github.com/microsoft/vcpkg). Dependencies (GoogleTest, Google Benchmark) are pinned
in `vcpkg.json` and installed automatically on first configure.

```sh
# one-time vcpkg setup
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=~/vcpkg   # add to your shell profile

# configure, build, test
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

| Preset    | Build type | Notes                                                        |
|-----------|------------|--------------------------------------------------------------|
| `debug`   | Debug      | Day-to-day development                                       |
| `release` | Release    | Benchmarks                                                   |
| `asan`    | Debug      | AddressSanitizer + UndefinedBehaviorSanitizer; run before committing memory-handling changes |
| `tsan`    | Debug      | ThreadSanitizer; run before committing concurrency changes   |
| `rosetta-avx2` | Debug | macOS only: x86_64 + AVX2 build run under Rosetta 2, to test AVX2 kernels on a Mac (correctness only, never timing) |

Build output goes to `build/<preset>/`.

## Datasets

Python tooling uses [uv](https://docs.astral.sh/uv/) with Python 3.11.

```sh
uv sync
uv run python scripts/prepare_datasets.py siftsmall       # SIFT10K, ~5 MB
uv run python scripts/prepare_datasets.py sift1m glove100 # ~500 MB and ~460 MB downloads
uv run pytest
```

Datasets are written to `data/<name>/` (gitignored) as `base.fbin`, `query.fbin`,
`groundtruth.ibin`, and `meta.json`. The binary format is a little-endian `uint32` header
`(num_vectors, dimension)` followed by row-major `float32` or `int32` values.

## Benchmarks

Every number comes from a script that saves raw JSON (with commit, hardware, and parameters) under
`results/`, and tables are generated from those files.

```sh
cmake --build --preset release
uv run python bench/run_search_bench.py --dataset siftsmall --index brute_force
uv run python bench/run_search_bench.py --dataset siftsmall --index hnsw --M 16 --ef-construction 200
uv run python bench/run_reference_bench.py --dataset siftsmall --library hnswlib
uv run python bench/run_reference_bench.py --dataset siftsmall --library faiss --index hnsw
uv run python bench/run_reference_bench.py --dataset siftsmall --library faiss --index flat
uv run python bench/run_search_bench.py --dataset siftsmall --kernel scalar   # without SIMD
uv run python bench/run_search_bench.py --dataset siftsmall --threads 4        # throughput mode
uv run python bench/run_search_bench.py --dataset siftsmall --index pq --pq-m 16   # rerank sweep
uv run python bench/run_storage_bench.py --dataset siftsmall   # WAL, checkpoint, recovery
uv run python bench/run_filter_bench.py --dataset siftsmall    # filtered search by selectivity
uv run python bench/run_micro_bench.py --repetitions 5
uv run python bench/make_tables.py        # writes results/tables.md
uv run python bench/plot_recall_qps.py    # writes results/plots/recall_qps_<dataset>.png
uv run python bench/plot_pq_memory.py     # writes results/plots/pq_memory_<dataset>.png
uv run python bench/plot_filter.py        # writes results/plots/filter_<dataset>.png
```

HNSW is compiled in only when `src/index/hnsw.cpp` exists; until then `--index hnsw` is
unavailable and `tests/hnsw_test.cpp` (the HNSW spec) is not built.
