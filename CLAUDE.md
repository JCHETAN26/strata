# Strata

Distributed vector search engine written from scratch in C++20, with a RAG layer on top.
Given millions of vectors, Strata finds the most similar ones in milliseconds (HNSW), and uses
that to answer questions from documents with cited sources.

This is a portfolio project. Correctness, clarity, tests, and measured results matter more than
feature count. Every performance claim must be backed by a reproducible benchmark.

The full phase-by-phase plan lives in `buildplan.md`. Check it before starting new work.

## Ground rule: the core algorithm is mine

I (the developer) write or deeply review the core HNSW logic myself, because I must be able to
explain every line in interviews. Specifically, for `src/index/hnsw*`:

- Do NOT write the core insertion, neighbor-selection heuristic, or search code unless I explicitly ask.
- DO explain the algorithm, review my code, point out bugs, suggest improvements, and write tests.
- When I ask you to implement part of it, explain each design choice so I understand it.

Everything else (build system, harnesses, bindings, scripts, server, tests) you can build normally.

## Repository layout

```
strata/
├── CMakeLists.txt, CMakePresets.json, vcpkg.json
├── include/strata/     # public headers
├── src/
│   ├── distance/       # scalar, NEON, AVX2 distance kernels + dispatch
│   ├── index/          # brute force, HNSW
│   ├── pq/             # product quantization
│   ├── storage/        # persistence, write-ahead log
│   └── filter/         # metadata filtering
├── server/             # gRPC service, shard coordinator
├── python/             # Python bindings (nanobind or pybind11)
├── rag/                # Python RAG layer: BM25 fusion, reranking, answer generation
├── bench/              # C++ benchmarks + Python scripts that produce results tables/charts
├── tests/              # GoogleTest unit and integration tests
├── scripts/            # dataset download and conversion
├── data/               # datasets (gitignored)
├── results/            # raw benchmark output and generated charts
└── docs/               # design doc, dev log
```

## Machines

- **Development (now):** MacBook Air M2, 8 GB RAM, ARM, macOS. Fanless: it throttles under
  sustained load. Keep dev datasets small (SIFT10K; SIFT1M only when needed). Never run long
  benchmarks here except the final ARM results run.
- **Results and server work (later):** Ubuntu 22.04, Ryzen 7 5800H (8C/16T, AVX2),
  ~19 GiB usable RAM (mixed 8+16 GB, partly single-channel), RTX 3050 4 GB. Docker, gRPC shards,
  embeddings, and all final benchmarks run there.
- Code must build and pass tests on both macOS/ARM and Linux/x86.

## Build

- C++20, CMake + Ninja, dependencies via vcpkg manifest (`vcpkg.json`) so versions match on both OSes.
- Presets: `debug`, `release`, `asan` (AddressSanitizer + UndefinedBehaviorSanitizer).
- Run tests under the `asan` preset before committing any change to memory-handling code.
- Keep `CMakePresets.json` and the README's build instructions in sync.

## C++ conventions

- Modern C++20: RAII, no raw `new`/`delete`, `std::span` for views over vector data.
- Performance-critical paths (distance kernels, graph traversal) favor flat contiguous memory and
  simple loops over abstraction. Don't add indirection to hot paths without a benchmark showing it's free.
- No exceptions across hot paths; use clear error returns (e.g., `std::expected`-style) at API boundaries.
- Thread safety must be documented on every public class.
- Format with clang-format; keep clang-tidy clean.

## SIMD

- The scalar kernel is the reference implementation. Every SIMD kernel (NEON, AVX2) must have a
  test asserting results match scalar within a documented floating-point tolerance.
- Select kernels at compile time by architecture, with the scalar fallback always available.

## Testing

- GoogleTest for unit tests. Every component has tests, including edge cases
  (empty index, single vector, duplicate vectors, dimension mismatches).
- Search quality is tested by recall against brute-force ground truth, not by exact result lists.
- Crash-recovery tests for the WAL: kill mid-write, restart, verify no acknowledged write is lost.

## Benchmarks and results

- Every result comes from a script in `bench/` that saves raw data to `results/` and generates the
  table or chart. No hand-copied numbers.
- Always compare against a baseline: brute force, hnswlib, FAISS, scalar vs. SIMD, with vs. without.
- Follow ann-benchmarks methodology: report recall@10 vs. QPS curves, not single points.
- Record hardware, dataset, parameters, and commit hash with every result.
- Report averages over multiple runs with variance.
- Report weaknesses honestly (e.g., where Strata trails FAISS).

## Datasets

- SIFT10K, SIFT1M, GloVe-100 from ann-benchmarks (HDF5), converted to raw binary by
  `scripts/` so the C++ code has no HDF5 dependency. Stored in `data/` (gitignored).
- BEIR (SciFact, FiQA, NFCorpus) and a HotpotQA subset for the RAG layer.
- Never commit datasets or large generated files.

## Python (bench, bindings, rag)

- Type hints, pytest, ruff.
- Answer generation uses the Anthropic API with `claude-haiku-4-5`. API keys live in `.env`
  (gitignored). Never commit secrets.

## Workflow

- Work in small, testable steps. Propose a plan before large changes.
- When unsure about a design decision, propose options with trade-offs and ask.
- Keep `docs/devlog.md` updated with key decisions and anything that went wrong.
- GitHub owner: JCHETAN26. Repository: `strata`.