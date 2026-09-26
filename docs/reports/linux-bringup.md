# Linux bring-up (IdeaPad, Ubuntu 22.04, Ryzen 7 5800H)

Unattended bring-up of Strata on the Linux results/server machine. Goal: build and test on Linux
with GCC 13, verify AVX2 natively, rebuild the SIFT datasets, build the Python bindings, and run
short sanity benchmarks. No long benchmarks were run and no numbers here are for publication.

- **Date:** 2026-09-25
- **Commit at completion:** `c4a4b27` (plus this report)
- **Operator:** autonomous session (Claude Opus 4.8), no human input during the run.

## Toolchain

| Tool | Version | Notes |
|------|---------|-------|
| OS | Ubuntu 22.04 (kernel 6.8.0-138-generic) | |
| CPU | AMD Ryzen 7 5800H, 8C/16T | AVX2 + FMA present |
| RAM | ~19 GiB usable | |
| C++ compiler | GCC 13.4.0 (`/usr/bin/g++-13`) | system default `g++` is 11.4.0 |
| CMake | 4.4.3 | |
| Ninja | 1.11.1 | |
| vcpkg | `$VCPKG_ROOT=/home/grv22/vcpkg`, baseline `617ef1c0` | tl-expected, utf8proc, gtest, benchmark |
| uv | 0.8.22 | manages the Python venv (CPython 3.11.13) |

The `vm.mmap_rnd_bits` sysctl for TSan was already set by the operator; TSan ran without the usual
ASLR workaround.

## CMake presets

The default `debug`/`release`/`asan`/`tsan` presets set no compiler, so on Linux they would pick up
the system default `g++` (11.4). Rather than touch the existing (macOS-working) presets, I added:

- A hidden `linux-base` preset, gated by `condition: hostSystemName == Linux`, that pins
  `/usr/bin/gcc-13` and `/usr/bin/g++-13`.
- `linux-debug`, `linux-release`, `linux-asan`, `linux-tsan`, each inheriting
  `["<original>", "linux-base"]` so they reuse the original cache variables and only add the
  compiler. Matching `buildPresets` and `testPresets` were added (the test presets inherit the
  originals to keep the ASan/TSan/`outputOnFailure` environment).

The macOS presets (including `rosetta-avx2`, which stays `Darwin`-only) are unchanged. On Linux,
`cmake --list-presets` shows the four `linux-*` presets and correctly hides `rosetta-avx2`.

Use `--preset linux-debug` (etc.) on this machine; the bare `debug`/`release`/... presets are for
macOS.

## Build and test results

All four presets: **161 tests, 100% passed, 3 skipped**, no sanitizer reports. The 3 skips are
metric-inapplicable AVX2 kernel cases that `GTEST_SKIP` themselves at runtime
(`SimdKernel.ExactOnIntegerData/avx2_cosine`, `SimdKernel.CosineZeroVectorIsOne/avx2_{l2,ip}`) —
not Linux failures; they skip on macOS too.

| Preset | Configure | Build | Tests | Time |
|--------|-----------|-------|-------|------|
| `linux-debug`   | OK | OK | 161/161 pass (3 skip) | ~84 s |
| `linux-release` | OK | OK | 161/161 pass (3 skip) | ~19 s |
| `linux-asan`    | OK | OK | 161/161 pass (3 skip) | ~138 s |
| `linux-tsan`    | OK | OK | 161/161 pass (3 skip) | ~174 s |

The ASan run initially had **1 failure** (`SnapshotTest.EmptySnapshot`); it is fixed (see below) and
the table reflects the post-fix result.

## Fixes made

Four small commits, all keeping the macOS build working:

1. **`build: add Linux GCC 13 CMake presets`** — the preset work above.

2. **`test: include <algorithm> for std::ranges in bm25/filter tests`** — GCC 13's libstdc++ does
   not transitively include `<algorithm>`, so `std::ranges::count` (`tests/bm25_test.cpp:42`) and
   `std::ranges::sort` (`tests/filter_test.cpp:164`) failed to compile:
   `error: 'count'/'sort' is not a member of 'std::ranges'`. macOS libc++ happened to pull it in.
   Added the explicit include to both (a no-op on macOS). This was the only compile break.

3. **`fix(storage): avoid memcpy with null dst on empty snapshot`** — real UB caught by UBSan under
   the asan preset:
   ```
   src/storage/snapshot.cpp:115:14: runtime error: null pointer passed as argument 1,
   which is declared to never be null
       #0 strata::read_snapshot(...) src/storage/snapshot.cpp:115
       #1 TestBody tests/storage_test.cpp:142   (SnapshotTest.EmptySnapshot)
   ```
   For an empty snapshot `count == 0`, so `vector_bytes == 0` and the destination
   `snapshot.vectors.data().data()` is null; `memcpy(null, …, 0)` is undefined even though it copies
   nothing. Guarded the copy on `vector_bytes != 0`. Re-verified under the asan preset (161/161).
   This is a genuine latent bug, not Linux-specific — macOS just didn't trip UBSan on it.

4. **`test(python): find strata_reference under linux-* build dirs`** — the Python↔C++ agreement
   test searched only `build/{release,debug,asan}` for the `strata_reference` tool. The new presets
   build into `build/linux-*`, so the cross-check silently skipped on Linux. Added the `linux-*`
   dirs to the search list (macOS paths unchanged); the cross-check now runs.

No files under `src/index/hnsw*` were touched (none exist yet, and HNSW remains off). No `sudo`,
Docker, or `git push`.

## AVX2 / SIMD verification

- **CPU flags:** `/proc/cpuinfo` and `lscpu` both report `avx2` and `fma` (no AVX-512).
- **Compiler:** `g++-13 -mavx2 -mfma` defines `__AVX2__` and `__FMA__`; `include/strata/distance.hpp`
  sets `STRATA_HAS_AVX2` from those, so the AVX2 kernels are compiled in.
- **Dispatch selected at runtime:** the search harness reports `"kernel": "avx2"`, and the Python
  module's `build_info()` reports `{'kernel': 'avx2', 'compiler': 'GNU 13.4.0',
  'fp_flags': '-ffp-contract=off -fno-fast-math -mavx2 -mfma', 'target': 'x86_64'}`.
- **SIMD-vs-scalar tests pass:** the `SimdKernel.*/avx2_{l2,ip,cosine}` cases (match-scalar-within-
  tolerance, exact-on-integer-data, identical-vectors, no-read-past-end) all pass; only the
  metric-inapplicable cases skip.
- **Speedup sanity (dim 128 micro-bench):** AVX2 vs scalar ≈ 8–10× (`avx2/l2/128` 6.06 ns vs
  `scalar/l2/128` 54.0 ns; `avx2/ip/128` 5.40 ns vs 52.7 ns), consistent with AVX2 actually running.

## Datasets

Rebuilt natively with `uv run --group bench python scripts/prepare_datasets.py siftsmall sift1m`
(not copied from the Mac). Both wrote to `data/` (gitignored):

| Dataset | base | query | groundtruth | SHA-256 of source |
|---------|------|-------|-------------|-------------------|
| `siftsmall` (SIFT10K) | 10000×128 | 100×128 | 100×100 | `b8f1e59b…52770e` |
| `sift1m` | 1000000×128 | 10000×128 | 10000×100 | `dd6f0a6e…0ca5984` |

`siftsmall` is fetched over FTP (irisa.fr) and `sift1m` over HTTP (ann-benchmarks.com); both
downloaded fine here.

## Python bindings

Built with `CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 uv sync --reinstall-package strata` (GCC 13 for
consistency with the C++ build; scikit-build-core + nanobind, Release). The module imports and
reports the AVX2 build info above.

Python test suite: **70 passed, 6 skipped** (`PYTHONPATH= uv run pytest -q`). The 6 skips are all
out-of-scope prerequisites: HNSW not built (1), and BEIR/SciFact, Porter vocab, and BEIR
embeddings not prepared (5) — none requested for this bring-up.

## Sanity benchmarks (NOT for publication)

Short, single-machine, small-N runs only, to confirm the harnesses work on Linux. Not averaged, not
pinned, background load present — **do not cite these**.

- **Brute-force search, SIFT10K, k=10, 3 runs** (`strata_search --data data/siftsmall`):
  - AVX2: ~11.8–12.2k QPS, recall@10 = 1.0
  - scalar: ~1.79k QPS, recall@10 = 1.0  (≈6.6× end-to-end, dominated by distance)
- **Distance micro-bench** (`strata_bench`, min-time 0.05s): see AVX2/scalar numbers above.
- **Filtered search, SIFT10K** (`strata_filter_bench`, selectivity 0.1/0.5): runs, recall = 1.0,
  prefilter faster than scan-check as expected (e.g. ~38k vs ~17k QPS at selectivity ≈0.1).
- **Storage** (`strata_storage_bench`, SIFT10K, 1 run): ~325k inserts/s no-sync, ~247 inserts/s with
  fsync, WAL recovery ~20 ms / snapshot recovery ~16 ms. (fsync rate is disk-bound; expected.)

## Differences from the macOS build

- **Compiler:** GCC 13.4.0 (libstdc++) vs AppleClang (libc++). libstdc++ needs the explicit
  `<algorithm>` include (fix #2); this is the only source-level portability delta found.
- **SIMD family:** AVX2 + FMA on x86_64 vs NEON on arm64. Both verified against the scalar reference.
- **`build_info`:** `kFpFlags` includes `-mavx2 -mfma` and `kTarget = x86_64` here.
- **Presets/paths:** Linux uses `linux-*` presets → `build/linux-*` directories; macOS uses the bare
  presets → `build/<preset>`.
- **Tests run:** 161 here (dataset-dependent and hybrid-fusion tests included); otherwise identical.

## Unresolved / caveats (with exact errors)

1. **ROS environment leaks into pytest (environmental, worked around, not a project bug).**
   The machine's ambient `PYTHONPATH` includes
   `/opt/ros/humble/lib/python3.10/site-packages`, whose `launch_testing` registers a pytest plugin.
   Plain `uv run pytest` fails at *collection*:
   ```
   File ".../ros/humble/.../launch/utilities/type_utils.py", line 29, in <module>
       import yaml
   ModuleNotFoundError: No module named 'yaml'
   ```
   Diagnosis: pytest autoloads the ROS plugin from the leaked path; it can't import `yaml` inside the
   venv. Workaround: run with `PYTHONPATH=` cleared (used for all Python results above). Suggested
   permanent fix: set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (or clear `PYTHONPATH`) in the test
   environment, or add it to `[tool.pytest.ini_options]`/CI env. Not committed since it's a
   workstation-config issue, not a repo issue.

2. **Harmless CMake warning.** Every Linux configure prints
   `Manually-specified variables were not used by the project: CMAKE_C_COMPILER`, because the project
   is CXX-only. `CMAKE_C_COMPILER` is kept for vcpkg's C ports; the warning is cosmetic and can be
   ignored (or silenced later with a `LANGUAGES CXX C` project() line if desired).

No problem hit the 3-attempts-and-stop rule; everything attempted was resolved.

## Notes for reproducibility

- The `linux-debug` build directory was first configured at 22:57 while the working tree was still
  settling (`tests/CMakeLists.txt` reached its committed content, with `fusion_test.cpp`, at 23:00),
  so the first debug binary was momentarily stale (153 tests). A reconfigure + rebuild fixed it to
  161. If you ever see a preset short on tests, just re-run `cmake --preset <name>` before building.

## Suggested next steps

1. Add a CI job (or documented local step) that runs `linux-{debug,release,asan,tsan}` with
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, so the ROS leak can't bite and the GCC-13 build stays green.
2. When `src/index/hnsw.cpp` lands, HNSW tests and the HNSW skips (C++ and Python) light up
   automatically; re-run the four presets and the Python suite.
3. This machine is the designated results box: once HNSW exists, run the real recall@10-vs-QPS
   sweeps here (SIFT1M, multiple runs with variance) per the benchmarking rules — the sanity numbers
   above are explicitly not those.
4. Optional cleanup: silence the `CMAKE_C_COMPILER` warning, and consider a top-level convenience so
   `README` build instructions mention the `linux-*` presets alongside the macOS ones.
