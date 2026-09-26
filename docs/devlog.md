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
