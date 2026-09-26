#pragma once

#include <cstdint>
#include <filesystem>

#include "strata/error.hpp"
#include "strata/matrix.hpp"

namespace strata {

// Readers and writers for the .fbin / .ibin format produced by scripts/prepare_datasets.py:
// little-endian uint32 num_rows, uint32 num_cols, then num_rows * num_cols values, row-major.
// Thread safety: free functions with no shared state.

[[nodiscard]] Expected<Matrix<float>> read_fbin(const std::filesystem::path& path);
[[nodiscard]] Expected<Matrix<std::int32_t>> read_ibin(const std::filesystem::path& path);

[[nodiscard]] Expected<void> write_fbin(const std::filesystem::path& path,
                                        const Matrix<float>& matrix);
[[nodiscard]] Expected<void> write_ibin(const std::filesystem::path& path,
                                        const Matrix<std::int32_t>& matrix);

// A benchmark dataset directory: base.fbin, query.fbin, groundtruth.ibin.
struct Dataset {
  Matrix<float> base;
  Matrix<float> query;
  Matrix<std::int32_t> groundtruth;
};

[[nodiscard]] Expected<Dataset> load_dataset(const std::filesystem::path& dir);

}  // namespace strata
