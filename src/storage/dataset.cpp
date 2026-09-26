#include "strata/dataset.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace strata {

static_assert(std::endian::native == std::endian::little,
              "binary formats assume a little-endian host");

namespace {

template <typename T>
Expected<Matrix<T>> read_bin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return make_error(ErrorCode::kIoError, "cannot open " + path.string());
  }
  std::array<std::uint32_t, 2> header{};
  if (!in.read(reinterpret_cast<char*>(header.data()), sizeof(header))) {
    return make_error(ErrorCode::kCorruptData, path.string() + ": missing header");
  }
  const std::size_t rows = header[0];
  const std::size_t cols = header[1];
  if (rows > 0 && cols == 0) {
    return make_error(ErrorCode::kCorruptData, path.string() + ": zero columns");
  }

  std::error_code ec;
  const auto file_size = std::filesystem::file_size(path, ec);
  if (ec) {
    return make_error(ErrorCode::kIoError, path.string() + ": " + ec.message());
  }
  const std::size_t expected_size = sizeof(header) + rows * cols * sizeof(T);
  if (file_size != expected_size) {
    return make_error(ErrorCode::kCorruptData,
                      path.string() + ": header says " + std::to_string(rows) + "x" +
                          std::to_string(cols) + " (" + std::to_string(expected_size) +
                          " bytes), file is " + std::to_string(file_size) + " bytes");
  }

  std::vector<T> data(rows * cols);
  const auto bytes = static_cast<std::streamsize>(data.size() * sizeof(T));
  if (!in.read(reinterpret_cast<char*>(data.data()), bytes)) {
    return make_error(ErrorCode::kIoError, path.string() + ": short read");
  }
  return Matrix<T>(rows, cols, std::move(data));
}

template <typename T>
Expected<void> write_bin(const std::filesystem::path& path, const Matrix<T>& matrix) {
  constexpr auto kMax = std::numeric_limits<std::uint32_t>::max();
  if (matrix.rows() > kMax || matrix.cols() > kMax) {
    return make_error(ErrorCode::kInvalidArgument, "matrix too large for uint32 header");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return make_error(ErrorCode::kIoError, "cannot open " + path.string() + " for writing");
  }
  const std::array<std::uint32_t, 2> header{static_cast<std::uint32_t>(matrix.rows()),
                                            static_cast<std::uint32_t>(matrix.cols())};
  out.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
  const auto data = matrix.data();
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size_bytes()));
  if (!out) {
    return make_error(ErrorCode::kIoError, path.string() + ": write failed");
  }
  return {};
}

}  // namespace

Expected<Matrix<float>> read_fbin(const std::filesystem::path& path) {
  return read_bin<float>(path);
}

Expected<Matrix<std::int32_t>> read_ibin(const std::filesystem::path& path) {
  return read_bin<std::int32_t>(path);
}

Expected<void> write_fbin(const std::filesystem::path& path, const Matrix<float>& matrix) {
  return write_bin(path, matrix);
}

Expected<void> write_ibin(const std::filesystem::path& path, const Matrix<std::int32_t>& matrix) {
  return write_bin(path, matrix);
}

Expected<Dataset> load_dataset(const std::filesystem::path& dir) {
  auto base = read_fbin(dir / "base.fbin");
  if (!base) {
    return tl::unexpected(base.error());
  }
  auto query = read_fbin(dir / "query.fbin");
  if (!query) {
    return tl::unexpected(query.error());
  }
  auto groundtruth = read_ibin(dir / "groundtruth.ibin");
  if (!groundtruth) {
    return tl::unexpected(groundtruth.error());
  }
  if (base->cols() != query->cols()) {
    return make_error(ErrorCode::kDimensionMismatch,
                      "base has dimension " + std::to_string(base->cols()) +
                          ", query has dimension " + std::to_string(query->cols()));
  }
  if (groundtruth->rows() != query->rows()) {
    return make_error(ErrorCode::kCorruptData,
                      "groundtruth has " + std::to_string(groundtruth->rows()) + " rows for " +
                          std::to_string(query->rows()) + " queries");
  }
  return Dataset{std::move(*base), std::move(*query), std::move(*groundtruth)};
}

}  // namespace strata
