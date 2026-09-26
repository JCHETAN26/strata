#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace strata {

// Row-major matrix in one contiguous buffer. Row i is data[i * cols, (i + 1) * cols).
// Thread safety: concurrent const access is safe; mutation requires exclusive access.
template <typename T>
class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
  // Precondition: data.size() == rows * cols.
  Matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
      : rows_(rows), cols_(cols), data_(std::move(data)) {
    assert(data_.size() == rows_ * cols_);
  }

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] bool empty() const noexcept { return rows_ == 0; }

  [[nodiscard]] std::span<const T> row(std::size_t i) const noexcept {
    assert(i < rows_);
    return {data_.data() + i * cols_, cols_};
  }
  [[nodiscard]] std::span<T> row(std::size_t i) noexcept {
    assert(i < rows_);
    return {data_.data() + i * cols_, cols_};
  }

  [[nodiscard]] std::span<const T> data() const noexcept { return data_; }
  [[nodiscard]] std::span<T> data() noexcept { return data_; }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<T> data_;
};

}  // namespace strata
