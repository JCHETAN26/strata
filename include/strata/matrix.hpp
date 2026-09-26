#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace strata {

// Non-owning row-major view: rows x cols values starting at `data`. Used by APIs that only read
// their input, so callers (e.g. NumPy via the Python bindings) can pass memory they own without a
// copy. The viewed memory must outlive the view and not change while an API reads it.
// Thread safety: same as the memory it views.
template <typename T>
class MatrixView {
 public:
  MatrixView() = default;
  MatrixView(T* data, std::size_t rows, std::size_t cols) noexcept
      : data_(data), rows_(rows), cols_(cols) {}

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] bool empty() const noexcept { return rows_ == 0; }
  [[nodiscard]] std::span<T> row(std::size_t i) const noexcept {
    assert(i < rows_);
    return {data_ + i * cols_, cols_};
  }
  [[nodiscard]] std::span<T> data() const noexcept { return {data_, rows_ * cols_}; }

 private:
  T* data_ = nullptr;
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
};

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

  // Implicit, so every API taking MatrixView<const T> also accepts a Matrix<T>.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator MatrixView<const T>() const noexcept { return {data_.data(), rows_, cols_}; }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<T> data_;
};

}  // namespace strata
