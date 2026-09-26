#include "strata/brute_force.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <string>

namespace strata {

namespace {

tl::unexpected<Error> dimension_error(std::size_t expected, std::size_t got) {
  return make_error(
      ErrorCode::kDimensionMismatch,
      "expected dimension " + std::to_string(expected) + ", got " + std::to_string(got));
}

}  // namespace

BruteForceIndex::BruteForceIndex(std::size_t dim, Metric metric)
    : dim_(dim), metric_(metric), distance_(distance_function(metric)) {}

Expected<BruteForceIndex> BruteForceIndex::create(std::size_t dim, Metric metric) {
  if (dim == 0) {
    return make_error(ErrorCode::kInvalidArgument, "dimension must be positive");
  }
  return BruteForceIndex(dim, metric);
}

Expected<VectorId> BruteForceIndex::add(std::span<const float> vector) {
  if (vector.size() != dim_) {
    return dimension_error(dim_, vector.size());
  }
  if (size_ >= std::numeric_limits<VectorId>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "index is full");
  }
  data_.insert(data_.end(), vector.begin(), vector.end());
  return static_cast<VectorId>(size_++);
}

Expected<void> BruteForceIndex::add_batch(const Matrix<float>& vectors) {
  if (vectors.empty()) {
    return {};
  }
  if (vectors.cols() != dim_) {
    return dimension_error(dim_, vectors.cols());
  }
  if (size_ + vectors.rows() > std::numeric_limits<VectorId>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "batch would overflow the id space");
  }
  const auto values = vectors.data();
  data_.insert(data_.end(), values.begin(), values.end());
  size_ += vectors.rows();
  return {};
}

Expected<std::vector<Neighbor>> BruteForceIndex::search(std::span<const float> query,
                                                        std::size_t k) const {
  if (query.size() != dim_) {
    return dimension_error(dim_, query.size());
  }
  k = std::min(k, size_);
  std::vector<Neighbor> heap;  // max-heap on (distance, id): heap.front() is the worst kept
  if (k == 0) {
    return heap;
  }
  heap.reserve(k + 1);

  const float* row = data_.data();
  for (std::size_t i = 0; i < size_; ++i, row += dim_) {
    const Neighbor candidate{static_cast<VectorId>(i), distance_(query, {row, dim_})};
    if (heap.size() < k) {
      heap.push_back(candidate);
      std::push_heap(heap.begin(), heap.end());
    } else if (candidate < heap.front()) {
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = candidate;
      std::push_heap(heap.begin(), heap.end());
    }
  }
  std::sort_heap(heap.begin(), heap.end());
  return heap;
}

std::span<const float> BruteForceIndex::vector(VectorId id) const noexcept {
  assert(id < size_);
  return {data_.data() + static_cast<std::size_t>(id) * dim_, dim_};
}

}  // namespace strata
