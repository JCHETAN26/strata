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

BruteForceIndex::BruteForceIndex(std::size_t dim, Metric metric, KernelSet kernels)
    : dim_(dim), metric_(metric), distance_(distance_function(metric, kernels)) {}

Expected<BruteForceIndex> BruteForceIndex::create(std::size_t dim, Metric metric,
                                                  KernelSet kernels) {
  if (dim == 0) {
    return make_error(ErrorCode::kInvalidArgument, "dimension must be positive");
  }
  return BruteForceIndex(dim, metric, kernels);
}

Expected<VectorId> BruteForceIndex::add(std::span<const float> vector) {
  if (vector.size() != dim_) {
    return dimension_error(dim_, vector.size());
  }
  if (size_ >= std::numeric_limits<VectorId>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "index is full");
  }
  data_.insert(data_.end(), vector.begin(), vector.end());
  deleted_.push_back(0);
  return static_cast<VectorId>(size_++);
}

Expected<void> BruteForceIndex::add_batch(MatrixView<const float> vectors) {
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
  deleted_.resize(deleted_.size() + vectors.rows(), 0);
  size_ += vectors.rows();
  return {};
}

Expected<void> BruteForceIndex::remove(VectorId id) {
  if (id >= size_) {
    return make_error(ErrorCode::kNotFound, "no vector with id " + std::to_string(id));
  }
  if (deleted_[id] != 0) {
    return make_error(ErrorCode::kNotFound, "vector " + std::to_string(id) + " already deleted");
  }
  deleted_[id] = 1;
  ++num_deleted_;
  return {};
}

bool BruteForceIndex::is_deleted(VectorId id) const noexcept {
  assert(id < size_);
  return deleted_[id] != 0;
}

// Two instantiations so an index with no deletes pays nothing for tombstone checks.
template <bool kSkipDeleted>
void BruteForceIndex::scan(std::span<const float> query, std::size_t k,
                           std::vector<Neighbor>& heap) const {
  const float* row = data_.data();
  for (std::size_t i = 0; i < size_; ++i, row += dim_) {
    if constexpr (kSkipDeleted) {
      if (deleted_[i] != 0) {
        continue;
      }
    }
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
}

Expected<std::vector<Neighbor>> BruteForceIndex::search(std::span<const float> query,
                                                        std::size_t k) const {
  if (query.size() != dim_) {
    return dimension_error(dim_, query.size());
  }
  k = std::min(k, live_size());
  std::vector<Neighbor> heap;  // max-heap on (distance, id): heap.front() is the worst kept
  if (k == 0) {
    return heap;
  }
  heap.reserve(k + 1);
  if (num_deleted_ == 0) {
    scan<false>(query, k, heap);
  } else {
    scan<true>(query, k, heap);
  }
  std::sort_heap(heap.begin(), heap.end());
  return heap;
}

namespace {

void push_top(std::vector<Neighbor>& heap, std::size_t k, Neighbor candidate) {
  if (heap.size() < k) {
    heap.push_back(candidate);
    std::push_heap(heap.begin(), heap.end());
  } else if (candidate < heap.front()) {
    std::pop_heap(heap.begin(), heap.end());
    heap.back() = candidate;
    std::push_heap(heap.begin(), heap.end());
  }
}

}  // namespace

Expected<std::vector<Neighbor>> BruteForceIndex::search_filtered(std::span<const float> query,
                                                                 std::size_t k,
                                                                 const Bitset& allowed) const {
  if (query.size() != dim_) {
    return dimension_error(dim_, query.size());
  }
  if (allowed.size() != size_) {
    return make_error(ErrorCode::kInvalidArgument,
                      "filter bitset has " + std::to_string(allowed.size()) + " bits for " +
                          std::to_string(size_) + " vectors");
  }
  std::vector<Neighbor> heap;
  if (k == 0) {
    return heap;
  }
  heap.reserve(k + 1);
  allowed.for_each_set([&](std::size_t i) {
    if (deleted_[i] == 0) {
      push_top(heap, k,
               {static_cast<VectorId>(i), distance_(query, {data_.data() + i * dim_, dim_})});
    }
  });
  std::sort_heap(heap.begin(), heap.end());
  return heap;
}

Expected<std::vector<Neighbor>> BruteForceIndex::search_predicate(
    std::span<const float> query, std::size_t k,
    const std::function<bool(VectorId)>& allowed_fn) const {
  if (query.size() != dim_) {
    return dimension_error(dim_, query.size());
  }
  std::vector<Neighbor> heap;
  if (k == 0) {
    return heap;
  }
  heap.reserve(k + 1);
  const float* row = data_.data();
  for (std::size_t i = 0; i < size_; ++i, row += dim_) {
    const auto id = static_cast<VectorId>(i);
    if (deleted_[i] == 0 && allowed_fn(id)) {
      push_top(heap, k, {id, distance_(query, {row, dim_})});
    }
  }
  std::sort_heap(heap.begin(), heap.end());
  return heap;
}

Expected<std::vector<std::vector<Neighbor>>> BruteForceIndex::search_batch(
    MatrixView<const float> queries, std::size_t k, ThreadPool& pool) const {
  if (!queries.empty() && queries.cols() != dim_) {
    return dimension_error(dim_, queries.cols());
  }
  std::vector<std::vector<Neighbor>> results(queries.rows());
  // Dimensions are validated above, so search() cannot fail here.
  pool.parallel_for(queries.rows(),
                    [&](std::size_t i) { results[i] = std::move(*search(queries.row(i), k)); });
  return results;
}

std::span<const float> BruteForceIndex::vector(VectorId id) const noexcept {
  assert(id < size_);
  return {data_.data() + static_cast<std::size_t>(id) * dim_, dim_};
}

}  // namespace strata
