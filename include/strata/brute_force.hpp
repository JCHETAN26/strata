#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/matrix.hpp"
#include "strata/thread_pool.hpp"
#include "strata/types.hpp"

namespace strata {

// Exact k-nearest-neighbor search by scanning every vector. This is the ground truth that
// approximate indexes are measured against, and the baseline for every benchmark.
//
// Vectors live in one contiguous buffer; ids are assigned densely in insertion order.
//
// Thread safety: concurrent calls to const methods (search, size, ...) are safe. add/add_batch
// require exclusive access: no other call may run concurrently with them.
class BruteForceIndex {
 public:
  // Fails if dim == 0. `kernels` selects SIMD or the scalar reference (for comparisons).
  [[nodiscard]] static Expected<BruteForceIndex> create(std::size_t dim, Metric metric,
                                                        KernelSet kernels = KernelSet::kBest);

  // Appends a vector and returns its id. Fails on dimension mismatch.
  Expected<VectorId> add(std::span<const float> vector);
  // Appends every row. Fails (adding nothing) on dimension mismatch.
  Expected<void> add_batch(const Matrix<float>& vectors);

  // The min(k, size()) nearest vectors, sorted by (distance, id) ascending.
  // Fails on dimension mismatch. k == 0 or an empty index gives an empty result.
  [[nodiscard]] Expected<std::vector<Neighbor>> search(std::span<const float> query,
                                                       std::size_t k) const;

  // search() for every row of `queries`, spread across `pool`. results[i] answers queries.row(i).
  // Fails on dimension mismatch before doing any work.
  [[nodiscard]] Expected<std::vector<std::vector<Neighbor>>> search_batch(
      const Matrix<float>& queries, std::size_t k, ThreadPool& pool) const;

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
  [[nodiscard]] Metric metric() const noexcept { return metric_; }
  [[nodiscard]] std::span<const float> vector(VectorId id) const noexcept;

 private:
  BruteForceIndex(std::size_t dim, Metric metric, KernelSet kernels);

  std::size_t dim_;
  Metric metric_;
  DistanceFn distance_;
  std::size_t size_ = 0;
  std::vector<float> data_;
};

}  // namespace strata
