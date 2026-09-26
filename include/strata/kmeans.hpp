#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "strata/error.hpp"
#include "strata/matrix.hpp"
#include "strata/thread_pool.hpp"

namespace strata {

struct KMeansParams {
  std::size_t k = 256;
  std::size_t max_iterations = 25;
  std::uint64_t seed = 42;
  // Stop early when the relative drop in inertia (sum of squared distances) falls below this.
  double tolerance = 1e-4;
};

struct KMeansResult {
  Matrix<float> centroids;                // k x dim
  std::vector<std::uint32_t> assignment;  // cluster of each input row
  double inertia = 0.0;                   // sum of squared L2 distances to assigned centroids
  std::size_t iterations = 0;
};

// Lloyd's k-means under squared L2, with k-means++ initialization.
// Empty clusters are re-seeded by splitting the largest cluster (its centroid is copied and
// nudged), so every centroid ends up used when there are at least k distinct points.
// Deterministic for a given seed and pool size.
//
// Fails if k == 0, the data is empty, or there are fewer rows than k.
// Thread safety: pure function; uses `pool` (if given) for the assignment step.
[[nodiscard]] Expected<KMeansResult> kmeans(const Matrix<float>& data, const KMeansParams& params,
                                            ThreadPool* pool = nullptr);

}  // namespace strata
