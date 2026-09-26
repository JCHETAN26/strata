#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/matrix.hpp"
#include "strata/types.hpp"

namespace strata {

// Recall@k: for each query, the fraction of the true k nearest neighbors (the first k columns of
// the groundtruth row) that appear among the first k results; averaged over queries.
// A query that returns fewer than k results counts the missing ones as misses.
//
// Matching is by id, so when several vectors tie at the k-th distance, an equally good result can
// count as a miss (seen on SIFT1M, whose features are integers). recall_at_k_with_ties avoids this.
//
// Fails if k == 0, k exceeds the groundtruth width, or the result and groundtruth row counts
// differ. Thread safety: pure function.
[[nodiscard]] Expected<double> recall_at_k(std::span<const std::vector<Neighbor>> results,
                                           const Matrix<std::int32_t>& groundtruth, std::size_t k);

// Distance from each query to its k-th true nearest neighbor (groundtruth column k - 1).
// Fails on dimension or row-count mismatch, k == 0, k > groundtruth width, or an out-of-range id.
[[nodiscard]] Expected<std::vector<float>> kth_neighbor_distances(
    const Matrix<float>& base, const Matrix<float>& queries,
    const Matrix<std::int32_t>& groundtruth, Metric metric, std::size_t k);

// Tie-aware recall@k (the ann-benchmarks definition): a result counts as a hit if its distance is
// at most the k-th true neighbor's distance, plus a relative tolerance of kRecallTolerance for
// float rounding. Only the first k results of each query are considered.
// Precondition for meaningful results: result distances are exact (true for every Strata index,
// which ranks candidates by full-precision distance).
inline constexpr float kRecallTolerance = 1e-5F;
[[nodiscard]] Expected<double> recall_at_k_with_ties(std::span<const std::vector<Neighbor>> results,
                                                     std::span<const float> kth_distances,
                                                     std::size_t k);

}  // namespace strata
