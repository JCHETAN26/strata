#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "strata/error.hpp"
#include "strata/matrix.hpp"
#include "strata/types.hpp"

namespace strata {

// Recall@k: for each query, the fraction of the true k nearest neighbors (the first k columns of
// the groundtruth row) that appear among the first k results; averaged over queries.
// A query that returns fewer than k results counts the missing ones as misses.
//
// Matching is by id. Ties at the k-th distance can make an equally good result count as a miss;
// with float features this is rare and is the same convention hnswlib and FAISS examples use.
//
// Fails if k == 0, k exceeds the groundtruth width, or the result and groundtruth row counts
// differ. Thread safety: pure function.
[[nodiscard]] Expected<double> recall_at_k(std::span<const std::vector<Neighbor>> results,
                                           const Matrix<std::int32_t>& groundtruth, std::size_t k);

}  // namespace strata
