#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "strata/error.hpp"
#include "strata/types.hpp"

namespace strata {

// Result fusion for hybrid retrieval. Inputs are ranked lists from different retrievers over the
// same id space (e.g. a vector index and Bm25Index), each in the usual Neighbor convention:
// sorted best first, `distance` lower is better (Bm25Index stores negated scores there).
//
// Outputs are Neighbors sorted by fused score descending, ties by ascending id, with `distance`
// holding the *negated* fused score, so they sort like every other result list.
// Thread safety: pure functions.

// Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009):
//   score(d) = sum over lists containing d of 1 / (k + rank(d)),  rank starting at 1.
// Only ranks matter, not the retrievers' score scales. k = 60 is the paper's value.
// Fails if k < 0.
[[nodiscard]] Expected<std::vector<Neighbor>> reciprocal_rank_fusion(
    std::span<const std::vector<Neighbor>> lists, std::size_t top_k, double k = 60.0);

// Weighted score fusion (a convex combination of normalized scores):
//   score(d) = sum over lists i of weights[i] * norm_i(d)
// where norm_i min-max scales list i's similarities (-distance) to [0, 1] over that list's
// candidates (all equal -> 1), and a document missing from list i gets norm_i = 0.
// Fails if weights.size() != lists.size() or any weight is negative or not finite.
[[nodiscard]] Expected<std::vector<Neighbor>> weighted_score_fusion(
    std::span<const std::vector<Neighbor>> lists, std::span<const double> weights,
    std::size_t top_k);

}  // namespace strata
