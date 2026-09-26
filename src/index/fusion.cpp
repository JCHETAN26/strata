#include "strata/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace strata {

namespace {

// Top k of accumulated scores as Neighbors (distance = -score), score desc then id asc.
std::vector<Neighbor> top_by_score(const std::unordered_map<VectorId, double>& scores,
                                   std::size_t top_k) {
  std::vector<std::pair<double, VectorId>> ranked;
  ranked.reserve(scores.size());
  for (const auto& [id, score] : scores) {
    ranked.emplace_back(score, id);
  }
  const auto better = [](const auto& a, const auto& b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  };
  const std::size_t n = std::min(top_k, ranked.size());
  std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(n), ranked.end(),
                    better);
  std::vector<Neighbor> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back({ranked[i].second, static_cast<float>(-ranked[i].first)});
  }
  return out;
}

}  // namespace

Expected<std::vector<Neighbor>> reciprocal_rank_fusion(std::span<const std::vector<Neighbor>> lists,
                                                       std::size_t top_k, double k) {
  if (!(k >= 0.0) || !std::isfinite(k)) {
    return make_error(ErrorCode::kInvalidArgument, "RRF k must be a finite non-negative number");
  }
  std::unordered_map<VectorId, double> scores;
  for (const auto& list : lists) {
    for (std::size_t rank = 0; rank < list.size(); ++rank) {
      scores[list[rank].id] += 1.0 / (k + static_cast<double>(rank + 1));
    }
  }
  return top_by_score(scores, top_k);
}

Expected<std::vector<Neighbor>> weighted_score_fusion(std::span<const std::vector<Neighbor>> lists,
                                                      std::span<const double> weights,
                                                      std::size_t top_k) {
  if (weights.size() != lists.size()) {
    return make_error(ErrorCode::kInvalidArgument, "one weight per list required");
  }
  for (double w : weights) {
    if (!(w >= 0.0) || !std::isfinite(w)) {
      return make_error(ErrorCode::kInvalidArgument, "weights must be finite and non-negative");
    }
  }
  std::unordered_map<VectorId, double> scores;
  for (std::size_t i = 0; i < lists.size(); ++i) {
    const auto& list = lists[i];
    if (list.empty()) {
      continue;
    }
    // Similarity = -distance; min-max over this list's candidates.
    double lo = -static_cast<double>(list.front().distance);
    double hi = lo;
    for (const auto& n : list) {
      const double s = -static_cast<double>(n.distance);
      lo = std::min(lo, s);
      hi = std::max(hi, s);
    }
    const double range = hi - lo;
    for (const auto& n : list) {
      const double s = -static_cast<double>(n.distance);
      const double norm = range > 0.0 ? (s - lo) / range : 1.0;
      scores[n.id] += weights[i] * norm;
    }
  }
  return top_by_score(scores, top_k);
}

}  // namespace strata
