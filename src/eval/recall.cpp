#include "strata/recall.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace strata {

Expected<double> recall_at_k(std::span<const std::vector<Neighbor>> results,
                             const Matrix<std::int32_t>& groundtruth, std::size_t k) {
  if (k == 0) {
    return make_error(ErrorCode::kInvalidArgument, "k must be positive");
  }
  if (k > groundtruth.cols()) {
    return make_error(ErrorCode::kInvalidArgument, "k = " + std::to_string(k) +
                                                       " exceeds groundtruth width " +
                                                       std::to_string(groundtruth.cols()));
  }
  if (results.size() != groundtruth.rows()) {
    return make_error(ErrorCode::kInvalidArgument,
                      std::to_string(results.size()) + " result rows for " +
                          std::to_string(groundtruth.rows()) + " groundtruth rows");
  }
  if (results.empty()) {
    return make_error(ErrorCode::kInvalidArgument, "no queries");
  }

  std::size_t hits = 0;
  std::vector<std::int32_t> truth(k);
  for (std::size_t q = 0; q < results.size(); ++q) {
    const auto row = groundtruth.row(q).first(k);
    std::copy(row.begin(), row.end(), truth.begin());
    std::sort(truth.begin(), truth.end());
    const std::size_t n = std::min(k, results[q].size());
    for (std::size_t i = 0; i < n; ++i) {
      const auto id = static_cast<std::int32_t>(results[q][i].id);
      if (std::binary_search(truth.begin(), truth.end(), id)) {
        ++hits;
      }
    }
  }
  return static_cast<double>(hits) / static_cast<double>(results.size() * k);
}

Expected<std::vector<float>> kth_neighbor_distances(const Matrix<float>& base,
                                                    const Matrix<float>& queries,
                                                    const Matrix<std::int32_t>& groundtruth,
                                                    Metric metric, std::size_t k) {
  if (k == 0 || k > groundtruth.cols()) {
    return make_error(ErrorCode::kInvalidArgument, "k must be in [1, groundtruth width]");
  }
  if (base.cols() != queries.cols()) {
    return make_error(ErrorCode::kDimensionMismatch, "base and query dimensions differ");
  }
  if (queries.rows() != groundtruth.rows()) {
    return make_error(ErrorCode::kInvalidArgument, "query and groundtruth row counts differ");
  }
  const DistanceFn dist = distance_function(metric);
  std::vector<float> out(queries.rows());
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    const std::int32_t id = groundtruth.row(q)[k - 1];
    if (id < 0 || static_cast<std::size_t>(id) >= base.rows()) {
      return make_error(ErrorCode::kCorruptData,
                        "groundtruth id " + std::to_string(id) + " out of range");
    }
    out[q] = dist(queries.row(q), base.row(static_cast<std::size_t>(id)));
  }
  return out;
}

Expected<double> recall_at_k_with_ties(std::span<const std::vector<Neighbor>> results,
                                       std::span<const float> kth_distances, std::size_t k) {
  if (k == 0) {
    return make_error(ErrorCode::kInvalidArgument, "k must be positive");
  }
  if (results.size() != kth_distances.size()) {
    return make_error(ErrorCode::kInvalidArgument, "result and threshold counts differ");
  }
  if (results.empty()) {
    return make_error(ErrorCode::kInvalidArgument, "no queries");
  }
  std::size_t hits = 0;
  for (std::size_t q = 0; q < results.size(); ++q) {
    const float threshold =
        kth_distances[q] + kRecallTolerance * std::max(1.0F, std::abs(kth_distances[q]));
    const std::size_t n = std::min(k, results[q].size());
    for (std::size_t i = 0; i < n; ++i) {
      if (results[q][i].distance <= threshold) {
        ++hits;
      }
    }
  }
  return static_cast<double>(hits) / static_cast<double>(results.size() * k);
}

}  // namespace strata
