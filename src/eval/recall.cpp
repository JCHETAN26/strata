#include "strata/recall.hpp"

#include <algorithm>
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

}  // namespace strata
