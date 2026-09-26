#include "strata/recall.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace strata {
namespace {

Matrix<std::int32_t> groundtruth(std::vector<std::int32_t> ids, std::size_t rows,
                                 std::size_t cols) {
  return {rows, cols, std::move(ids)};
}

std::vector<Neighbor> ids(std::initializer_list<VectorId> list) {
  std::vector<Neighbor> out;
  for (VectorId id : list) {
    out.push_back({id, 0.0F});
  }
  return out;
}

TEST(Recall, PerfectMatch) {
  const auto gt = groundtruth({1, 2, 3, 4, 5, 6}, 2, 3);
  const std::vector<std::vector<Neighbor>> results{ids({1, 2, 3}), ids({4, 5, 6})};
  EXPECT_DOUBLE_EQ(*recall_at_k(results, gt, 3), 1.0);
}

TEST(Recall, OrderWithinTopKDoesNotMatter) {
  const auto gt = groundtruth({1, 2, 3}, 1, 3);
  const std::vector<std::vector<Neighbor>> results{ids({3, 1, 2})};
  EXPECT_DOUBLE_EQ(*recall_at_k(results, gt, 3), 1.0);
}

TEST(Recall, PartialMatch) {
  const auto gt = groundtruth({1, 2, 3, 4, 5, 6}, 2, 3);
  const std::vector<std::vector<Neighbor>> results{ids({1, 9, 9}), ids({4, 5, 9})};
  EXPECT_DOUBLE_EQ(*recall_at_k(results, gt, 3), 3.0 / 6.0);
}

TEST(Recall, UsesOnlyFirstKGroundtruthColumns) {
  const auto gt = groundtruth({1, 2, 3, 4}, 1, 4);
  // 3 and 4 are true neighbors, but not in the top 2.
  const std::vector<std::vector<Neighbor>> results{ids({1, 3})};
  EXPECT_DOUBLE_EQ(*recall_at_k(results, gt, 2), 0.5);
}

TEST(Recall, ShortResultListsCountMissesAsZero) {
  const auto gt = groundtruth({1, 2, 3, 4}, 1, 4);
  const std::vector<std::vector<Neighbor>> results{ids({1})};
  EXPECT_DOUBLE_EQ(*recall_at_k(results, gt, 4), 0.25);
}

TEST(Recall, ExtraResultsBeyondKAreIgnored) {
  const auto gt = groundtruth({1, 2}, 1, 2);
  const std::vector<std::vector<Neighbor>> results{ids({9, 1, 2})};
  EXPECT_DOUBLE_EQ(*recall_at_k(results, gt, 2), 0.5);
}

TEST(Recall, InvalidArguments) {
  const auto gt = groundtruth({1, 2, 3}, 1, 3);
  const std::vector<std::vector<Neighbor>> one{ids({1})};
  EXPECT_FALSE(recall_at_k(one, gt, 0).has_value());
  EXPECT_FALSE(recall_at_k(one, gt, 4).has_value());
  const std::vector<std::vector<Neighbor>> two{ids({1}), ids({2})};
  EXPECT_FALSE(recall_at_k(two, gt, 1).has_value());
  const auto empty_gt = groundtruth({}, 0, 3);
  EXPECT_FALSE(recall_at_k({}, empty_gt, 1).has_value());
}

}  // namespace
}  // namespace strata
