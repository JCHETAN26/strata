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

std::vector<Neighbor> with_distances(std::initializer_list<float> distances) {
  std::vector<Neighbor> out;
  VectorId id = 0;
  for (float d : distances) {
    out.push_back({id++, d});
  }
  return out;
}

TEST(RecallWithTies, EqualDistanceCountsAsHit) {
  // True 2nd neighbor is at distance 5; a different vector also at 5 is just as good.
  const std::vector<float> kth{5.0F};
  const std::vector<std::vector<Neighbor>> results{with_distances({1.0F, 5.0F})};
  EXPECT_DOUBLE_EQ(*recall_at_k_with_ties(results, kth, 2), 1.0);
}

TEST(RecallWithTies, FartherResultIsAMiss) {
  const std::vector<float> kth{5.0F, 2.0F};
  const std::vector<std::vector<Neighbor>> results{with_distances({1.0F, 6.0F}),
                                                   with_distances({2.0F, 2.0F})};
  EXPECT_DOUBLE_EQ(*recall_at_k_with_ties(results, kth, 2), 3.0 / 4.0);
}

TEST(RecallWithTies, ToleranceIsRelative) {
  const std::vector<float> kth{40000.0F};
  const std::vector<std::vector<Neighbor>> close{with_distances({40000.2F})};
  const std::vector<std::vector<Neighbor>> far{with_distances({40001.0F})};
  EXPECT_DOUBLE_EQ(*recall_at_k_with_ties(close, kth, 1), 1.0);
  EXPECT_DOUBLE_EQ(*recall_at_k_with_ties(far, kth, 1), 0.0);
}

TEST(RecallWithTies, ShortResultListsAndInvalidArguments) {
  const std::vector<float> kth{5.0F};
  const std::vector<std::vector<Neighbor>> results{with_distances({1.0F})};
  EXPECT_DOUBLE_EQ(*recall_at_k_with_ties(results, kth, 4), 0.25);
  EXPECT_FALSE(recall_at_k_with_ties(results, kth, 0).has_value());
  EXPECT_FALSE(recall_at_k_with_ties(results, std::vector<float>{1.0F, 2.0F}, 1).has_value());
}

TEST(KthNeighborDistances, ComputesDistanceToKthTrueNeighbor) {
  const Matrix<float> base(3, 1, {0.0F, 10.0F, 3.0F});
  const Matrix<float> queries(1, 1, {1.0F});
  const auto gt = groundtruth({0, 2, 1}, 1, 3);
  auto d = kth_neighbor_distances(base, queries, gt, Metric::kL2, 2);
  ASSERT_TRUE(d.has_value());
  EXPECT_FLOAT_EQ((*d)[0], 4.0F);  // id 2 at 3.0, squared distance from 1.0
}

TEST(KthNeighborDistances, RejectsBadInput) {
  const Matrix<float> base(2, 1, {0.0F, 1.0F});
  const Matrix<float> queries(1, 1, {0.0F});
  EXPECT_FALSE(
      kth_neighbor_distances(base, queries, groundtruth({0, 7}, 1, 2), Metric::kL2, 2).has_value());
  EXPECT_FALSE(
      kth_neighbor_distances(base, Matrix<float>(1, 2), groundtruth({0, 1}, 1, 2), Metric::kL2, 1)
          .has_value());
  EXPECT_FALSE(
      kth_neighbor_distances(base, queries, groundtruth({0, 1}, 1, 2), Metric::kL2, 3).has_value());
}

}  // namespace
}  // namespace strata
