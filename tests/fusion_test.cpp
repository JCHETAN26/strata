#include "strata/fusion.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace strata {
namespace {

using List = std::vector<Neighbor>;

// A ranked list from ids in order (distances only need to be increasing).
List ranked(std::initializer_list<VectorId> ids) {
  List out;
  float d = 0.0F;
  for (VectorId id : ids) {
    out.push_back({id, d});
    d += 1.0F;
  }
  return out;
}

float score(const Neighbor& n) { return -n.distance; }

TEST(Rrf, HandComputedExample) {
  const std::vector<List> lists = {ranked({1, 2, 3}), ranked({3, 1, 4})};
  auto fused = reciprocal_rank_fusion(lists, 10, 60.0);
  ASSERT_TRUE(fused);
  // 1: 1/61 + 1/62; 3: 1/63 + 1/61; 2: 1/62; 4: 1/63
  ASSERT_EQ(fused->size(), 4U);
  EXPECT_EQ((*fused)[0].id, 1U);
  EXPECT_EQ((*fused)[1].id, 3U);
  EXPECT_EQ((*fused)[2].id, 2U);
  EXPECT_EQ((*fused)[3].id, 4U);
  EXPECT_FLOAT_EQ(score((*fused)[0]), static_cast<float>(1.0 / 61 + 1.0 / 62));
  EXPECT_FLOAT_EQ(score((*fused)[3]), static_cast<float>(1.0 / 63));
}

TEST(Rrf, IgnoresScoreScalesOnlyRanksMatter) {
  List a = {{5, -1000.0F}, {6, 3.0F}};  // wildly different distance scales
  List b = {{5, 0.001F}, {6, 0.002F}};
  const std::vector<List> lists = {a, b};
  auto fused = reciprocal_rank_fusion(lists, 2);
  ASSERT_TRUE(fused);
  EXPECT_EQ((*fused)[0].id, 5U);
  EXPECT_FLOAT_EQ(score((*fused)[0]), static_cast<float>(2.0 / 61));
}

TEST(Rrf, TiesBreakByIdAndTopKTruncates) {
  const std::vector<List> lists = {ranked({9, 2}), ranked({2, 9})};  // 2 and 9 tie exactly
  auto fused = reciprocal_rank_fusion(lists, 1);
  ASSERT_TRUE(fused);
  ASSERT_EQ(fused->size(), 1U);
  EXPECT_EQ((*fused)[0].id, 2U);
}

TEST(Rrf, EdgeCases) {
  EXPECT_TRUE(reciprocal_rank_fusion(std::vector<List>{}, 5)->empty());
  EXPECT_TRUE(reciprocal_rank_fusion(std::vector<List>{List{}, List{}}, 5)->empty());
  EXPECT_TRUE(reciprocal_rank_fusion(std::vector<List>{ranked({1})}, 0)->empty());
  EXPECT_FALSE(reciprocal_rank_fusion(std::vector<List>{ranked({1})}, 5, -1.0));
  // k = 0 is allowed: score 1/rank.
  auto fused = reciprocal_rank_fusion(std::vector<List>{ranked({7})}, 5, 0.0);
  ASSERT_TRUE(fused);
  EXPECT_FLOAT_EQ(score((*fused)[0]), 1.0F);
}

TEST(WeightedFusion, MinMaxNormalizationAndWeights) {
  // Dense: similarities 0.9, 0.5, 0.1 -> norms 1, 0.5, 0. BM25: scores 20, 10 -> norms 1, 0.
  const List dense = {{1, -0.9F}, {2, -0.5F}, {3, -0.1F}};
  const List bm25 = {{3, -20.0F}, {4, -10.0F}};
  const std::vector<List> lists = {dense, bm25};
  const std::vector<double> weights = {0.7, 0.3};
  auto fused = weighted_score_fusion(lists, weights, 10);
  ASSERT_TRUE(fused);
  ASSERT_EQ(fused->size(), 4U);
  // 1: 0.7; 2: 0.35; 3: 0 + 0.3 = 0.3; 4: 0.
  EXPECT_EQ((*fused)[0].id, 1U);
  EXPECT_NEAR(score((*fused)[0]), 0.7, 1e-6);
  EXPECT_EQ((*fused)[1].id, 2U);
  EXPECT_NEAR(score((*fused)[1]), 0.35, 1e-6);
  EXPECT_EQ((*fused)[2].id, 3U);
  EXPECT_NEAR(score((*fused)[2]), 0.3, 1e-6);
  EXPECT_EQ((*fused)[3].id, 4U);
}

TEST(WeightedFusion, ExtremeWeightsReproduceOneList) {
  const List dense = {{1, 0.1F}, {2, 0.2F}, {3, 0.3F}};
  const List bm25 = {{3, -5.0F}, {2, -4.0F}, {1, -3.0F}};
  const std::vector<List> lists = {dense, bm25};
  auto only_dense = weighted_score_fusion(lists, std::vector<double>{1.0, 0.0}, 3);
  auto only_bm25 = weighted_score_fusion(lists, std::vector<double>{0.0, 1.0}, 3);
  ASSERT_TRUE(only_dense && only_bm25);
  EXPECT_EQ((*only_dense)[0].id, 1U);
  EXPECT_EQ((*only_bm25)[0].id, 3U);
}

TEST(WeightedFusion, ConstantListNormalizesToOne) {
  const std::vector<List> lists = {List{{4, 2.0F}, {5, 2.0F}}};
  auto fused = weighted_score_fusion(lists, std::vector<double>{0.5}, 5);
  ASSERT_TRUE(fused);
  EXPECT_FLOAT_EQ(score((*fused)[0]), 0.5F);
  EXPECT_FLOAT_EQ(score((*fused)[1]), 0.5F);
}

TEST(WeightedFusion, RejectsBadWeights) {
  const std::vector<List> lists = {ranked({1}), ranked({2})};
  EXPECT_FALSE(weighted_score_fusion(lists, std::vector<double>{1.0}, 5));
  EXPECT_FALSE(weighted_score_fusion(lists, std::vector<double>{1.0, -0.1}, 5));
  EXPECT_FALSE(weighted_score_fusion(lists, std::vector<double>{1.0, std::nan("")}, 5));
}

}  // namespace
}  // namespace strata
