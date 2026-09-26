#include "strata/brute_force.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <vector>

#include "strata/dataset.hpp"
#include "strata/recall.hpp"
#include "test_util.hpp"

namespace strata {
namespace {

BruteForceIndex make_index(std::size_t dim, Metric metric = Metric::kL2) {
  auto index = BruteForceIndex::create(dim, metric);
  EXPECT_TRUE(index.has_value());
  return std::move(*index);
}

TEST(BruteForce, RejectsZeroDimension) {
  auto index = BruteForceIndex::create(0, Metric::kL2);
  ASSERT_FALSE(index.has_value());
  EXPECT_EQ(index.error().code, ErrorCode::kInvalidArgument);
}

TEST(BruteForce, EmptyIndexReturnsNoResults) {
  const auto index = make_index(3);
  const std::vector<float> q{1, 2, 3};
  auto result = index.search(q, 10);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->empty());
}

TEST(BruteForce, SingleVector) {
  auto index = make_index(2);
  ASSERT_EQ(index.add(std::vector<float>{3, 4}), VectorId{0});
  auto result = index.search(std::vector<float>{0, 0}, 5);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), 1U);
  EXPECT_EQ((*result)[0].id, 0U);
  EXPECT_FLOAT_EQ((*result)[0].distance, 25.0F);
}

TEST(BruteForce, KZeroReturnsNoResults) {
  auto index = make_index(2);
  ASSERT_TRUE(index.add(std::vector<float>{1, 1}));
  auto result = index.search(std::vector<float>{0, 0}, 0);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->empty());
}

TEST(BruteForce, KLargerThanSizeReturnsEverything) {
  auto index = make_index(4);
  ASSERT_TRUE(index.add_batch(test::random_matrix(7, 4, 1)));
  auto result = index.search(std::vector<float>{0, 0, 0, 0}, 100);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), 7U);
}

TEST(BruteForce, DuplicateVectorsTieBreakById) {
  auto index = make_index(2);
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(index.add(std::vector<float>{1, 1}));
  }
  ASSERT_TRUE(index.add(std::vector<float>{0, 0}));
  auto result = index.search(std::vector<float>{1, 1}, 3);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), 3U);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ((*result)[i].id, i);
    EXPECT_EQ((*result)[i].distance, 0.0F);
  }
}

TEST(BruteForce, DimensionMismatchIsAnError) {
  auto index = make_index(3);
  auto added = index.add(std::vector<float>{1, 2});
  ASSERT_FALSE(added.has_value());
  EXPECT_EQ(added.error().code, ErrorCode::kDimensionMismatch);

  auto batch = index.add_batch(test::random_matrix(2, 5, 1));
  ASSERT_FALSE(batch.has_value());
  EXPECT_EQ(batch.error().code, ErrorCode::kDimensionMismatch);
  EXPECT_EQ(index.size(), 0U);

  auto result = index.search(std::vector<float>{1, 2, 3, 4}, 1);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code, ErrorCode::kDimensionMismatch);
}

TEST(BruteForce, IdsAreAssignedInInsertionOrder) {
  auto index = make_index(2);
  EXPECT_EQ(index.add(std::vector<float>{0, 0}), VectorId{0});
  ASSERT_TRUE(index.add_batch(test::random_matrix(3, 2, 1)));
  EXPECT_EQ(index.add(std::vector<float>{0, 0}), VectorId{4});
  EXPECT_EQ(index.size(), 5U);
}

// Compare against a full sort of every distance, for each metric.
class BruteForceMetric : public ::testing::TestWithParam<Metric> {};

TEST_P(BruteForceMetric, MatchesFullSort) {
  const Metric metric = GetParam();
  const auto base = test::random_matrix(500, 16, 42);
  const auto queries = test::random_matrix(20, 16, 7);
  auto index = make_index(16, metric);
  ASSERT_TRUE(index.add_batch(base));

  for (std::size_t q = 0; q < queries.rows(); ++q) {
    std::vector<Neighbor> expected;
    for (std::size_t i = 0; i < base.rows(); ++i) {
      expected.push_back({static_cast<VectorId>(i), distance(metric, queries.row(q), base.row(i))});
    }
    std::sort(expected.begin(), expected.end());
    expected.resize(10);

    auto result = index.search(queries.row(q), 10);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, expected);
  }
}

INSTANTIATE_TEST_SUITE_P(AllMetrics, BruteForceMetric,
                         ::testing::Values(Metric::kL2, Metric::kInnerProduct, Metric::kCosine),
                         [](const auto& info) { return std::string(to_string(info.param)); });

// End-to-end check against the published SIFT10K ground truth. Skips if the dataset is absent
// (run `uv run python scripts/prepare_datasets.py siftsmall`).
TEST(BruteForce, Sift10kRecallIsPerfect) {
  const std::filesystem::path dir = std::filesystem::path(STRATA_DATA_DIR) / "siftsmall";
  if (!std::filesystem::exists(dir / "base.fbin")) {
    GTEST_SKIP() << "SIFT10K not found in " << dir;
  }
  auto dataset = load_dataset(dir);
  ASSERT_TRUE(dataset.has_value()) << dataset.error().message;

  auto index = make_index(dataset->base.cols());
  ASSERT_TRUE(index.add_batch(dataset->base));

  std::vector<std::vector<Neighbor>> results;
  for (std::size_t q = 0; q < dataset->query.rows(); ++q) {
    auto r = index.search(dataset->query.row(q), 10);
    ASSERT_TRUE(r.has_value());
    results.push_back(std::move(*r));
  }
  auto recall = recall_at_k(results, dataset->groundtruth, 10);
  ASSERT_TRUE(recall.has_value());
  EXPECT_DOUBLE_EQ(*recall, 1.0);
}

}  // namespace
}  // namespace strata
