#include "strata/pq.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <random>
#include <set>
#include <vector>

#include "strata/brute_force.hpp"
#include "strata/dataset.hpp"
#include "strata/kmeans.hpp"
#include "strata/recall.hpp"
#include "test_util.hpp"

namespace strata {
namespace {

// Points around `k` well-separated centers (center c at (10c, 10c, ...)).
Matrix<float> clustered(std::size_t k, std::size_t per_cluster, std::size_t dim,
                        std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.0F, 0.1F);
  Matrix<float> m(k * per_cluster, dim);
  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t p = 0; p < per_cluster; ++p) {
      for (float& x : m.row(c * per_cluster + p)) {
        x = 10.0F * static_cast<float>(c) + noise(rng);
      }
    }
  }
  return m;
}

// --- k-means
// ---------------------------------------------------------------------------------------

TEST(KMeans, RecoversSeparatedClusters) {
  const auto data = clustered(5, 40, 3, 1);
  auto result = kmeans(data, {.k = 5, .max_iterations = 50, .seed = 3});
  ASSERT_TRUE(result);
  // Every true cluster maps to exactly one k-means cluster.
  std::set<std::uint32_t> labels;
  for (std::size_t c = 0; c < 5; ++c) {
    const auto label = result->assignment[c * 40];
    labels.insert(label);
    for (std::size_t p = 0; p < 40; ++p) {
      ASSERT_EQ(result->assignment[c * 40 + p], label);
    }
  }
  EXPECT_EQ(labels.size(), 5U);
  EXPECT_LT(result->inertia, 200 * 3 * 0.1 * 0.1 * 2);  // ~ noise variance only
}

TEST(KMeans, InertiaDoesNotIncreaseWithMoreIterations) {
  const auto data = test::random_matrix(2000, 8, 2);
  double previous = 1e30;
  for (std::size_t iters : {1U, 2U, 5U, 20U}) {
    auto r = kmeans(data, {.k = 32, .max_iterations = iters, .seed = 5, .tolerance = 0});
    ASSERT_TRUE(r);
    EXPECT_LE(r->inertia, previous * (1 + 1e-9)) << iters << " iterations";
    previous = r->inertia;
  }
}

TEST(KMeans, DeterministicForSeed) {
  const auto data = test::random_matrix(500, 4, 3);
  auto a = kmeans(data, {.k = 16, .seed = 9});
  auto b = kmeans(data, {.k = 16, .seed = 9});
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->assignment, b->assignment);
}

TEST(KMeans, ParallelMatchesSerial) {
  const auto data = test::random_matrix(3000, 8, 4);
  ThreadPool pool(4);
  auto serial = kmeans(data, {.k = 20, .seed = 1});
  auto parallel = kmeans(data, {.k = 20, .seed = 1}, &pool);
  ASSERT_TRUE(serial && parallel);
  EXPECT_EQ(serial->assignment, parallel->assignment);
}

TEST(KMeans, NoEmptyClustersWithEnoughDistinctPoints) {
  // Heavily duplicated data: 3 distinct points x 100 copies, plus 20 distinct points.
  Matrix<float> data(320, 2);
  for (std::size_t i = 0; i < 300; ++i) {
    data.row(i)[0] = static_cast<float>(i % 3);
  }
  for (std::size_t i = 300; i < 320; ++i) {
    data.row(i)[0] = 100.0F + static_cast<float>(i);
    data.row(i)[1] = 7.0F;
  }
  auto r = kmeans(data, {.k = 23, .max_iterations = 30, .seed = 2});
  ASSERT_TRUE(r);
  std::vector<int> counts(23, 0);
  for (auto a : r->assignment) {
    ++counts[a];
  }
  for (std::size_t c = 0; c < counts.size(); ++c) {
    EXPECT_GT(counts[c], 0) << "cluster " << c << " is empty";
  }
}

TEST(KMeans, InvalidArguments) {
  EXPECT_FALSE(kmeans(Matrix<float>(0, 3), {.k = 1}));
  EXPECT_FALSE(kmeans(test::random_matrix(5, 3, 1), {.k = 0}));
  EXPECT_FALSE(kmeans(test::random_matrix(5, 3, 1), {.k = 6}));
}

// --- Product quantizer
// -----------------------------------------------------------------------------

TEST(ProductQuantizer, RejectsBadShapes) {
  const auto data = test::random_matrix(300, 10, 1);
  EXPECT_FALSE(ProductQuantizer::train(data, Metric::kL2, {.m = 3}));  // 10 % 3 != 0
  EXPECT_FALSE(ProductQuantizer::train(data, Metric::kL2, {.m = 0}));
  EXPECT_FALSE(ProductQuantizer::train(test::random_matrix(100, 10, 1), Metric::kL2, {.m = 5}));
}

// With 256 centroids per subspace and <= 256 distinct training points, every point is its own
// centroid, so encode -> decode is lossless.
TEST(ProductQuantizer, LosslessWhenPointsFitInCodebook) {
  const auto data = test::random_matrix(256, 8, 2);
  auto pq = ProductQuantizer::train(data, Metric::kL2, {.m = 4, .kmeans_iterations = 50});
  ASSERT_TRUE(pq);
  std::vector<std::uint8_t> code(4);
  std::vector<float> decoded(8);
  for (std::size_t i = 0; i < data.rows(); ++i) {
    pq->encode(data.row(i), code);
    pq->decode(code, decoded);
    for (std::size_t j = 0; j < 8; ++j) {
      ASSERT_FLOAT_EQ(decoded[j], data.row(i)[j]) << "row " << i;
    }
  }
}

TEST(ProductQuantizer, MoreSubquantizersReduceError) {
  const auto data = test::random_matrix(3000, 32, 3);
  double previous = 1e30;
  for (std::size_t m : {2U, 4U, 8U, 16U}) {
    auto pq = ProductQuantizer::train(data, Metric::kL2, {.m = m, .kmeans_iterations = 10});
    ASSERT_TRUE(pq);
    std::vector<std::uint8_t> code(m);
    std::vector<float> decoded(32);
    double error = 0;
    for (std::size_t i = 0; i < data.rows(); ++i) {
      pq->encode(data.row(i), code);
      pq->decode(code, decoded);
      error += scalar::l2_squared(data.row(i), decoded);
    }
    EXPECT_LT(error, previous) << "m = " << m;
    previous = error;
  }
}

// ADC distance must equal the exact distance between the query and the decoded vector.
class PqAdc : public ::testing::TestWithParam<Metric> {};

TEST_P(PqAdc, TableDistanceEqualsDistanceToReconstruction) {
  const Metric metric = GetParam();
  const auto data = test::random_matrix(1000, 16, 4);
  const auto queries = test::random_matrix(20, 16, 5);
  auto pq = ProductQuantizer::train(data, metric, {.m = 4, .kmeans_iterations = 5});
  ASSERT_TRUE(pq);
  std::vector<float> table(4 * 256);
  std::vector<std::uint8_t> code(4);
  std::vector<float> decoded(16);
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    pq->compute_table(queries.row(q), table);
    // For cosine, ADC compares the normalized query with the (normalized-space) reconstruction.
    std::vector<float> query(queries.row(q).begin(), queries.row(q).end());
    if (metric == Metric::kCosine) {
      float norm = 0;
      for (float x : query) {
        norm += x * x;
      }
      for (float& x : query) {
        x /= std::sqrt(norm);
      }
    }
    for (std::size_t i = 0; i < 50; ++i) {
      pq->encode(data.row(i), code);
      pq->decode(code, decoded);
      const float want = metric == Metric::kCosine
                             ? 1.0F - scalar::inner_product(query, decoded) * -1.0F
                             : distance(metric, query, decoded);
      EXPECT_NEAR(pq->table_distance(table, code), want, 1e-4F * std::max(1.0F, std::abs(want)));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllMetrics, PqAdc,
                         ::testing::Values(Metric::kL2, Metric::kInnerProduct, Metric::kCosine),
                         [](const auto& info) { return std::string(to_string(info.param)); });

// --- PQ index
// -------------------------------------------------------------------------------------

TEST(PqIndex, EmptyAndEdgeCases) {
  auto pq = ProductQuantizer::train(test::random_matrix(300, 8, 1), Metric::kL2, {.m = 2});
  ASSERT_TRUE(pq);
  auto index = PqIndex::create(*pq, /*keep_originals=*/false);
  ASSERT_TRUE(index);
  auto empty = index->search(std::vector<float>(8, 0.0F), 5);
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->empty());
  EXPECT_FALSE(index->search(std::vector<float>(7, 0.0F), 5));
  EXPECT_FALSE(index->add_batch(test::random_matrix(3, 9, 1)));
  ASSERT_TRUE(index->add_batch(test::random_matrix(3, 8, 1)));
  EXPECT_EQ(index->search(std::vector<float>(8, 0.0F), 10)->size(), 3U);
  // Reranking needs the originals.
  EXPECT_FALSE(index->search(std::vector<float>(8, 0.0F), 2, {.rerank = 10}));
}

TEST(PqIndex, MemoryAccounting) {
  auto pq = ProductQuantizer::train(test::random_matrix(500, 16, 1), Metric::kL2, {.m = 8});
  ASSERT_TRUE(pq);
  auto index = PqIndex::create(*pq, true);
  ASSERT_TRUE(index && index->add_batch(test::random_matrix(100, 16, 2)));
  EXPECT_EQ(index->code_bytes(), 100U * 8U);
  EXPECT_EQ(index->original_bytes(), 100U * 16U * sizeof(float));
  EXPECT_EQ(index->codebook_bytes(), 8U * 256U * 2U * sizeof(float));
}

TEST(PqIndex, ParallelAddMatchesSerial) {
  const auto data = test::random_matrix(1000, 16, 3);
  auto pq = ProductQuantizer::train(data, Metric::kL2, {.m = 4, .kmeans_iterations = 5});
  ASSERT_TRUE(pq);
  auto serial = PqIndex::create(*pq, false);
  auto parallel = PqIndex::create(*pq, false);
  ThreadPool pool(4);
  ASSERT_TRUE(serial->add_batch(data));
  ASSERT_TRUE(parallel->add_batch(data, &pool));
  const auto q = test::random_matrix(1, 16, 4);
  EXPECT_EQ(*serial->search(q.row(0), 10), *parallel->search(q.row(0), 10));
}

TEST(PqIndex, RerankReturnsExactDistancesAndImprovesRecall) {
  const std::filesystem::path dir = std::filesystem::path(STRATA_DATA_DIR) / "siftsmall";
  if (!std::filesystem::exists(dir / "base.fbin")) {
    GTEST_SKIP() << "SIFT10K not found";
  }
  auto ds = load_dataset(dir);
  ASSERT_TRUE(ds);
  auto pq = ProductQuantizer::train(ds->base, Metric::kL2, {.m = 16, .kmeans_iterations = 10});
  ASSERT_TRUE(pq);
  auto index = PqIndex::create(*pq, true);
  ASSERT_TRUE(index && index->add_batch(ds->base));
  auto kth = kth_neighbor_distances(ds->base, ds->query, ds->groundtruth, Metric::kL2, 10);
  ASSERT_TRUE(kth);

  auto recall_for = [&](std::size_t rerank) {
    std::vector<std::vector<Neighbor>> results;
    for (std::size_t q = 0; q < ds->query.rows(); ++q) {
      auto r = index->search(ds->query.row(q), 10, {.rerank = rerank});
      EXPECT_TRUE(r);
      // Recompute exact distances so tie-aware recall is meaningful for ADC results too.
      for (auto& n : *r) {
        n.distance = scalar::l2_squared(ds->query.row(q), ds->base.row(n.id));
      }
      results.push_back(*r);
    }
    return *recall_at_k_with_ties(results, *kth, 10);
  };
  const double adc_only = recall_for(0);
  const double reranked = recall_for(100);
  EXPECT_GT(adc_only, 0.3);  // 16 bytes/vector vs 512: approximate but useful
  EXPECT_GT(reranked, adc_only);
  EXPECT_GE(reranked, 0.9);

  auto r = index->search(ds->query.row(0), 5, {.rerank = 50});
  ASSERT_TRUE(r);
  for (const auto& n : *r) {
    EXPECT_EQ(n.distance, distance(Metric::kL2, ds->query.row(0), ds->base.row(n.id)));
  }
}

}  // namespace
}  // namespace strata
