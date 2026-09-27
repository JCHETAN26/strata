// Spec for HnswIndex (include/strata/hnsw.hpp). Compiled only once src/index/hnsw.cpp exists.

#include "strata/hnsw.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <ostream>
#include <set>
#include <thread>
#include <vector>

#include "strata/brute_force.hpp"
#include "strata/dataset.hpp"
#include "strata/recall.hpp"
#include "test_util.hpp"

namespace strata {

// Readable parameterized test names (see test_util.hpp). Outside the anonymous namespace so ADL
// finds it.
inline void PrintTo(NeighborSelection selection, std::ostream* os) {
  *os << (selection == NeighborSelection::kSimple ? "simple" : "heuristic");
}

namespace {

HnswIndex make_index(std::size_t dim, Metric metric = Metric::kL2, HnswParams params = {}) {
  auto index = HnswIndex::create(dim, metric, params);
  EXPECT_TRUE(index.has_value()) << index.error().message;
  return std::move(*index);
}

// Tie-aware recall@k of the index against brute force over the same data.
double measure_recall(const HnswIndex& index, const Matrix<float>& base,
                      const Matrix<float>& queries, std::size_t k, std::size_t ef) {
  auto exact = BruteForceIndex::create(base.cols(), index.metric());
  EXPECT_TRUE(exact && exact->add_batch(base));
  std::vector<std::vector<Neighbor>> results;
  std::vector<float> kth;
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto truth = exact->search(queries.row(q), k);
    auto found = index.search(queries.row(q), k, ef);
    EXPECT_TRUE(truth && found);
    kth.push_back(truth->back().distance);
    results.push_back(std::move(*found));
  }
  return *recall_at_k_with_ties(results, kth, k);
}

// Fraction of nodes reachable from the entry point along layer-0 edges. A node no search can
// reach can never be returned.
double reachable_fraction(const HnswIndex& index) {
  std::vector<bool> seen(index.size(), false);
  std::deque<VectorId> frontier{*index.entry_point()};
  seen[*index.entry_point()] = true;
  std::size_t reached = 1;
  while (!frontier.empty()) {
    const VectorId id = frontier.front();
    frontier.pop_front();
    for (VectorId nbr : index.neighbors(id, 0)) {
      if (!seen[nbr]) {
        seen[nbr] = true;
        ++reached;
        frontier.push_back(nbr);
      }
    }
  }
  return static_cast<double>(reached) / static_cast<double>(index.size());
}

// --- Construction and edge cases -----------------------------------------------------------------

TEST(Hnsw, RejectsInvalidParameters) {
  EXPECT_FALSE(HnswIndex::create(0, Metric::kL2).has_value());
  EXPECT_FALSE(HnswIndex::create(4, Metric::kL2, {.M = 1}).has_value());
  EXPECT_FALSE(HnswIndex::create(4, Metric::kL2, {.M = 16, .ef_construction = 0}).has_value());
}

TEST(Hnsw, EmptyIndex) {
  const auto index = make_index(3);
  EXPECT_EQ(index.size(), 0U);
  EXPECT_EQ(index.entry_point(), std::nullopt);
  EXPECT_EQ(index.max_level(), -1);
  auto result = index.search(std::vector<float>{1, 2, 3}, 10, 50);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->empty());
}

TEST(Hnsw, SingleVector) {
  auto index = make_index(2);
  ASSERT_EQ(index.add(std::vector<float>{3, 4}), VectorId{0});
  EXPECT_EQ(index.entry_point(), VectorId{0});
  EXPECT_EQ(index.max_level(), index.level(0));
  auto result = index.search(std::vector<float>{0, 0}, 5, 10);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), 1U);
  EXPECT_EQ((*result)[0].id, 0U);
  EXPECT_FLOAT_EQ((*result)[0].distance, 25.0F);
}

TEST(Hnsw, KZeroReturnsNothing) {
  auto index = make_index(4);
  ASSERT_TRUE(index.add_batch(test::random_matrix(10, 4, 1)));
  auto result = index.search(std::vector<float>{0, 0, 0, 0}, 0, 10);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->empty());
}

TEST(Hnsw, DimensionMismatchIsAnError) {
  auto index = make_index(3);
  auto added = index.add(std::vector<float>{1, 2});
  ASSERT_FALSE(added.has_value());
  EXPECT_EQ(added.error().code, ErrorCode::kDimensionMismatch);
  EXPECT_FALSE(index.add_batch(test::random_matrix(2, 5, 1)).has_value());
  EXPECT_EQ(index.size(), 0U);
  auto result = index.search(std::vector<float>{1, 2, 3, 4}, 1, 10);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code, ErrorCode::kDimensionMismatch);
}

TEST(Hnsw, IdsAreAssignedInInsertionOrder) {
  auto index = make_index(2);
  EXPECT_EQ(index.add(std::vector<float>{0, 0}), VectorId{0});
  ASSERT_TRUE(index.add_batch(test::random_matrix(3, 2, 1)));
  EXPECT_EQ(index.add(std::vector<float>{0, 0}), VectorId{4});
  EXPECT_EQ(index.size(), 5U);
  EXPECT_EQ(index.vector(4)[0], 0.0F);
}

// Re-adding vectors read from the index's own storage: the insert may reallocate that storage,
// so the index must copy first (ASan catches the use-after-free otherwise).
TEST(Hnsw, AddingItsOwnVectorsIsSafe) {
  auto index = make_index(4);
  ASSERT_TRUE(index.add_batch(test::random_matrix(3, 4, 1)));
  for (VectorId id = 0; id < 51; ++id) {
    ASSERT_TRUE(index.add(index.vector(id % 3)));
  }
  ASSERT_TRUE(index.add_batch(MatrixView<const float>(index.vector(0).data(), 3, 4)));
  ASSERT_EQ(index.size(), 57U);
  for (VectorId id = 3; id < index.size(); ++id) {
    EXPECT_TRUE(std::ranges::equal(index.vector(id), index.vector(id % 3))) << id;
  }
}

TEST(Hnsw, KLargerThanSizeReturnsEverything) {
  auto index = make_index(4);
  ASSERT_TRUE(index.add_batch(test::random_matrix(20, 4, 1)));
  auto result = index.search(std::vector<float>{0, 0, 0, 0}, 100, 100);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), 20U);
}

TEST(Hnsw, EfSmallerThanKStillReturnsK) {
  auto index = make_index(8);
  ASSERT_TRUE(index.add_batch(test::random_matrix(500, 8, 1)));
  auto result = index.search(test::random_matrix(1, 8, 2).row(0), 10, 1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), 10U);
}

// Many identical vectors: every candidate is at distance 0 from every other, which is exactly
// where a careless neighbor-selection heuristic disconnects the graph.
TEST(Hnsw, DuplicateVectors) {
  auto index = make_index(4, Metric::kL2, {.M = 8, .ef_construction = 50});
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(index.add(std::vector<float>{1, 2, 3, 4}));
  }
  auto result = index.search(std::vector<float>{1, 2, 3, 4}, 10, 50);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), 10U);
  std::set<VectorId> ids;
  for (const auto& n : *result) {
    EXPECT_EQ(n.distance, 0.0F);
    ids.insert(n.id);
  }
  EXPECT_EQ(ids.size(), 10U);
}

TEST(Hnsw, SameSeedSameResults) {
  const auto base = test::random_matrix(1000, 16, 3);
  const auto queries = test::random_matrix(20, 16, 4);
  auto a = make_index(16);
  auto b = make_index(16);
  ASSERT_TRUE(a.add_batch(base));
  ASSERT_TRUE(b.add_batch(base));
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    EXPECT_EQ(*a.search(queries.row(q), 10, 40), *b.search(queries.row(q), 10, 40));
  }
}

// --- Result contract ---------------------------------------------------------------------------

TEST(Hnsw, ResultsAreSortedDistinctAndExact) {
  const auto base = test::random_matrix(2000, 16, 5);
  const auto queries = test::random_matrix(50, 16, 6);
  auto index = make_index(16);
  ASSERT_TRUE(index.add_batch(base));
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto result = index.search(queries.row(q), 10, 64);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 10U);
    EXPECT_TRUE(std::is_sorted(result->begin(), result->end()));
    std::set<VectorId> ids;
    for (const auto& n : *result) {
      ASSERT_LT(n.id, index.size());
      ids.insert(n.id);
      EXPECT_EQ(n.distance, distance(Metric::kL2, queries.row(q), base.row(n.id)));
    }
    EXPECT_EQ(ids.size(), result->size());
  }
}

// Concurrent const searches share the index but not scratch state. Under the tsan preset this
// checks the read path is race-free; everywhere it checks results match a serial run.
TEST(Hnsw, ConcurrentSearchesMatchSerial) {
  const auto base = test::random_matrix(2000, 16, 20);
  const auto queries = test::random_matrix(64, 16, 21);
  auto index = make_index(16);
  ASSERT_TRUE(index.add_batch(base));
  std::vector<std::vector<Neighbor>> serial;
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    serial.push_back(*index.search(queries.row(q), 10, 50));
  }

  constexpr std::size_t kThreads = 4;
  std::vector<std::size_t> mismatches(kThreads, 0);
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (std::size_t q = 0; q < queries.rows(); ++q) {
        auto found = index.search(queries.row(q), 10, 50);
        mismatches[t] += (found && *found == serial[q]) ? 0 : 1;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (std::size_t t = 0; t < kThreads; ++t) {
    EXPECT_EQ(mismatches[t], 0U) << "thread " << t;
  }
}

// --- Graph invariants ----------------------------------------------------------------------------

class HnswGraph : public ::testing::Test {
 protected:
  static constexpr std::size_t kSize = 5000;
  static constexpr std::size_t kM = 16;

  void SetUp() override {
    index_.emplace(make_index(16, Metric::kL2, {.M = kM, .ef_construction = 100}));
    ASSERT_TRUE(index_->add_batch(test::random_matrix(kSize, 16, 7)));
  }

  std::optional<HnswIndex> index_;
};

TEST_F(HnswGraph, DegreeBounds) {
  for (VectorId id = 0; id < kSize; ++id) {
    for (int layer = 0; layer <= index_->level(id); ++layer) {
      const std::size_t bound = layer == 0 ? 2 * kM : kM;
      ASSERT_LE(index_->neighbors(id, layer).size(), bound) << "node " << id << " layer " << layer;
    }
  }
}

TEST_F(HnswGraph, NeighborListsAreWellFormed) {
  for (VectorId id = 0; id < kSize; ++id) {
    for (int layer = 0; layer <= index_->level(id); ++layer) {
      std::set<VectorId> seen;
      for (VectorId nbr : index_->neighbors(id, layer)) {
        ASSERT_LT(nbr, kSize);
        EXPECT_NE(nbr, id) << "self-loop at node " << id;
        EXPECT_TRUE(seen.insert(nbr).second) << "duplicate edge " << id << "->" << nbr;
        EXPECT_GE(index_->level(nbr), layer) << "neighbor " << nbr << " not on layer " << layer;
      }
    }
  }
}

TEST_F(HnswGraph, EntryPointIsOnTopLayer) {
  ASSERT_TRUE(index_->entry_point().has_value());
  EXPECT_EQ(index_->level(*index_->entry_point()), index_->max_level());
  for (VectorId id = 0; id < kSize; ++id) {
    ASSERT_LE(index_->level(id), index_->max_level());
  }
}

// P(level >= 1) = 1/M with mL = 1/ln(M). Expected 312.5 of 5000, sd ~17; allow +-5 sd.
TEST_F(HnswGraph, LevelDistributionMatchesMl) {
  std::size_t upper = 0;
  for (VectorId id = 0; id < kSize; ++id) {
    upper += index_->level(id) >= 1 ? 1 : 0;
  }
  EXPECT_GT(upper, 228U);
  EXPECT_LT(upper, 397U);
}

// Every node should be reachable from the entry point along layer-0 edges; otherwise search can
// never return it. Allow a tiny fraction for pruning artifacts.
TEST_F(HnswGraph, Layer0IsReachableFromEntryPoint) {
  EXPECT_GE(reachable_fraction(*index_), 0.999);
}

// --- Search quality ------------------------------------------------------------------------------

class HnswRecall : public ::testing::TestWithParam<Metric> {};

TEST_P(HnswRecall, HighRecallOnRandomData) {
  const auto base = test::random_matrix(3000, 32, 11);
  const auto queries = test::random_matrix(100, 32, 12);
  auto index = make_index(32, GetParam());
  ASSERT_TRUE(index.add_batch(base));
  EXPECT_GE(measure_recall(index, base, queries, 10, 128), 0.95);
}

INSTANTIATE_TEST_SUITE_P(AllMetrics, HnswRecall,
                         ::testing::Values(Metric::kL2, Metric::kInnerProduct, Metric::kCosine),
                         test::PrintedName{});

// Both selection modes must meet the spec on easy (uniform) data; the heuristic's advantage on
// clustered data is measured by bench/run_selection_comparison.py, not asserted here.
class HnswSelection : public ::testing::TestWithParam<NeighborSelection> {};

TEST_P(HnswSelection, RecallAndReachability) {
  const auto base = test::random_matrix(3000, 32, 18);
  const auto queries = test::random_matrix(100, 32, 19);
  auto index = make_index(32, Metric::kL2, {.selection = GetParam()});
  ASSERT_TRUE(index.add_batch(base));
  EXPECT_GE(measure_recall(index, base, queries, 10, 128), 0.95);
  EXPECT_GE(reachable_fraction(index), 0.999);
}

INSTANTIATE_TEST_SUITE_P(BothModes, HnswSelection,
                         ::testing::Values(NeighborSelection::kSimple,
                                           NeighborSelection::kHeuristic),
                         test::PrintedName{});

// The worked example in docs/explainers/hnsw.md, section 4. A tight cluster of three points on
// one side of the new node q = (0, 0) and a lone point on the other:
//
//   B(-3,0)            q(0,0)  A0(1,0) A1(1.1,0) A2(1.2,0)
//
// With M = 2, closest-M links q to A0 and A1 (both in the cluster). The heuristic keeps A0, then
// rejects A1 and A2 (each is closer to A0 than to q: 0.01 < 1.21), and keeps B (closer to q, 9,
// than to A0, 16).
TEST(Hnsw, HeuristicPrefersDiverseNeighbors) {
  const std::vector<std::vector<float>> points{
      {1.0F, 0.0F}, {1.1F, 0.0F}, {1.2F, 0.0F}, {-3.0F, 0.0F}, {0.0F, 0.0F}};
  auto links_of_q = [&](NeighborSelection selection) {
    auto index =
        make_index(2, Metric::kL2, {.M = 2, .ef_construction = 100, .selection = selection});
    for (const auto& point : points) {
      EXPECT_TRUE(index.add(point).has_value());
    }
    const auto links = index.neighbors(4, 0);
    return std::set<VectorId>(links.begin(), links.end());
  };
  EXPECT_EQ(links_of_q(NeighborSelection::kSimple), (std::set<VectorId>{0, 1}));
  EXPECT_EQ(links_of_q(NeighborSelection::kHeuristic), (std::set<VectorId>{0, 3}));
}

TEST(Hnsw, RecallImprovesWithEf) {
  const auto base = test::random_matrix(3000, 32, 13);
  const auto queries = test::random_matrix(100, 32, 14);
  auto index = make_index(32, Metric::kL2, {.M = 8, .ef_construction = 64});
  ASSERT_TRUE(index.add_batch(base));
  const double low = measure_recall(index, base, queries, 10, 10);
  const double high = measure_recall(index, base, queries, 10, 256);
  EXPECT_GT(high, low);
  EXPECT_GE(high, 0.95);
}

TEST(Hnsw, IncrementalInsertsAfterSearch) {
  const auto first = test::random_matrix(1000, 16, 15);
  const auto second = test::random_matrix(1000, 16, 16);
  auto index = make_index(16);
  ASSERT_TRUE(index.add_batch(first));
  ASSERT_TRUE(index.search(first.row(0), 10, 50).has_value());
  ASSERT_TRUE(index.add_batch(second));

  Matrix<float> all(2000, 16);
  std::ranges::copy(first.data(), all.data().begin());
  std::ranges::copy(second.data(), all.data().begin() + 1000 * 16);
  EXPECT_GE(measure_recall(index, all, test::random_matrix(100, 16, 17), 10, 100), 0.95);
}

TEST(Hnsw, Sift10kRecall) {
  const std::filesystem::path dir = std::filesystem::path(STRATA_DATA_DIR) / "siftsmall";
  if (!std::filesystem::exists(dir / "base.fbin")) {
    GTEST_SKIP() << "SIFT10K not found in " << dir;
  }
  auto dataset = load_dataset(dir);
  ASSERT_TRUE(dataset.has_value());
  auto index = make_index(dataset->base.cols());
  ASSERT_TRUE(index.add_batch(dataset->base));
  EXPECT_GE(measure_recall(index, dataset->base, dataset->query, 10, 100), 0.98);
}

}  // namespace
}  // namespace strata
