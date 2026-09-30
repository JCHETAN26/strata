// Filtered HNSW search: the graph strategy (filtered-out nodes traversed like tombstones), the
// exact pre-filter, and the automatic choice between them with its runtime fallback.

#include <gtest/gtest.h>

#include <set>
#include <thread>
#include <vector>

#include "strata/brute_force.hpp"
#include "strata/filter.hpp"
#include "strata/hnsw.hpp"
#include "test_util.hpp"

namespace strata {
namespace {

constexpr std::size_t kCount = 4000;
constexpr std::size_t kDim = 16;
constexpr std::size_t kK = 10;

// Deterministic, uniform, uncorrelated with the vectors: bucket = hash(id) % 1000, so
// `bucket < b` matches about b / 1000 of the ids.
std::int64_t bucket_of(std::size_t id) {
  std::uint64_t x = id + 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return static_cast<std::int64_t>((x ^ (x >> 31U)) % 1000);
}

class HnswFilter : public ::testing::Test {
 protected:
  void SetUp() override {
    index_.emplace(*HnswIndex::create(kDim, Metric::kL2, {.ef_construction = 100}));
    ASSERT_TRUE(index_->add_batch(base_));
    ASSERT_TRUE(exact_.add_batch(base_));
    table_.emplace(*AttributeTable::create({{"bucket", ColumnType::kInt}}));
    for (std::size_t id = 0; id < kCount; ++id) {
      ASSERT_TRUE(table_->append({bucket_of(id)}));
    }
  }

  // Filter matching about `per_mille` / 1000 of the ids.
  [[nodiscard]] CompiledFilter filter(std::int64_t per_mille) const {
    return *CompiledFilter::compile(Filter::range("bucket", 0, per_mille - 1), *table_);
  }
  void remove(VectorId id) {
    ASSERT_TRUE(index_->remove(id));
    ASSERT_TRUE(exact_.remove(id));
  }
  // Exact top-k over the matching live vectors.
  [[nodiscard]] std::vector<Neighbor> truth(std::size_t q, const CompiledFilter& f) const {
    return *exact_.search_filtered(queries_.row(q), kK, f.evaluate());
  }
  [[nodiscard]] Expected<std::vector<Neighbor>> run(std::size_t q, const CompiledFilter& f,
                                                    FilteredSearchOptions options,
                                                    FilteredSearchStats* stats = nullptr) const {
    return index_->search_filtered(queries_.row(q), kK, f, options, stats);
  }
  // Recall@k by id of a strategy against exact filtered search, over all queries.
  [[nodiscard]] double recall(const CompiledFilter& f, const FilteredSearchOptions& options) const {
    std::size_t hits = 0;
    std::size_t expected = 0;
    for (std::size_t q = 0; q < queries_.rows(); ++q) {
      std::set<VectorId> ids;
      for (const auto& n : truth(q, f)) {
        ids.insert(n.id);
      }
      expected += ids.size();
      const auto found = run(q, f, options);  // held: iterating *run(...) would dangle
      for (const auto& n : *found) {
        hits += ids.count(n.id);
      }
    }
    return expected == 0 ? 1.0 : static_cast<double>(hits) / static_cast<double>(expected);
  }
  // Every returned id matches, is live, and results are sorted with exact distances.
  ::testing::AssertionResult valid(std::size_t q, const CompiledFilter& f,
                                   const std::vector<Neighbor>& found) const {
    if (!std::ranges::is_sorted(found)) {
      return ::testing::AssertionFailure() << "unsorted";
    }
    for (const auto& n : found) {
      if (!f.matches(n.id) || index_->is_deleted(n.id)) {
        return ::testing::AssertionFailure() << "returned non-matching or deleted id " << n.id;
      }
      if (n.distance != distance(Metric::kL2, queries_.row(q), base_.row(n.id))) {
        return ::testing::AssertionFailure() << "inexact distance for " << n.id;
      }
    }
    return ::testing::AssertionSuccess();
  }

  Matrix<float> base_ = test::random_matrix(kCount, kDim, 90);
  Matrix<float> queries_ = test::random_matrix(60, kDim, 91);
  std::optional<HnswIndex> index_;
  BruteForceIndex exact_ = *BruteForceIndex::create(kDim, Metric::kL2);
  std::optional<AttributeTable> table_;
};

const FilteredSearchOptions kGraph{.ef_search = 200, .strategy = FilterStrategy::kGraph};
const FilteredSearchOptions kPre{.strategy = FilterStrategy::kPreFilter};

TEST_F(HnswFilter, GraphRecallAgainstBruteForceOverMatchingSet) {
  for (std::int64_t per_mille : {10, 100, 500}) {
    const auto f = filter(per_mille);
    EXPECT_GE(recall(f, kGraph), 0.95) << per_mille << " per mille";
  }
}

TEST_F(HnswFilter, PreFilterIsExact) {
  for (std::int64_t per_mille : {1, 10, 100, 500}) {
    const auto f = filter(per_mille);
    for (std::size_t q = 0; q < queries_.rows(); ++q) {
      ASSERT_EQ(*run(q, f, kPre), truth(q, f)) << per_mille << " per mille, query " << q;
    }
  }
}

TEST_F(HnswFilter, NeverReturnsANonMatchingId) {
  for (std::int64_t per_mille : {1, 10, 100, 500, 999}) {
    const auto f = filter(per_mille);
    for (const auto& options :
         {kGraph, kPre, FilteredSearchOptions{}, FilteredSearchOptions{.ef_search = 10}}) {
      for (std::size_t q = 0; q < queries_.rows(); ++q) {
        auto found = run(q, f, options);
        ASSERT_TRUE(found);
        ASSERT_TRUE(valid(q, f, *found)) << per_mille << " per mille, query " << q;
      }
    }
  }
}

TEST_F(HnswFilter, FilterMatchingNothing) {
  const auto none = *CompiledFilter::compile(Filter::range("bucket", 2000, 3000), *table_);
  for (const auto& options : {kGraph, kPre, FilteredSearchOptions{}}) {
    FilteredSearchStats stats;
    auto found = run(0, none, options, &stats);
    ASSERT_TRUE(found);
    EXPECT_TRUE(found->empty());
  }
  FilteredSearchStats stats;
  ASSERT_TRUE(run(0, none, {}, &stats));
  EXPECT_EQ(stats.used, FilterStrategy::kPreFilter);  // estimated 0: nothing to search for
}

// A filter matching everything must give exactly the unfiltered answers: the graph strategy is
// the same traversal (its widened stopping rule never differs when nothing is rejected), and the
// pre-filter is exact brute force.
TEST_F(HnswFilter, FilterMatchingEverything) {
  const auto all = filter(1000);
  for (std::size_t ef : {std::size_t{10}, std::size_t{64}, std::size_t{200}}) {
    const FilteredSearchOptions graph{.ef_search = ef, .strategy = FilterStrategy::kGraph};
    for (std::size_t q = 0; q < queries_.rows(); ++q) {
      ASSERT_EQ(*run(q, all, graph), *index_->search(queries_.row(q), kK, ef)) << "ef " << ef;
    }
  }
  for (std::size_t q = 0; q < queries_.rows(); ++q) {
    ASSERT_EQ(*run(q, all, kPre), *exact_.search(queries_.row(q), kK));
  }
}

TEST_F(HnswFilter, CombinedWithDeletes) {
  for (VectorId id = 0; id < kCount; id += 3) {
    remove(id);
  }
  const auto f = filter(100);
  EXPECT_GE(recall(f, kGraph), 0.95);
  for (std::size_t q = 0; q < queries_.rows(); ++q) {
    ASSERT_EQ(*run(q, f, kPre), truth(q, f));
    for (const auto& options : {kGraph, FilteredSearchOptions{}}) {
      auto found = run(q, f, options);
      ASSERT_TRUE(found);
      ASSERT_TRUE(valid(q, f, *found));
    }
  }
}

TEST_F(HnswFilter, FewerMatchesThanKReturnsAllOfThem) {
  const auto few = *CompiledFilter::compile(Filter::range("bucket", 0, 0), *table_);  // ~4 ids
  const std::size_t matches = few.evaluate().count();
  ASSERT_GT(matches, 0U);
  ASSERT_LT(matches, kK);
  for (const auto& options : {kGraph, kPre, FilteredSearchOptions{}}) {
    auto found = run(0, few, options);
    ASSERT_TRUE(found);
    EXPECT_EQ(*found, truth(0, few));
  }
}

// The choice by estimated selectivity alone (the fallback is tested separately; on an index this
// small it would rightly send many graph searches to the pre-filter, which costs little here).
TEST_F(HnswFilter, AutoPicksEachSideOfTheThreshold) {
  FilteredSearchStats stats;
  ASSERT_TRUE(run(0, filter(1), {.prefilter_below = 0.02, .fallback_budget = 0}, &stats));  // ~0.1%
  EXPECT_EQ(stats.used, FilterStrategy::kPreFilter);
  EXPECT_LT(stats.estimated_selectivity, 0.02);
  ASSERT_TRUE(
      run(0, filter(500), {.prefilter_below = 0.02, .fallback_budget = 0}, &stats));  // ~50%
  EXPECT_EQ(stats.used, FilterStrategy::kGraph);
  EXPECT_FALSE(stats.fell_back);
  EXPECT_NEAR(stats.estimated_selectivity, 0.5, 0.06);
  // The threshold is per call.
  ASSERT_TRUE(run(0, filter(500), {.prefilter_below = 0.9, .fallback_budget = 0}, &stats));
  EXPECT_EQ(stats.used, FilterStrategy::kPreFilter);
}

// Near the threshold, the first 1000-id sample cannot decide, so a larger one is taken.
TEST_F(HnswFilter, ResamplesNearTheThreshold) {
  FilteredSearchStats stats;
  ASSERT_TRUE(run(0, filter(20), {.prefilter_below = 0.02, .fallback_budget = 0},
                  &stats));  // ~2%, at the threshold
  EXPECT_TRUE(stats.resampled);
  ASSERT_TRUE(run(0, filter(500), {.prefilter_below = 0.02, .fallback_budget = 0},
                  &stats));  // far above it
  EXPECT_FALSE(stats.resampled);
}

// Force the graph (threshold 0) on a very selective filter with a small budget: the search gives
// up and the pre-filter answers, exactly. With the fallback off it stays on the graph.
TEST_F(HnswFilter, FallbackToPreFilterWhenTheGraphSearchIsTooLong) {
  const auto rare = filter(2);  // ~0.2%: the graph must expand most of the index to find ef
  FilteredSearchStats stats;
  const FilteredSearchOptions tight{.prefilter_below = 0.0, .fallback_budget = 0.05};
  for (std::size_t q = 0; q < 10; ++q) {
    auto found = run(q, rare, tight, &stats);
    ASSERT_TRUE(found);
    EXPECT_TRUE(stats.fell_back) << "query " << q;
    EXPECT_EQ(stats.used, FilterStrategy::kPreFilter);
    EXPECT_EQ(*found, truth(q, rare));
  }
  ASSERT_TRUE(run(0, rare, {.prefilter_below = 0.0, .fallback_budget = 0.0}, &stats));
  EXPECT_FALSE(stats.fell_back);
  EXPECT_EQ(stats.used, FilterStrategy::kGraph);
  // A forced graph search never falls back, whatever the budget.
  ASSERT_TRUE(run(0, rare, {.strategy = FilterStrategy::kGraph, .fallback_budget = 0.01}, &stats));
  EXPECT_FALSE(stats.fell_back);
}

// A caller that knows the selectivity passes it, and auto uses it instead of estimating.
TEST_F(HnswFilter, KnownSelectivityIsUsedInsteadOfAnEstimate) {
  FilteredSearchStats stats;
  const auto half = filter(500);
  ASSERT_TRUE(run(0, half, {.fallback_budget = 0, .selectivity = 0.001}, &stats));
  EXPECT_EQ(stats.used, FilterStrategy::kPreFilter);  // told it is rare
  EXPECT_DOUBLE_EQ(stats.estimated_selectivity, 0.001);
  EXPECT_FALSE(stats.resampled);
  ASSERT_TRUE(run(0, half, {.fallback_budget = 0, .selectivity = 0.5}, &stats));
  EXPECT_EQ(stats.used, FilterStrategy::kGraph);
}

// The auto estimate is cached on the filter: repeated queries, and copies of the filter, reuse it.
TEST_F(HnswFilter, SelectivityEstimateIsCachedPerFilter) {
  const auto f = filter(20);  // near the default threshold, so the precise sample is used too
  const double coarse = f.coarse_selectivity();
  const double precise = f.precise_selectivity();
  EXPECT_EQ(coarse, f.estimate_selectivity(CompiledFilter::kCoarseSamples, 1));
  EXPECT_EQ(precise, f.estimate_selectivity(CompiledFilter::kPreciseSamples, 2));
  const CompiledFilter copy = f;  // NOLINT(performance-unnecessary-copy-initialization)
  EXPECT_EQ(copy.coarse_selectivity(), coarse);
  FilteredSearchStats first;
  FilteredSearchStats second;
  ASSERT_TRUE(run(0, f, {.prefilter_below = 0.02, .fallback_budget = 0}, &first));
  ASSERT_TRUE(run(1, copy, {.prefilter_below = 0.02, .fallback_budget = 0}, &second));
  EXPECT_EQ(first.estimated_selectivity, second.estimated_selectivity);
  EXPECT_EQ(first.used, second.used);
}

// Many threads asking one fresh filter for its estimate at once all get the same value (TSan
// checks the call_once).
TEST_F(HnswFilter, ConcurrentFirstEstimatesAgree) {
  const auto f = filter(30);
  std::vector<double> seen(8, -1);
  std::vector<std::thread> threads;
  threads.reserve(seen.size());
  for (std::size_t t = 0; t < seen.size(); ++t) {
    threads.emplace_back([&, t] { seen[t] = f.precise_selectivity(); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (double v : seen) {
    EXPECT_EQ(v, seen[0]);
  }
}

TEST_F(HnswFilter, BitsetFormAgreesWithCompiledFilter) {
  const auto f = filter(100);
  const Bitset bits = f.evaluate();
  for (const auto& options : {kGraph, kPre}) {
    for (std::size_t q = 0; q < queries_.rows(); ++q) {
      ASSERT_EQ(*index_->search_filtered(queries_.row(q), kK, bits, options), *run(q, f, options));
    }
  }
  FilteredSearchStats stats;
  ASSERT_TRUE(index_->search_filtered(queries_.row(0), kK, bits, {}, &stats));
  EXPECT_DOUBLE_EQ(stats.estimated_selectivity,
                   static_cast<double>(bits.count()) / static_cast<double>(kCount));
}

TEST_F(HnswFilter, Errors) {
  const auto f = filter(100);
  auto wrong_dim = index_->search_filtered(std::vector<float>(kDim + 1), kK, f);
  ASSERT_FALSE(wrong_dim);
  EXPECT_EQ(wrong_dim.error().code, ErrorCode::kDimensionMismatch);
  auto wrong_rows = index_->search_filtered(queries_.row(0), kK, Bitset(kCount + 1));
  ASSERT_FALSE(wrong_rows);
  EXPECT_EQ(wrong_rows.error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE(index_->search_filtered(queries_.row(0), 0, f));
  EXPECT_TRUE(index_->search_filtered(queries_.row(0), 0, f)->empty());
}

// Const filtered searches from several threads (for TSan) agree with a serial run.
TEST_F(HnswFilter, ConcurrentFilteredSearchesMatchSerial) {
  const auto f = filter(50);
  std::vector<std::vector<Neighbor>> serial;
  for (std::size_t q = 0; q < queries_.rows(); ++q) {
    serial.push_back(*run(q, f, {}));
  }
  std::vector<std::size_t> mismatches(4, 0);
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      for (std::size_t q = 0; q < queries_.rows(); ++q) {
        auto found = run(q, f, {});
        mismatches[t] += (found && *found == serial[q]) ? 0 : 1;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (std::size_t t = 0; t < 4; ++t) {
    EXPECT_EQ(mismatches[t], 0U) << "thread " << t;
  }
}

}  // namespace
}  // namespace strata
