// Parallel HNSW build (add_batch with a ThreadPool). A parallel graph depends on thread timing, so
// these tests check graph quality against the sequential build (recall, reachability, degrees,
// invariants) rather than equality, plus the exact cases: one thread, levels, and what follows a
// save/load. Run under the tsan preset, they also check the locking.

#include <gtest/gtest.h>

#include <deque>
#include <filesystem>
#include <set>
#include <vector>

#include "hnsw_test_util.hpp"
#include "strata/brute_force.hpp"
#include "strata/dataset.hpp"
#include "strata/hnsw.hpp"
#include "strata/thread_pool.hpp"
#include "test_util.hpp"

#if defined(__SANITIZE_THREAD__)
#define STRATA_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define STRATA_TSAN 1
#endif
#endif

namespace strata {
namespace {

namespace fs = std::filesystem;

HnswIndex make_index(std::size_t dim, HnswParams params = {}) {
  auto index = HnswIndex::create(dim, Metric::kL2, params);
  EXPECT_TRUE(index.has_value()) << index.error().message;
  return std::move(*index);
}

// Structural invariants every build must keep: degree bounds, no self-loops or duplicate links,
// neighbors present on their layer, and the entry point on the top layer.
::testing::AssertionResult well_formed(const HnswIndex& index) {
  const std::size_t m = index.params().M;
  for (VectorId id = 0; id < index.size(); ++id) {
    for (int layer = 0; layer <= index.level(id); ++layer) {
      const auto links = index.neighbors(id, layer);
      if (links.size() > (layer == 0 ? 2 * m : m)) {
        return ::testing::AssertionFailure() << "node " << id << " over capacity on " << layer;
      }
      std::set<VectorId> seen;
      for (VectorId nbr : links) {
        if (nbr == id || !seen.insert(nbr).second || nbr >= index.size() ||
            index.level(nbr) < layer) {
          return ::testing::AssertionFailure() << "bad link " << id << " -> " << nbr;
        }
      }
    }
  }
  if (index.size() != 0 && index.level(*index.entry_point()) != index.max_level()) {
    return ::testing::AssertionFailure() << "entry point not on the top layer";
  }
  return ::testing::AssertionSuccess();
}

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

double mean_layer0_degree(const HnswIndex& index) {
  std::size_t edges = 0;
  for (VectorId id = 0; id < index.size(); ++id) {
    edges += index.neighbors(id, 0).size();
  }
  return static_cast<double>(edges) / static_cast<double>(index.size());
}

// Recall@10 by id against brute force at one ef_search.
double recall(const HnswIndex& index, const Matrix<float>& base, const Matrix<float>& queries,
              std::size_t ef) {
  auto exact = *BruteForceIndex::create(base.cols(), Metric::kL2);
  EXPECT_TRUE(exact.add_batch(base));
  std::size_t hits = 0;
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto truth = exact.search(queries.row(q), 10);
    auto found = index.search(queries.row(q), 10, ef);
    EXPECT_TRUE(truth && found);
    std::set<VectorId> ids;
    for (const auto& n : *truth) {
      ids.insert(n.id);
    }
    for (const auto& n : *found) {
      hits += ids.count(n.id);
    }
  }
  return static_cast<double>(hits) / static_cast<double>(10 * queries.rows());
}

// Recall within 0.01 of the sequential build at every ef.
void expect_recall_close(const HnswIndex& parallel, const HnswIndex& sequential,
                         const Matrix<float>& base, const Matrix<float>& queries) {
  for (std::size_t ef : {std::size_t{10}, std::size_t{40}, std::size_t{160}}) {
    const double seq = recall(sequential, base, queries, ef);
    const double par = recall(parallel, base, queries, ef);
    EXPECT_GE(par, seq - 0.01) << "ef " << ef << ": parallel " << par << ", sequential " << seq;
  }
}

// The parallel graph must be as good as the sequential one: well-formed, the same levels (drawn in
// id order), full layer-0 reachability, mean degree within 5%, and recall within 0.01.
void expect_same_quality(const Matrix<float>& base, const Matrix<float>& queries,
                         const HnswParams& params, std::size_t threads) {
  auto sequential = make_index(base.cols(), params);
  ASSERT_TRUE(sequential.add_batch(base));
  auto parallel = make_index(base.cols(), params);
  ThreadPool pool(threads);
  ASSERT_TRUE(parallel.add_batch(base, pool));

  EXPECT_TRUE(well_formed(parallel));
  for (VectorId id = 0; id < base.rows(); ++id) {
    ASSERT_EQ(parallel.level(id), sequential.level(id)) << "node " << id;
  }
  EXPECT_EQ(parallel.max_level(), sequential.max_level());
  EXPECT_GE(reachable_fraction(parallel), 0.999);
  const double seq_degree = mean_layer0_degree(sequential);
  EXPECT_NEAR(mean_layer0_degree(parallel), seq_degree, 0.05 * seq_degree);
  expect_recall_close(parallel, sequential, base, queries);
}

// A pool of one thread runs the parallel path inline and in id order: the graph must be exactly
// the sequential one, including when the index already holds nodes.
TEST(HnswParallel, OneThreadEqualsSequentialExactly) {
  const auto first = test::random_matrix(300, 16, 70);
  const auto batch = test::random_matrix(1500, 16, 71);
  const HnswParams params{.ef_construction = 64};
  ThreadPool one(1);
  for (bool prefilled : {false, true}) {
    auto parallel = make_index(16, params);
    auto sequential = make_index(16, params);
    if (prefilled) {
      ASSERT_TRUE(parallel.add_batch(first));
      ASSERT_TRUE(sequential.add_batch(first));
    }
    ASSERT_TRUE(parallel.add_batch(batch, one));
    ASSERT_TRUE(sequential.add_batch(batch));
    EXPECT_TRUE(test::same_hnsw_graph(parallel, sequential)) << "prefilled " << prefilled;
  }
}

TEST(HnswParallel, QualityMatchesSequentialOnRandomData) {
  expect_same_quality(test::random_matrix(6000, 32, 72), test::random_matrix(100, 32, 73),
                      {.ef_construction = 64}, 4);
}

TEST(HnswParallel, QualityMatchesSequentialOnSift10k) {
#ifdef STRATA_TSAN
  GTEST_SKIP() << "too slow under ThreadSanitizer; the random-data and stress tests cover it";
#endif
  const fs::path dir = fs::path(STRATA_DATA_DIR) / "siftsmall";
  if (!fs::exists(dir / "base.fbin")) {
    GTEST_SKIP() << "SIFT10K not found in " << dir;
  }
  auto dataset = load_dataset(dir);
  ASSERT_TRUE(dataset.has_value());
  expect_same_quality(dataset->base, dataset->query, {.ef_construction = 100}, 4);
}

// Many threads fighting over few, tiny neighbor lists: small M (layer-0 lists of 8, so almost every
// back-link overflows and triggers a re-selection under the lock), low-dimensional clustered
// points (the same few nodes are everyone's nearest neighbors), more threads than cores, repeated.
// Under TSan this is the race check; everywhere it checks the graph stays well-formed.
// Reachability is deliberately not checked here: this data is pathological for HNSW (5 tight
// clusters, M = 4), and whether its clusters end up linked depends on insertion order alone.
// Measured (2026-09-29): 20 sequential builds in shuffled orders gave layer-0 reachability from 0.2
// to 1.0, and parallel builds fall inside that range. So on this data reachability cannot tell a
// concurrency bug from ordering. The quality tests above check it on data where it is meaningful.
TEST(HnswParallel, HighContentionStress) {
  Matrix<float> data(3000, 4);
  const auto noise = test::random_matrix(3000, 4, 74);
  for (std::size_t i = 0; i < data.rows(); ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      data.row(i)[j] = static_cast<float>(i % 5) + (0.01F * noise.row(i)[j]);
    }
  }
  ThreadPool pool(8);
  for (unsigned round = 0; round < 3; ++round) {
    const HnswParams params{.M = 4, .ef_construction = 16, .seed = 100U + round};
    auto index = make_index(4, params);
    ASSERT_TRUE(index.add_batch(data, pool));
    ASSERT_EQ(index.size(), 3000U);
    EXPECT_TRUE(well_formed(index)) << "round " << round;
  }
}

// A parallel graph is not deterministic, but once built it is an ordinary index: saving, loading,
// and adding more sequentially must be deterministic, which needs the generator position the
// parallel build left (it drew the same levels in the same order as a sequential build would).
TEST(HnswParallel, SaveLoadThenSequentialAddIsDeterministic) {
  const auto dir = fs::temp_directory_path() / "strata_hnsw_parallel_save";
  fs::create_directories(dir);
  auto built = make_index(16, {.ef_construction = 64});
  ThreadPool pool(4);
  ASSERT_TRUE(built.add_batch(test::random_matrix(2000, 16, 75), pool));
  ASSERT_TRUE(built.save(dir / "p.snap"));
  auto loaded = HnswIndex::load(dir / "p.snap");
  ASSERT_TRUE(loaded) << loaded.error().message;
  EXPECT_TRUE(test::same_hnsw_graph(built, *loaded));
  const auto more = test::random_matrix(300, 16, 76);
  ASSERT_TRUE(built.add_batch(more));
  ASSERT_TRUE(loaded->add_batch(more));
  EXPECT_TRUE(test::same_hnsw_graph(built, *loaded));
  fs::remove_all(dir);
}

TEST(HnswParallel, BatchOnTopOfDeletesAndErrors) {
  auto index = make_index(8, {.ef_construction = 50});
  ThreadPool pool(4);
  ASSERT_TRUE(index.add_batch(test::random_matrix(500, 8, 77), pool));
  for (VectorId id = 0; id < 500; id += 5) {
    ASSERT_TRUE(index.remove(id));
  }
  ASSERT_TRUE(index.add_batch(test::random_matrix(1500, 8, 78), pool));
  EXPECT_TRUE(well_formed(index));
  const auto queries = test::random_matrix(30, 8, 79);
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto found = index.search(queries.row(q), 10, 64);
    ASSERT_TRUE(found);
    ASSERT_EQ(found->size(), 10U);
    for (const auto& n : *found) {
      EXPECT_FALSE(index.is_deleted(n.id));
    }
  }
  // A bad batch adds nothing.
  auto wrong = index.add_batch(test::random_matrix(10, 9, 80), pool);
  ASSERT_FALSE(wrong);
  EXPECT_EQ(wrong.error().code, ErrorCode::kDimensionMismatch);
  EXPECT_EQ(index.size(), 2000U);
}

}  // namespace
}  // namespace strata
