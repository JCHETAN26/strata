#include "strata/thread_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "strata/brute_force.hpp"
#include "test_util.hpp"

namespace strata {
namespace {

TEST(ThreadPool, SizeIncludesCaller) {
  EXPECT_EQ(ThreadPool(1).size(), 1U);
  EXPECT_EQ(ThreadPool(4).size(), 4U);
  EXPECT_GE(ThreadPool(0).size(), 1U);
}

TEST(ThreadPool, VisitsEveryIndexExactlyOnce) {
  ThreadPool pool(4);
  for (std::size_t n : {0U, 1U, 3U, 4U, 5U, 100U, 10007U}) {
    std::vector<std::atomic<int>> hits(n);
    pool.parallel_for(n, [&](std::size_t i) { hits[i].fetch_add(1); });
    for (std::size_t i = 0; i < n; ++i) {
      ASSERT_EQ(hits[i].load(), 1) << "n=" << n << " i=" << i;
    }
  }
}

TEST(ThreadPool, ExplicitChunkSizes) {
  ThreadPool pool(3);
  for (std::size_t chunk : {1U, 2U, 7U, 1000U}) {
    std::vector<std::atomic<int>> hits(500);
    pool.parallel_for(hits.size(), [&](std::size_t i) { hits[i].fetch_add(1); }, chunk);
    for (const auto& h : hits) {
      ASSERT_EQ(h.load(), 1);
    }
  }
}

TEST(ThreadPool, ManyConsecutiveJobs) {
  ThreadPool pool(4);
  std::atomic<std::size_t> total{0};
  for (int job = 0; job < 500; ++job) {
    pool.parallel_for(64, [&](std::size_t) { total.fetch_add(1); });
  }
  EXPECT_EQ(total.load(), 500U * 64U);
}

TEST(ThreadPool, UsesMultipleThreads) {
  ThreadPool pool(4);
  std::mutex m;
  std::set<std::thread::id> ids;
  pool.parallel_for(
      1000,
      [&](std::size_t) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        const std::lock_guard lock(m);
        ids.insert(std::this_thread::get_id());
      },
      1);
  EXPECT_GT(ids.size(), 1U);
}

TEST(ThreadPool, ConcurrentCallersAreSerialized) {
  ThreadPool pool(4);
  std::atomic<std::size_t> total{0};
  std::vector<std::thread> callers;
  for (int c = 0; c < 4; ++c) {
    callers.emplace_back([&] {
      for (int job = 0; job < 50; ++job) {
        pool.parallel_for(100, [&](std::size_t) { total.fetch_add(1); });
      }
    });
  }
  for (auto& c : callers) {
    c.join();
  }
  EXPECT_EQ(total.load(), 4U * 50U * 100U);
}

TEST(BruteForceBatch, MatchesSequentialSearch) {
  const auto base = test::random_matrix(2000, 24, 1);
  const auto queries = test::random_matrix(257, 24, 2);
  auto index = BruteForceIndex::create(24, Metric::kL2);
  ASSERT_TRUE(index && index->add_batch(base));
  ThreadPool pool(4);
  auto batch = index->search_batch(queries, 10, pool);
  ASSERT_TRUE(batch.has_value());
  ASSERT_EQ(batch->size(), queries.rows());
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    EXPECT_EQ((*batch)[q], *index->search(queries.row(q), 10));
  }
}

TEST(BruteForceBatch, DimensionMismatchAndEmpty) {
  auto index = BruteForceIndex::create(4, Metric::kL2);
  ASSERT_TRUE(index);
  ThreadPool pool(2);
  EXPECT_FALSE(index->search_batch(test::random_matrix(3, 5, 1), 1, pool).has_value());
  auto empty = index->search_batch(Matrix<float>(0, 4), 1, pool);
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());
}

}  // namespace
}  // namespace strata
