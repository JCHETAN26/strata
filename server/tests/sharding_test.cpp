// End-to-end sharding test: start several real shard gRPC servers on localhost, put a
// CoordinatorService in front of them, and check that insert / search / delete through the
// coordinator behave as a single exact index would.
//
// The shards use a flat (brute-force) index, so search is exact: the coordinator's merged top-k
// must equal a brute-force top-k computed here over all inserted vectors. That isolates the
// sharding logic (routing, id translation, scatter-gather merge) from HNSW's approximation.

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "coordinator_service.hpp"
#include "shard_service.hpp"
#include "strata/collection.hpp"
#include "strata/distance.hpp"

namespace strata::server {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kDim = 8;
constexpr std::uint32_t kNumShards = 3;

// A running shard server plus the temp directory its collection lives in.
struct RunningShard {
  std::unique_ptr<ShardService> service;
  std::unique_ptr<grpc::Server> server;
  fs::path dir;
  int port = 0;
};

class ShardingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const fs::path base = fs::temp_directory_path() /
                          ("strata_sharding_test_" +
                           std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                           std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    fs::create_directories(base);
    base_dir_ = base;

    std::vector<std::string> addresses;
    for (std::uint32_t s = 0; s < kNumShards; ++s) {
      auto shard = std::make_unique<RunningShard>();
      shard->dir = base / ("shard" + std::to_string(s));
      fs::create_directories(shard->dir);

      auto collection = Collection::open(shard->dir, kDim, Metric::kL2, {});
      ASSERT_TRUE(collection.has_value()) << collection.error().message;
      shard->service = std::make_unique<ShardService>(std::move(*collection), kDim, Metric::kL2);
      start(*shard);
      ASSERT_NE(shard->server, nullptr);
      addresses.push_back("127.0.0.1:" + std::to_string(shard->port));
      shards_.push_back(std::move(shard));
    }
    addresses_ = addresses;

    CoordinatorOptions options;
    options.shard_timeout = std::chrono::seconds(5);
    coordinator_ = std::make_unique<CoordinatorService>(addresses, kDim, Metric::kL2, options);
  }

  // Starts (or restarts) a shard's server. Port 0 picks a free port the first time; a restart
  // reuses it, so the coordinator's channel reconnects to the same address.
  static void start(RunningShard& shard) {
    grpc::ServerBuilder builder;
    int bound = 0;
    builder.AddListeningPort("127.0.0.1:" + std::to_string(shard.port),
                             grpc::InsecureServerCredentials(), &bound);
    builder.RegisterService(shard.service.get());
    shard.server = builder.BuildAndStart();
    shard.port = bound;
  }

  // Takes a shard off the network; its collection (and data) stay.
  static void stop(RunningShard& shard) {
    shard.server->Shutdown();
    shard.server.reset();
  }

  void TearDown() override {
    for (auto& shard : shards_) {
      if (shard->server) {
        shard->server->Shutdown();
      }
    }
    shards_.clear();
    std::error_code ec;
    fs::remove_all(base_dir_, ec);
  }

  // Brute-force L2 top-k over `vectors`, returning original indices nearest first.
  static std::vector<std::size_t> brute_force(const std::vector<std::vector<float>>& vectors,
                                              const std::vector<float>& query, std::size_t k) {
    std::vector<std::pair<float, std::size_t>> scored;
    scored.reserve(vectors.size());
    for (std::size_t i = 0; i < vectors.size(); ++i) {
      scored.emplace_back(scalar::l2_squared(query, vectors[i]), i);
    }
    std::sort(scored.begin(), scored.end());
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < std::min(k, scored.size()); ++i) {
      result.push_back(scored[i].second);
    }
    return result;
  }

  std::unique_ptr<CoordinatorService> coordinator_;
  std::vector<std::unique_ptr<RunningShard>> shards_;
  std::vector<std::string> addresses_;
  fs::path base_dir_;
};

TEST_F(ShardingTest, InsertSearchDeleteThroughCoordinator) {
  std::mt19937 rng(12345);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  constexpr std::size_t kNumVectors = 300;
  std::vector<std::vector<float>> vectors(kNumVectors, std::vector<float>(kDim));
  // global id -> original index, so search results (global ids) can be checked against the
  // brute-force reference (original indices).
  std::unordered_map<std::uint32_t, std::size_t> id_to_index;

  for (std::size_t i = 0; i < kNumVectors; ++i) {
    for (std::size_t d = 0; d < kDim; ++d) {
      vectors[i][d] = dist(rng);
    }
    v1::InsertRequest request;
    request.mutable_values()->Add(vectors[i].begin(), vectors[i].end());
    v1::InsertResponse response;
    grpc::Status status = coordinator_->Insert(nullptr, &request, &response);
    ASSERT_TRUE(status.ok()) << status.error_message();
    ASSERT_TRUE(id_to_index.emplace(response.id(), i).second) << "duplicate global id";
  }

  // Vectors should have spread round-robin, so no shard holds them all.
  {
    v1::StatsRequest request;
    v1::StatsResponse response;
    grpc::Status status = coordinator_->Stats(nullptr, &request, &response);
    ASSERT_TRUE(status.ok()) << status.error_message();
    EXPECT_EQ(response.size(), kNumVectors);
    EXPECT_EQ(response.live_size(), kNumVectors);
    EXPECT_EQ(response.num_shards(), kNumShards);
    EXPECT_EQ(response.dimension(), kDim);
  }

  // Search: the merged top-k must equal the exact brute-force top-k (as sets of vectors).
  constexpr std::size_t kK = 10;
  for (int q = 0; q < 20; ++q) {
    std::vector<float> query(kDim);
    for (std::size_t d = 0; d < kDim; ++d) {
      query[d] = dist(rng);
    }
    v1::SearchRequest request;
    request.mutable_values()->Add(query.begin(), query.end());
    request.set_k(kK);
    v1::SearchResponse response;
    grpc::Status status = coordinator_->Search(nullptr, &request, &response);
    ASSERT_TRUE(status.ok()) << status.error_message();
    ASSERT_EQ(static_cast<std::size_t>(response.neighbors_size()), kK);

    std::vector<std::size_t> got;
    float previous = -1.0f;
    for (const v1::Neighbor& n : response.neighbors()) {
      auto it = id_to_index.find(n.id());
      ASSERT_NE(it, id_to_index.end()) << "unknown global id in results";
      got.push_back(it->second);
      EXPECT_GE(n.distance(), previous) << "results not sorted by distance";
      previous = n.distance();
    }
    std::vector<std::size_t> expected = brute_force(vectors, query, kK);

    std::vector<std::size_t> got_sorted = got;
    std::sort(got_sorted.begin(), got_sorted.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(got_sorted, expected) << "coordinator top-k differs from exact brute force";
  }

  // Delete the exact top result for a fixed query through the coordinator, then confirm it is gone.
  {
    std::vector<float> query = vectors[0];
    v1::SearchRequest request;
    request.mutable_values()->Add(query.begin(), query.end());
    request.set_k(1);
    v1::SearchResponse response;
    ASSERT_TRUE(coordinator_->Search(nullptr, &request, &response).ok());
    ASSERT_EQ(response.neighbors_size(), 1);
    const std::uint32_t top_id = response.neighbors(0).id();
    EXPECT_EQ(id_to_index[top_id], 0u);  // vectors[0] queried by itself

    v1::DeleteRequest del;
    del.set_id(top_id);
    v1::DeleteResponse del_response;
    ASSERT_TRUE(coordinator_->Delete(nullptr, &del, &del_response).ok());

    // Idempotent: deleting again still succeeds.
    ASSERT_TRUE(coordinator_->Delete(nullptr, &del, &del_response).ok());

    v1::SearchResponse after;
    ASSERT_TRUE(coordinator_->Search(nullptr, &request, &after).ok());
    ASSERT_EQ(after.neighbors_size(), 1);
    EXPECT_NE(after.neighbors(0).id(), top_id) << "deleted id still returned";
  }

  // live_size dropped by exactly one.
  {
    v1::StatsRequest request;
    v1::StatsResponse response;
    ASSERT_TRUE(coordinator_->Stats(nullptr, &request, &response).ok());
    EXPECT_EQ(response.live_size(), kNumVectors - 1);
    EXPECT_EQ(response.size(), kNumVectors);
  }
}

TEST_F(ShardingTest, InsertBatchMatchesSingleInserts) {
  std::mt19937 rng(999);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  constexpr std::uint32_t kCount = 50;
  std::vector<float> flat;
  flat.reserve(kCount * kDim);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    for (std::size_t d = 0; d < kDim; ++d) {
      flat.push_back(dist(rng));
    }
  }

  v1::InsertBatchRequest request;
  request.mutable_values()->Add(flat.begin(), flat.end());
  request.set_count(kCount);
  v1::InsertBatchResponse response;
  grpc::Status status = coordinator_->InsertBatch(nullptr, &request, &response);
  ASSERT_TRUE(status.ok()) << status.error_message();
  ASSERT_EQ(static_cast<std::uint32_t>(response.ids_size()), kCount);

  // Ids are unique and the collection sees exactly kCount vectors.
  std::vector<std::uint32_t> ids(response.ids().begin(), response.ids().end());
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end()) << "duplicate ids from batch";

  v1::StatsRequest stats_request;
  v1::StatsResponse stats;
  ASSERT_TRUE(coordinator_->Stats(nullptr, &stats_request, &stats).ok());
  EXPECT_EQ(stats.size(), kCount);
}

// `count` random vectors, row-major.
std::vector<float> random_rows(std::uint32_t count, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> flat(static_cast<std::size_t>(count) * kDim);
  for (float& x : flat) {
    x = dist(rng);
  }
  return flat;
}

// Row i of a row-major matrix of kDim columns.
std::span<const float> row(const std::vector<float>& flat, std::size_t i) {
  return {flat.data() + i * kDim, kDim};
}

v1::InsertBatchRequest batch_request(const std::vector<float>& flat, std::uint32_t count) {
  v1::InsertBatchRequest request;
  request.mutable_values()->Add(flat.begin(), flat.end());
  request.set_count(count);
  return request;
}

std::size_t count_unassigned(const v1::InsertBatchResponse& response) {
  return static_cast<std::size_t>(
      std::count(response.ids().begin(), response.ids().end(), kUnassignedId));
}

TEST_F(ShardingTest, InsertBatchReportsPartialFailureAndRetriesWithoutDuplicates) {
  constexpr std::uint32_t kCount = 30;
  const std::vector<float> flat = random_rows(kCount, 7);

  // Shard 1 is down: its third of the batch fails, the other two shards' inserts succeed.
  stop(*shards_[1]);
  v1::InsertBatchRequest request = batch_request(flat, kCount);
  v1::InsertBatchResponse first;
  ASSERT_TRUE(coordinator_->InsertBatch(nullptr, &request, &first).ok());
  ASSERT_EQ(first.ids_size(), static_cast<int>(kCount));
  EXPECT_NE(first.error_code(), 0) << "a partial failure must be reported";
  EXPECT_NE(first.error_message().find("shard 1"), std::string::npos) << first.error_message();
  EXPECT_EQ(count_unassigned(first), kCount / kNumShards);

  // Bring the shard back and wait until the coordinator reaches it again (its channel reconnects
  // after a short backoff). Retrying earlier would also work: fresh inputs are dealt round-robin
  // anew on every attempt, so they would just land on the healthy shards.
  start(*shards_[1]);
  ASSERT_NE(shards_[1]->server, nullptr);
  {
    v1::StatsRequest ping;
    v1::StatsResponse pong;
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!coordinator_->Stats(nullptr, &ping, &pong).ok()) {
      ASSERT_LT(std::chrono::steady_clock::now(), give_up) << "shard 1 never came back";
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  // Retry with the ids the first attempt returned, as many times as it takes. Every attempt is
  // safe to repeat: inputs that have ids are verified, not inserted again.
  v1::InsertBatchResponse latest = first;
  for (int attempt = 0; attempt < 100 && latest.error_code() != 0; ++attempt) {
    v1::InsertBatchRequest retry = batch_request(flat, kCount);
    retry.mutable_ids()->CopyFrom(latest.ids());
    v1::InsertBatchResponse response;
    ASSERT_TRUE(coordinator_->InsertBatch(nullptr, &retry, &response).ok());
    // Ids a previous attempt reported never change or disappear.
    for (int i = 0; i < response.ids_size(); ++i) {
      if (latest.ids(i) != kUnassignedId) {
        ASSERT_EQ(response.ids(i), latest.ids(i));
      }
    }
    latest = response;
    if (latest.error_code() != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  ASSERT_EQ(latest.error_code(), 0) << latest.error_message();
  EXPECT_EQ(count_unassigned(latest), 0u);

  // Exactly kCount vectors exist: nothing was inserted twice.
  v1::StatsRequest stats_request;
  v1::StatsResponse stats;
  grpc::Status stats_status = coordinator_->Stats(nullptr, &stats_request, &stats);
  ASSERT_TRUE(stats_status.ok()) << stats_status.error_message();
  EXPECT_EQ(stats.size(), kCount);

  // Each id holds its own input: searching for input i finds latest.ids(i) at distance 0.
  for (std::uint32_t i = 0; i < kCount; ++i) {
    v1::SearchRequest search;
    const auto r = row(flat, i);
    search.mutable_values()->Add(r.begin(), r.end());
    search.set_k(1);
    v1::SearchResponse found;
    ASSERT_TRUE(coordinator_->Search(nullptr, &search, &found).ok());
    ASSERT_EQ(found.neighbors_size(), 1);
    EXPECT_EQ(found.neighbors(0).id(), latest.ids(static_cast<int>(i)));
    EXPECT_EQ(found.neighbors(0).distance(), 0.0f);
  }

  // Retrying a completed batch again is a no-op.
  v1::InsertBatchRequest again = batch_request(flat, kCount);
  again.mutable_ids()->CopyFrom(latest.ids());
  v1::InsertBatchResponse again_response;
  ASSERT_TRUE(coordinator_->InsertBatch(nullptr, &again, &again_response).ok());
  EXPECT_EQ(again_response.error_code(), 0);
  EXPECT_TRUE(
      std::equal(again_response.ids().begin(), again_response.ids().end(), latest.ids().begin()));
  ASSERT_TRUE(coordinator_->Stats(nullptr, &stats_request, &stats).ok());
  EXPECT_EQ(stats.size(), kCount);
}

TEST_F(ShardingTest, RetryIdsAreVerifiedAgainstTheStoredVectors) {
  constexpr std::uint32_t kCount = 6;
  std::vector<float> flat = random_rows(kCount, 11);
  v1::InsertBatchRequest request = batch_request(flat, kCount);
  v1::InsertBatchResponse inserted;
  ASSERT_TRUE(coordinator_->InsertBatch(nullptr, &request, &inserted).ok());
  ASSERT_EQ(inserted.error_code(), 0);

  v1::StatsRequest stats_request;
  v1::StatsResponse stats;

  // A retry that pairs an id with a different vector is refused, and inserts nothing.
  {
    std::vector<float> changed = flat;
    changed[2 * kDim] += 1.0f;
    v1::InsertBatchRequest retry = batch_request(changed, kCount);
    retry.mutable_ids()->CopyFrom(inserted.ids());
    v1::InsertBatchResponse response;
    ASSERT_TRUE(coordinator_->InsertBatch(nullptr, &retry, &response).ok());
    EXPECT_EQ(response.error_code(), static_cast<int>(grpc::StatusCode::INVALID_ARGUMENT));
    EXPECT_NE(response.error_message().find("different vector"), std::string::npos)
        << response.error_message();
  }
  // An id that was never assigned is refused.
  {
    v1::InsertBatchRequest retry = batch_request(flat, kCount);
    retry.mutable_ids()->CopyFrom(inserted.ids());
    retry.set_ids(0, 3000);
    v1::InsertBatchResponse response;
    ASSERT_TRUE(coordinator_->InsertBatch(nullptr, &retry, &response).ok());
    EXPECT_EQ(response.error_code(), static_cast<int>(grpc::StatusCode::INVALID_ARGUMENT));
  }
  ASSERT_TRUE(coordinator_->Stats(nullptr, &stats_request, &stats).ok());
  EXPECT_EQ(stats.size(), kCount);

  // A retry naming an id deleted since is accepted and does not bring the vector back.
  v1::DeleteRequest del;
  del.set_id(inserted.ids(3));
  v1::DeleteResponse del_response;
  ASSERT_TRUE(coordinator_->Delete(nullptr, &del, &del_response).ok());
  v1::InsertBatchRequest retry = batch_request(flat, kCount);
  retry.mutable_ids()->CopyFrom(inserted.ids());
  v1::InsertBatchResponse response;
  ASSERT_TRUE(coordinator_->InsertBatch(nullptr, &retry, &response).ok());
  EXPECT_EQ(response.error_code(), 0) << response.error_message();
  ASSERT_TRUE(coordinator_->Stats(nullptr, &stats_request, &stats).ok());
  EXPECT_EQ(stats.size(), kCount);
  EXPECT_EQ(stats.live_size(), kCount - 1);
}

TEST_F(ShardingTest, ShardRejectsABadRetryBeforeInsertingAnything) {
  // Directly at a shard: validation failures are the call's status, and nothing is inserted.
  ShardService& shard = *shards_[0]->service;
  constexpr std::uint32_t kCount = 4;
  const std::vector<float> flat = random_rows(kCount, 13);
  v1::InsertBatchRequest request = batch_request(flat, kCount);
  v1::InsertBatchResponse inserted;
  ASSERT_TRUE(shard.InsertBatch(nullptr, &request, &inserted).ok());
  ASSERT_EQ(inserted.error_code(), 0);

  // Input 0 retried with its id plus three new inputs, but input 1 claims input 0's id: refused.
  v1::InsertBatchRequest bad = batch_request(flat, kCount);
  bad.add_ids(inserted.ids(0));
  bad.add_ids(inserted.ids(0));
  bad.add_ids(kUnassignedId);
  bad.add_ids(kUnassignedId);
  v1::InsertBatchResponse response;
  grpc::Status status = shard.InsertBatch(nullptr, &bad, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  // Wrong lengths are refused too.
  v1::InsertBatchRequest short_values = batch_request(flat, kCount + 1);
  EXPECT_EQ(shard.InsertBatch(nullptr, &short_values, &response).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);
  v1::InsertBatchRequest short_ids = batch_request(flat, kCount);
  short_ids.add_ids(kUnassignedId);
  EXPECT_EQ(shard.InsertBatch(nullptr, &short_ids, &response).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);

  v1::StatsRequest stats_request;
  v1::StatsResponse stats;
  ASSERT_TRUE(shard.Stats(nullptr, &stats_request, &stats).ok());
  EXPECT_EQ(stats.size(), kCount);
}

TEST_F(ShardingTest, ShardRefusesIdsAtOrBeyondTheLimit) {
  // The coordinator passes each shard the bound that keeps its global ids inside 32 bits. A real
  // bound is ~4 billion / N; a tiny one exercises the same check.
  ShardService& shard = *shards_[0]->service;
  const std::vector<float> flat = random_rows(5, 17);

  v1::InsertBatchRequest request = batch_request(flat, 5);
  request.set_id_limit(3);
  v1::InsertBatchResponse response;
  ASSERT_TRUE(shard.InsertBatch(nullptr, &request, &response).ok());
  EXPECT_EQ(response.error_code(), static_cast<int>(grpc::StatusCode::RESOURCE_EXHAUSTED));
  const std::vector<std::uint32_t> ids(response.ids().begin(), response.ids().end());
  EXPECT_EQ(ids, (std::vector<std::uint32_t>{0, 1, 2, kUnassignedId, kUnassignedId}));

  v1::InsertRequest one;
  const auto first_row = row(flat, 0);
  one.mutable_values()->Add(first_row.begin(), first_row.end());
  one.set_id_limit(3);
  v1::InsertResponse one_response;
  EXPECT_EQ(shard.Insert(nullptr, &one, &one_response).error_code(),
            grpc::StatusCode::RESOURCE_EXHAUSTED);
  one.set_id_limit(0);  // no bound: a lone shard uses the whole 32-bit space
  EXPECT_TRUE(shard.Insert(nullptr, &one, &one_response).ok());
  EXPECT_EQ(one_response.id(), 3u);
}

TEST_F(ShardingTest, AsyncAndThreadPerShardFanoutAgree) {
  constexpr std::uint32_t kCount = 120;
  const std::vector<float> flat = random_rows(kCount, 19);
  v1::InsertBatchRequest request = batch_request(flat, kCount);
  v1::InsertBatchResponse inserted;
  ASSERT_TRUE(coordinator_->InsertBatch(nullptr, &request, &inserted).ok());
  ASSERT_EQ(inserted.error_code(), 0);

  CoordinatorOptions threaded;
  threaded.search_fanout = SearchFanout::kThreadPerShard;
  CoordinatorService baseline(addresses_, kDim, Metric::kL2, threaded);

  const std::vector<float> queries = random_rows(10, 23);
  for (std::size_t q = 0; q < 10; ++q) {
    v1::SearchRequest search;
    const auto r = row(queries, q);
    search.mutable_values()->Add(r.begin(), r.end());
    search.set_k(10);
    v1::SearchResponse async_result;
    v1::SearchResponse thread_result;
    ASSERT_TRUE(coordinator_->Search(nullptr, &search, &async_result).ok());
    ASSERT_TRUE(baseline.Search(nullptr, &search, &thread_result).ok());
    ASSERT_EQ(async_result.neighbors_size(), 10);
    ASSERT_EQ(async_result.SerializeAsString(), thread_result.SerializeAsString());
  }
}

TEST_F(ShardingTest, SearchFailsCleanlyWhenAShardIsDown) {
  stop(*shards_[2]);
  const std::vector<float> query = random_rows(1, 29);
  v1::SearchRequest search;
  search.mutable_values()->Add(query.begin(), query.end());
  search.set_k(5);
  v1::SearchResponse response;
  grpc::Status status = coordinator_->Search(nullptr, &search, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
  EXPECT_NE(status.error_message().find("shard 2"), std::string::npos) << status.error_message();
}

}  // namespace
}  // namespace strata::server
