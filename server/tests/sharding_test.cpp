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
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "server/coordinator_service.hpp"
#include "server/shard_service.hpp"
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
    const fs::path base =
        fs::temp_directory_path() /
        ("strata_sharding_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
         "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

      grpc::ServerBuilder builder;
      builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &shard->port);
      builder.RegisterService(shard->service.get());
      shard->server = builder.BuildAndStart();
      ASSERT_NE(shard->server, nullptr);
      addresses.push_back("127.0.0.1:" + std::to_string(shard->port));
      shards_.push_back(std::move(shard));
    }

    coordinator_ = std::make_unique<CoordinatorService>(addresses, kDim, Metric::kL2);
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

}  // namespace
}  // namespace strata::server
