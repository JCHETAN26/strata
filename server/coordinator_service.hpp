#pragma once

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "id_codec.hpp"
#include "strata/distance.hpp"
#include "strata/v1/vector_service.grpc.pb.h"

namespace strata::server {

// How Search reaches the shards.
enum class SearchFanout {
  // gRPC's callback API: every shard call is started at once and completes on gRPC's own
  // completion threads, so a query holds no thread per shard. The default.
  kAsync,
  // One new thread per shard per query (std::async around a blocking call). The original design,
  // kept only as the measured baseline for bench/run_coordinator_bench.py.
  kThreadPerShard,
};

struct CoordinatorOptions {
  // Credentials for the shard channels; null means plaintext (shards on loopback).
  std::shared_ptr<grpc::ChannelCredentials> shard_credentials;
  // Deadline for each call to a shard. A shard that does not answer in time fails that call with
  // DEADLINE_EXCEEDED instead of hanging the request forever.
  std::chrono::milliseconds shard_timeout{10'000};
  SearchFanout search_fanout = SearchFanout::kAsync;
};

// The sharding coordinator. It implements the same VectorService as a shard, so clients talk to it
// exactly as they would to a single node, and it fans requests out to the shards behind it:
//
//   Insert       -> route to one shard (round-robin), translate its local id to a global id.
//   InsertBatch  -> deal the vectors out round-robin, one call per shard, all shards in parallel.
//   Search       -> scatter to every shard in parallel, gather, merge the per-shard top-k into one.
//   Delete       -> decode the global id to (shard, local id), route to that one shard.
//   Stats        -> query every shard in parallel and aggregate.
//
// Vectors are spread across shards round-robin, so the data (and search work) divides evenly
// regardless of how ids are distributed. Global ids are assigned by ShardIdCodec and are 32-bit:
// with N shards each shard holds at most about 4.29 billion / N vectors, and an insert beyond that
// fails with RESOURCE_EXHAUSTED (the shard enforces the bound the coordinator passes it).
//
// InsertBatch is not atomic across shards: one shard can fail while the others succeed. The
// response reports every input's id or kUnassignedId, and the request's `ids` field makes a retry
// insert only what is missing (see the proto). Retries of the same batch must not run
// concurrently with each other: two in flight would both insert the missing inputs.
//
// Thread safety: all methods may be called concurrently (gRPC's server threads do). The shard
// stubs are thread-safe, the round-robin counter is atomic, and each call's state is its own.
class CoordinatorService final : public v1::VectorService::Service {
 public:
  // `shard_addresses` are host:port strings, one per shard, in a fixed order: a shard's index in
  // this list is baked into every global id it produced, so the same list must be used across
  // restarts for existing ids to keep resolving to the right shard.
  CoordinatorService(const std::vector<std::string>& shard_addresses, std::size_t dim,
                     Metric metric, CoordinatorOptions options = {});

  grpc::Status Insert(grpc::ServerContext* context, const v1::InsertRequest* request,
                      v1::InsertResponse* response) override;
  grpc::Status InsertBatch(grpc::ServerContext* context, const v1::InsertBatchRequest* request,
                           v1::InsertBatchResponse* response) override;
  grpc::Status Search(grpc::ServerContext* context, const v1::SearchRequest* request,
                      v1::SearchResponse* response) override;
  grpc::Status Delete(grpc::ServerContext* context, const v1::DeleteRequest* request,
                      v1::DeleteResponse* response) override;
  grpc::Status Stats(grpc::ServerContext* context, const v1::StatsRequest* request,
                     v1::StatsResponse* response) override;

 private:
  struct Shard {
    std::string address;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<v1::VectorService::Stub> stub;
  };

  // Next shard for round-robin insert routing. Wraps modulo shard count.
  std::uint32_t next_shard() {
    return static_cast<std::uint32_t>(round_robin_.fetch_add(1, std::memory_order_relaxed) %
                                      shards_.size());
  }

  // Sets the per-call shard deadline on `context`.
  void set_deadline(grpc::ClientContext& context) const;

  // Global id for a shard's local id, or nullopt if the shard returned an id outside the range
  // this coordinator gave it (possible only if something else wrote to the shard directly).
  [[nodiscard]] std::optional<VectorId> global_id(std::uint32_t shard, VectorId local) const;

  std::vector<Shard> shards_;
  ShardIdCodec codec_;
  std::size_t dim_;
  Metric metric_;
  CoordinatorOptions options_;
  std::atomic<std::uint64_t> round_robin_{0};
};

}  // namespace strata::server
