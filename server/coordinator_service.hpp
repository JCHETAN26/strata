#pragma once

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "id_codec.hpp"
#include "strata/distance.hpp"
#include "strata/v1/vector_service.grpc.pb.h"

namespace strata::server {

// The sharding coordinator. It implements the same VectorService as a shard, so clients talk to it
// exactly as they would to a single node, and it fans requests out to the shards behind it:
//
//   Insert   -> route to one shard (round-robin), translate its local id to a global id.
//   Search   -> scatter to every shard in parallel, gather, merge the per-shard top-k into one.
//   Delete   -> decode the global id to (shard, local id), route to that one shard.
//   Stats    -> query every shard and aggregate.
//
// Vectors are spread across shards round-robin, so the data (and search work) divides evenly
// regardless of how ids are distributed. Global ids are assigned by ShardIdCodec.
class CoordinatorService final : public v1::VectorService::Service {
 public:
  // `shard_addresses` are host:port strings, one per shard, in a fixed order: a shard's index in
  // this list is baked into every global id it produced, so the same list must be used across
  // restarts for existing ids to keep resolving to the right shard.
  CoordinatorService(const std::vector<std::string>& shard_addresses, std::size_t dim,
                     Metric metric);

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
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<v1::VectorService::Stub> stub;
  };

  // Next shard for round-robin insert routing. Wraps modulo shard count.
  std::uint32_t next_shard() {
    return static_cast<std::uint32_t>(round_robin_.fetch_add(1, std::memory_order_relaxed) %
                                      shards_.size());
  }

  std::vector<Shard> shards_;
  ShardIdCodec codec_;
  std::size_t dim_;
  Metric metric_;
  std::atomic<std::uint64_t> round_robin_{0};
};

}  // namespace strata::server
