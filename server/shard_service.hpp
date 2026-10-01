#pragma once

#include <grpcpp/grpcpp.h>

#include <cstddef>
#include <span>
#include <utility>

#include "strata/collection.hpp"
#include "strata/distance.hpp"
#include "strata/v1/vector_service.grpc.pb.h"

namespace strata::server {

// One shard: a gRPC front end over a single Collection (its own directory, WAL, and snapshot).
//
// Ids here are the Collection's local ids. On its own this is a complete, standalone vector-search
// node whose ids are already global (num_shards == 1). Behind a coordinator, several ShardServices
// each own a slice of the data and the coordinator translates their local ids to global ones.
//
// Thread safety: Collection's methods are all safe to call concurrently, so the service holds no
// lock of its own; gRPC's thread pool dispatches requests straight through.
class ShardService final : public v1::VectorService::Service {
 public:
  ShardService(Collection collection, std::size_t dim, Metric metric)
      : collection_(std::move(collection)), dim_(dim), metric_(metric) {}

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
  // Inserts one vector, refusing with RESOURCE_EXHAUSTED if its id would reach `id_limit` (0: no
  // limit). See InsertRequest.id_limit.
  grpc::Status insert_one(std::span<const float> vector, VectorId id_limit, VectorId* id);

  Collection collection_;
  std::size_t dim_;
  Metric metric_;
};

}  // namespace strata::server
