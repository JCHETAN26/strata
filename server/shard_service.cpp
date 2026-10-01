#include "server/shard_service.hpp"

#include <cstddef>
#include <span>
#include <vector>

#include "server/status_util.hpp"
#include "strata/distance.hpp"

namespace strata::server {

grpc::Status ShardService::Insert(grpc::ServerContext* /*context*/,
                                  const v1::InsertRequest* request, v1::InsertResponse* response) {
  std::span<const float> vector(request->values().data(),
                                static_cast<std::size_t>(request->values_size()));
  auto id = collection_.insert(vector);
  if (!id) {
    return to_status(id.error());
  }
  response->set_id(*id);
  return grpc::Status::OK;
}

grpc::Status ShardService::InsertBatch(grpc::ServerContext* /*context*/,
                                       const v1::InsertBatchRequest* request,
                                       v1::InsertBatchResponse* response) {
  const std::uint32_t count = request->count();
  if (count == 0) {
    return grpc::Status::OK;
  }
  const auto total = static_cast<std::size_t>(request->values_size());
  if (total % count != 0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "values length is not a multiple of count");
  }
  const std::size_t dim = total / count;
  const float* data = request->values().data();
  response->mutable_ids()->Reserve(static_cast<int>(count));
  for (std::uint32_t i = 0; i < count; ++i) {
    std::span<const float> vector(data + i * dim, dim);
    auto id = collection_.insert(vector);
    if (!id) {
      return to_status(id.error());
    }
    response->add_ids(*id);
  }
  return grpc::Status::OK;
}

grpc::Status ShardService::Search(grpc::ServerContext* /*context*/,
                                  const v1::SearchRequest* request, v1::SearchResponse* response) {
  std::span<const float> query(request->values().data(),
                               static_cast<std::size_t>(request->values_size()));
  const std::size_t k = request->k();
  const std::size_t ef_search = request->ef_search() == 0 ? 64 : request->ef_search();
  auto neighbors = collection_.search(query, k, ef_search);
  if (!neighbors) {
    return to_status(neighbors.error());
  }
  for (const Neighbor& n : *neighbors) {
    v1::Neighbor* out = response->add_neighbors();
    out->set_id(n.id);
    out->set_distance(n.distance);
  }
  return grpc::Status::OK;
}

grpc::Status ShardService::Delete(grpc::ServerContext* /*context*/,
                                  const v1::DeleteRequest* request,
                                  v1::DeleteResponse* /*response*/) {
  auto removed = collection_.remove(request->id());
  // Deleting an unknown or already-deleted id is not an error at the API level; the collection
  // reports kNotFound, which we swallow to keep Delete idempotent.
  if (!removed && removed.error().code != ErrorCode::kNotFound) {
    return to_status(removed.error());
  }
  return grpc::Status::OK;
}

grpc::Status ShardService::Stats(grpc::ServerContext* /*context*/,
                                 const v1::StatsRequest* /*request*/, v1::StatsResponse* response) {
  response->set_dimension(static_cast<std::uint32_t>(dim_));
  response->set_metric(std::string(to_string(metric_)));
  response->set_index_kind(collection_.index_kind() == IndexKind::kHnsw ? "hnsw" : "flat");
  response->set_size(collection_.size());
  response->set_live_size(collection_.live_size());
  response->set_num_shards(1);
  return grpc::Status::OK;
}

}  // namespace strata::server
