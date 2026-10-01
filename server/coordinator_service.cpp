#include "server/coordinator_service.hpp"

#include <algorithm>
#include <cstdint>
#include <future>
#include <string>
#include <utility>
#include <vector>

namespace strata::server {

CoordinatorService::CoordinatorService(const std::vector<std::string>& shard_addresses,
                                       std::size_t dim, Metric metric)
    : codec_(static_cast<std::uint32_t>(shard_addresses.size())), dim_(dim), metric_(metric) {
  shards_.reserve(shard_addresses.size());
  for (const std::string& address : shard_addresses) {
    Shard shard;
    shard.channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
    shard.stub = v1::VectorService::NewStub(shard.channel);
    shards_.push_back(std::move(shard));
  }
}

grpc::Status CoordinatorService::Insert(grpc::ServerContext* /*context*/,
                                        const v1::InsertRequest* request,
                                        v1::InsertResponse* response) {
  const std::uint32_t shard_index = next_shard();
  grpc::ClientContext client_context;
  v1::InsertResponse shard_response;
  grpc::Status status =
      shards_[shard_index].stub->Insert(&client_context, *request, &shard_response);
  if (!status.ok()) {
    return status;
  }
  response->set_id(codec_.to_global(shard_index, shard_response.id()));
  return grpc::Status::OK;
}

grpc::Status CoordinatorService::InsertBatch(grpc::ServerContext* /*context*/,
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
  const auto num_shards = static_cast<std::uint32_t>(shards_.size());

  // Deal the vectors out to shards round-robin, remembering each input's shard so the returned
  // local ids can be put back in input order as global ids.
  const std::uint32_t start = next_shard();
  std::vector<v1::InsertBatchRequest> per_shard(num_shards);
  std::vector<std::uint32_t> shard_of_input(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint32_t s = (start + i) % num_shards;
    shard_of_input[i] = s;
    v1::InsertBatchRequest& req = per_shard[s];
    const float* row = data + static_cast<std::size_t>(i) * dim;
    req.mutable_values()->Add(row, row + dim);
    req.set_count(req.count() + 1);
  }

  // One InsertBatch call per shard that received anything.
  std::vector<std::vector<std::uint32_t>> local_ids(num_shards);
  for (std::uint32_t s = 0; s < num_shards; ++s) {
    if (per_shard[s].count() == 0) {
      continue;
    }
    grpc::ClientContext client_context;
    v1::InsertBatchResponse shard_response;
    grpc::Status status =
        shards_[s].stub->InsertBatch(&client_context, per_shard[s], &shard_response);
    if (!status.ok()) {
      return status;
    }
    local_ids[s].assign(shard_response.ids().begin(), shard_response.ids().end());
  }

  // Reassemble in input order. Each shard handed back ids in the order it received its slice, and
  // that order matches the ascending input index, so a per-shard cursor lines them up.
  std::vector<std::uint32_t> cursor(num_shards, 0);
  response->mutable_ids()->Reserve(static_cast<int>(count));
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint32_t s = shard_of_input[i];
    const std::uint32_t local = local_ids[s][cursor[s]++];
    response->add_ids(codec_.to_global(s, local));
  }
  return grpc::Status::OK;
}

grpc::Status CoordinatorService::Search(grpc::ServerContext* /*context*/,
                                        const v1::SearchRequest* request,
                                        v1::SearchResponse* response) {
  const std::size_t k = request->k();

  // Scatter: one search per shard, in parallel. Each shard runs its own graph/scan concurrently,
  // which is the throughput win of sharding; the coordinator then merges.
  struct ShardResult {
    grpc::Status status;
    v1::SearchResponse response;
  };
  std::vector<std::future<ShardResult>> futures;
  futures.reserve(shards_.size());
  for (std::uint32_t s = 0; s < shards_.size(); ++s) {
    futures.push_back(std::async(std::launch::async, [this, s, request]() {
      ShardResult result;
      grpc::ClientContext client_context;
      result.status = shards_[s].stub->Search(&client_context, *request, &result.response);
      return result;
    }));
  }

  // Gather and merge. Convert each shard's local ids to global, collect, then keep the k smallest
  // by (distance, id) — the same order search() guarantees on one node.
  std::vector<std::pair<float, VectorId>> merged;
  for (std::uint32_t s = 0; s < shards_.size(); ++s) {
    ShardResult result = futures[s].get();
    if (!result.status.ok()) {
      return result.status;
    }
    for (const v1::Neighbor& n : result.response.neighbors()) {
      merged.emplace_back(n.distance(), codec_.to_global(s, n.id()));
    }
  }
  std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    return a.second < b.second;
  });
  if (merged.size() > k) {
    merged.resize(k);
  }
  for (const auto& [distance, id] : merged) {
    v1::Neighbor* out = response->add_neighbors();
    out->set_id(id);
    out->set_distance(distance);
  }
  return grpc::Status::OK;
}

grpc::Status CoordinatorService::Delete(grpc::ServerContext* /*context*/,
                                        const v1::DeleteRequest* request,
                                        v1::DeleteResponse* response) {
  const VectorId global_id = request->id();
  const std::uint32_t shard_index = codec_.shard_of(global_id);
  v1::DeleteRequest shard_request;
  shard_request.set_id(codec_.to_local(global_id));
  grpc::ClientContext client_context;
  return shards_[shard_index].stub->Delete(&client_context, shard_request, response);
}

grpc::Status CoordinatorService::Stats(grpc::ServerContext* /*context*/,
                                       const v1::StatsRequest* request,
                                       v1::StatsResponse* response) {
  std::uint64_t size = 0;
  std::uint64_t live_size = 0;
  std::string index_kind;
  for (Shard& shard : shards_) {
    grpc::ClientContext client_context;
    v1::StatsResponse shard_response;
    grpc::Status status = shard.stub->Stats(&client_context, *request, &shard_response);
    if (!status.ok()) {
      return status;
    }
    size += shard_response.size();
    live_size += shard_response.live_size();
    index_kind = shard_response.index_kind();
  }
  response->set_dimension(static_cast<std::uint32_t>(dim_));
  response->set_metric(std::string(to_string(metric_)));
  response->set_index_kind(index_kind);
  response->set_size(size);
  response->set_live_size(live_size);
  response->set_num_shards(static_cast<std::uint32_t>(shards_.size()));
  return grpc::Status::OK;
}

}  // namespace strata::server
