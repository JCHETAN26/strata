#include "shard_service.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "id_codec.hpp"
#include "status_util.hpp"
#include "strata/distance.hpp"

namespace strata::server {

grpc::Status ShardService::insert_one(std::span<const float> vector, VectorId id_limit,
                                      VectorId* id) {
  if (id_limit != 0 && collection_.size() >= id_limit) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "shard is full: its next id would not fit the 32-bit global id space"};
  }
  auto inserted = collection_.insert(vector);
  if (!inserted) {
    return to_status(inserted.error());
  }
  if (id_limit != 0 && *inserted >= id_limit) {
    // A concurrent insert took the last id between the check above and this insert. Undo it (a
    // tombstone), so an id outside the global space is never handed out.
    (void)collection_.remove(*inserted);
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "shard is full: its next id would not fit the 32-bit global id space"};
  }
  *id = *inserted;
  return grpc::Status::OK;
}

grpc::Status ShardService::Insert(grpc::ServerContext* /*context*/,
                                  const v1::InsertRequest* request, v1::InsertResponse* response) {
  std::span<const float> vector(request->values().data(),
                                static_cast<std::size_t>(request->values_size()));
  VectorId id = 0;
  if (grpc::Status status = insert_one(vector, request->id_limit(), &id); !status.ok()) {
    return status;
  }
  response->set_id(id);
  return grpc::Status::OK;
}

grpc::Status ShardService::InsertBatch(grpc::ServerContext* /*context*/,
                                       const v1::InsertBatchRequest* request,
                                       v1::InsertBatchResponse* response) {
  const std::uint32_t count = request->count();
  const auto total = static_cast<std::size_t>(request->values_size());
  const auto num_ids = static_cast<std::size_t>(request->ids_size());
  if (count == 0) {
    if (total != 0 || num_ids != 0) {
      return {grpc::StatusCode::INVALID_ARGUMENT, "count is 0 but values or ids are not empty"};
    }
    return grpc::Status::OK;
  }
  if (total != static_cast<std::size_t>(count) * dim_) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "expected count * dimension = " + std::to_string(count) + " * " + std::to_string(dim_) +
                " values, got " + std::to_string(total)};
  }
  if (num_ids != 0 && num_ids != count) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ids must be empty or have count entries"};
  }
  const float* data = request->values().data();
  auto row = [&](std::uint32_t i) { return std::span<const float>(data + i * dim_, dim_); };
  auto given_id = [&](std::uint32_t i) {
    return num_ids == 0 ? kUnassignedId : request->ids(static_cast<int>(i));
  };

  // Pass 1, no side effects: every input that carries an id from an earlier attempt must name a
  // vector this shard really stored for it. Checked for the whole batch before inserting anything,
  // so a malformed retry is rejected cleanly instead of half-applied. A deleted id is accepted
  // (its vector was inserted; the delete came later) and is not resurrected.
  if (num_ids != 0) {
    const std::size_t size = collection_.size();
    std::unordered_set<VectorId> seen;
    for (std::uint32_t i = 0; i < count; ++i) {
      const VectorId id = given_id(i);
      if (id == kUnassignedId) {
        continue;
      }
      if (!seen.insert(id).second) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "id " + std::to_string(id) + " is given for more than one input"};
      }
      if (id >= size) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "id " + std::to_string(id) + " (input " +
                                                        std::to_string(i) +
                                                        ") was never assigned by this shard"};
      }
      const auto stored = collection_.get(id);
      if (stored && std::memcmp(stored->data(), row(i).data(), dim_ * sizeof(float)) != 0) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "id " + std::to_string(id) +
                                                        " holds a different vector than input " +
                                                        std::to_string(i)};
      }
    }
  }

  // Pass 2: insert the inputs without an id, in input order, stopping at the first failure. The
  // response lists every input: its id (given or new), or kUnassignedId if it was not inserted.
  response->mutable_ids()->Reserve(static_cast<int>(count));
  for (std::uint32_t i = 0; i < count; ++i) {
    response->add_ids(kUnassignedId);
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    const VectorId given = given_id(i);
    if (given != kUnassignedId) {
      response->set_ids(static_cast<int>(i), given);
      continue;
    }
    VectorId id = 0;
    if (grpc::Status status = insert_one(row(i), request->id_limit(), &id); !status.ok()) {
      response->set_error_code(static_cast<std::int32_t>(status.error_code()));
      response->set_error_message("input " + std::to_string(i) + ": " + status.error_message());
      // Later inputs that carried ids keep them: they were inserted by an earlier attempt.
      for (std::uint32_t j = i + 1; j < count; ++j) {
        if (given_id(j) != kUnassignedId) {
          response->set_ids(static_cast<int>(j), given_id(j));
        }
      }
      return grpc::Status::OK;
    }
    response->set_ids(static_cast<int>(i), id);
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
