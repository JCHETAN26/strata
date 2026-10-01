#include "coordinator_service.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::server {
namespace {

// One in-flight call to one shard: its context, response, and final status.
template <typename Response>
struct ShardCall {
  std::uint32_t shard = 0;
  grpc::ClientContext context;
  Response response;
  grpc::Status status;
};

// Starts every call with gRPC's callback API and blocks until all have completed. `start(call,
// done)` must start one RPC whose completion callback is `done`.
//
// The completion callbacks run on gRPC's threads, possibly after this function's caller would
// otherwise have moved on, so the countdown they share lives in a shared_ptr each callback holds:
// nothing they touch after the final decrement can be destroyed under them. The calls themselves
// (contexts, responses) are the caller's and outlive the wait.
template <typename Response, typename Start>
void run_all(std::vector<ShardCall<Response>>& calls, Start&& start) {
  struct Countdown {
    std::mutex mutex;
    std::condition_variable cv;
    std::size_t pending = 0;
  };
  auto countdown = std::make_shared<Countdown>();
  countdown->pending = calls.size();
  for (ShardCall<Response>& call : calls) {
    start(call, [countdown, &call](grpc::Status status) {
      call.status = std::move(status);
      const std::lock_guard lock(countdown->mutex);
      if (--countdown->pending == 0) {
        countdown->cv.notify_all();
      }
    });
  }
  std::unique_lock lock(countdown->mutex);
  countdown->cv.wait(lock, [&] { return countdown->pending == 0; });
}

std::string shard_error(std::uint32_t shard, const std::string& address,
                        const grpc::Status& status) {
  return "shard " + std::to_string(shard) + " (" + address + "): " + status.error_message();
}

}  // namespace

CoordinatorService::CoordinatorService(const std::vector<std::string>& shard_addresses,
                                       std::size_t dim, Metric metric, CoordinatorOptions options)
    : codec_(static_cast<std::uint32_t>(shard_addresses.size())),
      dim_(dim),
      metric_(metric),
      options_(std::move(options)) {
  if (!options_.shard_credentials) {
    options_.shard_credentials = grpc::InsecureChannelCredentials();
  }
  // Reconnect quickly after a shard comes back. gRPC's default backoff grows to 120 s, so a shard
  // that was down for a few minutes could stay unreachable for up to two more after it recovered;
  // between attempts, calls fail fast with the last connection error. A coordinator talks to a
  // handful of known shards, so retrying them often costs nothing.
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_INITIAL_RECONNECT_BACKOFF_MS, 100);
  args.SetInt(GRPC_ARG_MIN_RECONNECT_BACKOFF_MS, 100);
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 2'000);
  shards_.reserve(shard_addresses.size());
  for (const std::string& address : shard_addresses) {
    Shard shard;
    shard.address = address;
    shard.channel = grpc::CreateCustomChannel(address, options_.shard_credentials, args);
    shard.stub = v1::VectorService::NewStub(shard.channel);
    shards_.push_back(std::move(shard));
  }
}

void CoordinatorService::set_deadline(grpc::ClientContext& context) const {
  context.set_deadline(std::chrono::system_clock::now() + options_.shard_timeout);
}

std::optional<VectorId> CoordinatorService::global_id(std::uint32_t shard, VectorId local) const {
  if (local >= codec_.local_limit(shard)) {
    return std::nullopt;
  }
  return codec_.to_global(shard, local);
}

grpc::Status CoordinatorService::Insert(grpc::ServerContext* /*context*/,
                                        const v1::InsertRequest* request,
                                        v1::InsertResponse* response) {
  const std::uint32_t shard_index = next_shard();
  v1::InsertRequest shard_request = *request;
  shard_request.set_id_limit(codec_.local_limit(shard_index));
  grpc::ClientContext client_context;
  set_deadline(client_context);
  v1::InsertResponse shard_response;
  grpc::Status status =
      shards_[shard_index].stub->Insert(&client_context, shard_request, &shard_response);
  if (!status.ok()) {
    return {status.error_code(), shard_error(shard_index, shards_[shard_index].address, status)};
  }
  const auto id = global_id(shard_index, shard_response.id());
  if (!id) {
    return {grpc::StatusCode::INTERNAL, "shard returned an id outside its global id range"};
  }
  response->set_id(*id);
  return grpc::Status::OK;
}

grpc::Status CoordinatorService::InsertBatch(grpc::ServerContext* /*context*/,
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
  const auto num_shards = static_cast<std::uint32_t>(shards_.size());

  // Route each input. One with an id from an earlier attempt goes back to the shard that id names,
  // as that shard's local id, to be verified there. The rest are dealt out round-robin.
  std::vector<v1::InsertBatchRequest> per_shard(num_shards);
  std::vector<std::vector<std::uint32_t>> inputs_of(num_shards);  // input index per shard row
  std::vector<bool> shard_has_ids(num_shards, false);
  std::uint32_t next = next_shard();
  for (std::uint32_t i = 0; i < count; ++i) {
    const VectorId given = num_ids == 0 ? kUnassignedId : request->ids(static_cast<int>(i));
    const std::uint32_t s = given == kUnassignedId ? next++ % num_shards : codec_.shard_of(given);
    v1::InsertBatchRequest& req = per_shard[s];
    const float* row = data + static_cast<std::size_t>(i) * dim_;
    req.mutable_values()->Add(row, row + dim_);
    req.set_count(req.count() + 1);
    req.add_ids(given == kUnassignedId ? kUnassignedId : codec_.to_local(given));
    shard_has_ids[s] = shard_has_ids[s] || given != kUnassignedId;
    inputs_of[s].push_back(i);
  }

  // Every input starts unassigned except those whose id the client already holds: those were
  // inserted by an earlier attempt and stay reported whatever happens to their shard now, so a
  // client that replaces its id list with this response never forgets them.
  response->mutable_ids()->Reserve(static_cast<int>(count));
  for (std::uint32_t i = 0; i < count; ++i) {
    response->add_ids(kUnassignedId);
  }
  for (std::uint32_t i = 0; i < num_ids; ++i) {
    response->set_ids(static_cast<int>(i), request->ids(static_cast<int>(i)));
  }

  // One call per shard that received anything, all in parallel.
  std::vector<std::uint32_t> called;
  for (std::uint32_t s = 0; s < num_shards; ++s) {
    if (per_shard[s].count() == 0) {
      continue;
    }
    if (!shard_has_ids[s]) {
      per_shard[s].clear_ids();  // nothing to verify on this shard
    }
    per_shard[s].set_id_limit(codec_.local_limit(s));
    called.push_back(s);
  }
  std::vector<ShardCall<v1::InsertBatchResponse>> calls(called.size());
  for (std::size_t c = 0; c < called.size(); ++c) {
    calls[c].shard = called[c];
  }
  run_all(calls, [&](ShardCall<v1::InsertBatchResponse>& call, auto done) {
    set_deadline(call.context);
    shards_[call.shard].stub->async()->InsertBatch(&call.context, &per_shard[call.shard],
                                                   &call.response, std::move(done));
  });

  // Put each shard's ids back in input order and keep the first failure, if any.
  auto record_error = [&](grpc::StatusCode code, const std::string& message) {
    if (response->error_code() == 0) {
      response->set_error_code(static_cast<std::int32_t>(code));
      response->set_error_message(message);
    }
  };
  for (const auto& call : calls) {
    const std::uint32_t s = call.shard;
    if (!call.status.ok()) {
      record_error(call.status.error_code(), shard_error(s, shards_[s].address, call.status));
      continue;
    }
    const std::vector<std::uint32_t>& inputs = inputs_of[s];
    if (static_cast<std::size_t>(call.response.ids_size()) != inputs.size()) {
      record_error(grpc::StatusCode::INTERNAL,
                   shard_error(s, shards_[s].address,
                               grpc::Status(grpc::StatusCode::INTERNAL, "wrong number of ids")));
      continue;
    }
    for (std::size_t k = 0; k < inputs.size(); ++k) {
      const VectorId local = call.response.ids(static_cast<int>(k));
      if (local == kUnassignedId) {
        continue;
      }
      const auto id = global_id(s, local);
      if (!id) {
        record_error(grpc::StatusCode::INTERNAL,
                     shard_error(s, shards_[s].address,
                                 grpc::Status(grpc::StatusCode::INTERNAL,
                                              "id outside the shard's global id range")));
        continue;
      }
      response->set_ids(static_cast<int>(inputs[k]), *id);
    }
    if (call.response.error_code() != 0) {
      record_error(
          static_cast<grpc::StatusCode>(call.response.error_code()),
          shard_error(s, shards_[s].address,
                      grpc::Status(grpc::StatusCode::UNKNOWN, call.response.error_message())));
    }
  }
  return grpc::Status::OK;
}

grpc::Status CoordinatorService::Search(grpc::ServerContext* /*context*/,
                                        const v1::SearchRequest* request,
                                        v1::SearchResponse* response) {
  const std::size_t k = request->k();
  const auto num_shards = static_cast<std::uint32_t>(shards_.size());

  // Scatter: one search per shard, all in flight at once. Each shard runs its own graph or scan
  // concurrently, which is the throughput win of sharding; the coordinator then merges.
  std::vector<ShardCall<v1::SearchResponse>> calls(num_shards);
  for (std::uint32_t s = 0; s < num_shards; ++s) {
    calls[s].shard = s;
    set_deadline(calls[s].context);
  }
  if (options_.search_fanout == SearchFanout::kAsync) {
    run_all(calls, [&](ShardCall<v1::SearchResponse>& call, auto done) {
      shards_[call.shard].stub->async()->Search(&call.context, request, &call.response,
                                                std::move(done));
    });
  } else {
    std::vector<std::future<void>> threads;
    threads.reserve(num_shards);
    for (auto& call : calls) {
      threads.push_back(std::async(std::launch::async, [this, &call, request] {
        call.status = shards_[call.shard].stub->Search(&call.context, *request, &call.response);
      }));
    }
    for (auto& thread : threads) {
      thread.get();
    }
  }

  // Gather and merge. Convert each shard's local ids to global, collect, then keep the k smallest
  // by (distance, id), the same order search() guarantees on one node.
  std::vector<std::pair<float, VectorId>> merged;
  for (const auto& call : calls) {
    if (!call.status.ok()) {
      return {call.status.error_code(),
              shard_error(call.shard, shards_[call.shard].address, call.status)};
    }
    for (const v1::Neighbor& n : call.response.neighbors()) {
      const auto id = global_id(call.shard, n.id());
      if (!id) {
        return {grpc::StatusCode::INTERNAL, "shard returned an id outside its global id range"};
      }
      merged.emplace_back(n.distance(), *id);
    }
  }
  const std::size_t keep = std::min(k, merged.size());
  std::partial_sort(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(keep),
                    merged.end());
  for (std::size_t i = 0; i < keep; ++i) {
    v1::Neighbor* out = response->add_neighbors();
    out->set_id(merged[i].second);
    out->set_distance(merged[i].first);
  }
  return grpc::Status::OK;
}

grpc::Status CoordinatorService::Delete(grpc::ServerContext* /*context*/,
                                        const v1::DeleteRequest* request,
                                        v1::DeleteResponse* response) {
  const VectorId global = request->id();
  if (global == kUnassignedId) {
    return grpc::Status::OK;  // never an id: deleting it is a no-op, like any unknown id
  }
  const std::uint32_t shard_index = codec_.shard_of(global);
  v1::DeleteRequest shard_request;
  shard_request.set_id(codec_.to_local(global));
  grpc::ClientContext client_context;
  set_deadline(client_context);
  grpc::Status status = shards_[shard_index].stub->Delete(&client_context, shard_request, response);
  if (!status.ok()) {
    return {status.error_code(), shard_error(shard_index, shards_[shard_index].address, status)};
  }
  return grpc::Status::OK;
}

grpc::Status CoordinatorService::Stats(grpc::ServerContext* /*context*/,
                                       const v1::StatsRequest* request,
                                       v1::StatsResponse* response) {
  std::vector<ShardCall<v1::StatsResponse>> calls(shards_.size());
  for (std::uint32_t s = 0; s < calls.size(); ++s) {
    calls[s].shard = s;
  }
  run_all(calls, [&](ShardCall<v1::StatsResponse>& call, auto done) {
    set_deadline(call.context);
    shards_[call.shard].stub->async()->Stats(&call.context, request, &call.response,
                                             std::move(done));
  });
  std::uint64_t size = 0;
  std::uint64_t live_size = 0;
  std::string index_kind;
  for (const auto& call : calls) {
    if (!call.status.ok()) {
      return {call.status.error_code(),
              shard_error(call.shard, shards_[call.shard].address, call.status)};
    }
    size += call.response.size();
    live_size += call.response.live_size();
    index_kind = call.response.index_kind();
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
