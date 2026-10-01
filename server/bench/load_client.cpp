// strata_load: a standalone client for multi-machine benchmarks, run on its own machine against
// a coordinator (or a single shard).
//
//   strata_load insert --target host:port --data DIR --ids-out ids.u32
//                      [--batch 1000] [--ca ca.pem --token-file token]
//   strata_load query  --target host:port --data DIR --ids ids.u32 --queries 20000
//                      (--concurrency N | --rate QPS) [--k 10] [--ef-search 64] [--warmup 1000]
//                      [--channels 4] [--ca ca.pem --token-file token]
//
// insert loads base.fbin through InsertBatch, in order, and writes the id the service assigned to
// each base row (uint32 per row). A batch that partially fails is retried with the ids it got, so
// a retry never inserts a vector twice (see the proto); after 20 failed attempts it gives up.
//
// query sends the dataset's queries (cycling) and prints one JSON object: throughput, latency
// percentiles, and recall@k against the ground truth (mapped through the insert's id file).
//   --concurrency N  closed loop: N workers, each sending its next query when the last returns.
//                    Measures capacity; latency is that at full load.
//   --rate R         open loop: query i is sent at start + i / R by one dispatcher thread using
//                    the async API, whatever happened before. Latency is measured from that due
//                    time, so queueing behind a slow query counts (no coordinated omission); the
//                    output reports how late the dispatcher was, which should stay near zero.
// Several channels (--channels) give several TCP connections, so one HTTP/2 connection is not the
// bottleneck at high load.

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "id_codec.hpp"
#include "security.hpp"
#include "strata/dataset.hpp"
#include "strata/recall.hpp"
#include "strata/v1/vector_service.grpc.pb.h"

namespace {

namespace fs = std::filesystem;
namespace v1 = strata::v1;
using Clock = std::chrono::steady_clock;

struct Args {
  std::string mode;
  std::string target;
  fs::path data;
  fs::path ids_file;
  fs::path ca_file;
  fs::path token_file;
  std::size_t batch = 1000;
  std::size_t queries = 20000;
  std::size_t warmup = 1000;
  int concurrency = 0;
  double rate = 0;
  int channels = 4;
  std::uint32_t k = 10;
  std::uint32_t ef_search = 64;
};

Args parse(int argc, char** argv) {
  if (argc < 2) {
    throw std::invalid_argument(
        "usage: strata_load insert|query --target host:port --data DIR ...");
  }
  Args args;
  args.mode = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const std::string value = argv[i + 1];
    if (flag == "--target") {
      args.target = value;
    } else if (flag == "--data") {
      args.data = value;
    } else if (flag == "--ids-out" || flag == "--ids") {
      args.ids_file = value;
    } else if (flag == "--ca") {
      args.ca_file = value;
    } else if (flag == "--token-file") {
      args.token_file = value;
    } else if (flag == "--batch") {
      args.batch = std::stoul(value);
    } else if (flag == "--queries") {
      args.queries = std::stoul(value);
    } else if (flag == "--warmup") {
      args.warmup = std::stoul(value);
    } else if (flag == "--concurrency") {
      args.concurrency = std::stoi(value);
    } else if (flag == "--rate") {
      args.rate = std::stod(value);
    } else if (flag == "--channels") {
      args.channels = std::stoi(value);
    } else if (flag == "--k") {
      args.k = static_cast<std::uint32_t>(std::stoul(value));
    } else if (flag == "--ef-search") {
      args.ef_search = static_cast<std::uint32_t>(std::stoul(value));
    } else {
      throw std::invalid_argument("unknown flag " + flag);
    }
  }
  if ((args.mode != "insert" && args.mode != "query") || args.target.empty() || args.data.empty() ||
      args.ids_file.empty() || args.batch == 0 || args.channels <= 0) {
    throw std::invalid_argument("missing or invalid arguments; see the comment in load_client.cpp");
  }
  if (args.mode == "query" && (args.concurrency > 0) == (args.rate > 0)) {
    throw std::invalid_argument("query needs exactly one of --concurrency and --rate");
  }
  return args;
}

std::shared_ptr<grpc::ChannelCredentials> credentials(const Args& args) {
  strata::server::ShardClientConfig config{args.ca_file, args.token_file};
  if (auto refused = strata::server::shard_client_error(config)) {
    throw std::invalid_argument(*refused);
  }
  auto creds = strata::server::make_shard_credentials(config);
  if (!creds) {
    throw std::runtime_error(creds.error().message);
  }
  return *creds;
}

std::vector<std::unique_ptr<v1::VectorService::Stub>> stubs(const Args& args, int count) {
  const auto creds = credentials(args);
  std::vector<std::unique_ptr<v1::VectorService::Stub>> out;
  for (int c = 0; c < count; ++c) {
    grpc::ChannelArguments channel_args;
    // A private subchannel pool: each channel gets its own connection instead of sharing one.
    channel_args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
    channel_args.SetInt("strata.channel_index", c);
    out.push_back(
        v1::VectorService::NewStub(grpc::CreateCustomChannel(args.target, creds, channel_args)));
  }
  return out;
}

int run_insert(const Args& args) {
  auto dataset = strata::load_dataset(args.data);
  if (!dataset) {
    throw std::runtime_error(dataset.error().message);
  }
  const strata::Matrix<float>& base = dataset->base;
  auto stub = std::move(stubs(args, 1)[0]);
  std::vector<std::uint32_t> ids(base.rows(), strata::server::kUnassignedId);
  const auto start = Clock::now();
  std::size_t retries = 0;
  for (std::size_t first = 0; first < base.rows(); first += args.batch) {
    const std::size_t count = std::min(args.batch, base.rows() - first);
    v1::InsertBatchRequest request;
    request.set_count(static_cast<std::uint32_t>(count));
    for (std::size_t i = first; i < first + count; ++i) {
      const auto row = base.row(i);
      request.mutable_values()->Add(row.begin(), row.end());
    }
    for (int attempt = 0;; ++attempt) {
      grpc::ClientContext context;
      v1::InsertBatchResponse response;
      const grpc::Status status = stub->InsertBatch(&context, request, &response);
      if (!status.ok()) {
        throw std::runtime_error("InsertBatch rejected: " + status.error_message());
      }
      if (response.error_code() == 0) {
        std::copy(response.ids().begin(), response.ids().end(),
                  ids.begin() + static_cast<std::ptrdiff_t>(first));
        break;
      }
      if (attempt == 20) {
        throw std::runtime_error("InsertBatch still failing: " + response.error_message());
      }
      ++retries;
      request.mutable_ids()->CopyFrom(response.ids());  // idempotent retry
      std::this_thread::sleep_for(std::chrono::milliseconds(200 * (attempt + 1)));
    }
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
  std::ofstream out(args.ids_file, std::ios::binary);
  out.write(reinterpret_cast<const char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(std::uint32_t)));
  if (!out) {
    throw std::runtime_error("cannot write " + args.ids_file.string());
  }
  std::cout << "{\"mode\": \"insert\", \"vectors\": " << base.rows() << ", \"seconds\": " << seconds
            << ", \"vectors_per_second\": " << static_cast<double>(base.rows()) / seconds
            << ", \"batch\": " << args.batch << ", \"retried_batches\": " << retries << "}\n";
  return 0;
}

double percentile(std::vector<double>& sorted, double p) {
  const auto index = static_cast<std::size_t>(p * static_cast<double>(sorted.size() - 1) + 0.5);
  return sorted[std::min(index, sorted.size() - 1)];
}

int run_query(const Args& args) {
  auto dataset = strata::load_dataset(args.data);
  if (!dataset) {
    throw std::runtime_error(dataset.error().message);
  }
  // Service id -> base row, from the insert's id file.
  std::vector<std::uint32_t> ids(dataset->base.rows());
  {
    std::ifstream in(args.ids_file, std::ios::binary);
    in.read(reinterpret_cast<char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(std::uint32_t)));
    if (!in) {
      throw std::runtime_error("cannot read " + args.ids_file.string() + " for this dataset");
    }
  }
  std::unordered_map<std::uint32_t, std::uint32_t> row_of;
  row_of.reserve(ids.size());
  for (std::uint32_t row = 0; row < ids.size(); ++row) {
    row_of[ids[row]] = row;
  }

  const std::size_t num_queries = dataset->query.rows();
  std::vector<v1::SearchRequest> requests(num_queries);
  for (std::size_t q = 0; q < num_queries; ++q) {
    const auto row = dataset->query.row(q);
    requests[q].mutable_values()->Add(row.begin(), row.end());
    requests[q].set_k(args.k);
    requests[q].set_ef_search(args.ef_search);
  }
  const auto channel_stubs = stubs(args, args.channels);

  // Recall over one pass of the dataset's queries (also warms connections and caches).
  std::vector<std::vector<strata::Neighbor>> results(num_queries);
  for (std::size_t q = 0; q < num_queries; ++q) {
    grpc::ClientContext context;
    v1::SearchResponse response;
    const grpc::Status status = channel_stubs[0]->Search(&context, requests[q], &response);
    if (!status.ok()) {
      throw std::runtime_error("search failed: " + status.error_message());
    }
    for (const auto& n : response.neighbors()) {
      const auto it = row_of.find(n.id());
      results[q].push_back(
          {it == row_of.end() ? strata::server::kUnassignedId : it->second, n.distance()});
    }
  }
  auto recall = strata::recall_at_k(results, dataset->groundtruth, args.k);
  if (!recall) {
    throw std::runtime_error(recall.error().message);
  }

  // Timed phase. Queries [0, warmup) are sent but not recorded.
  const std::size_t total = args.warmup + args.queries;
  const bool open_loop = args.rate > 0;
  std::atomic<std::size_t> errors{0};
  std::vector<std::vector<double>> latencies;
  std::vector<std::vector<double>> lateness;
  const auto start = Clock::now() + std::chrono::milliseconds(100);

  if (open_loop) {
    // One dispatcher thread issues every query at its due time with the async API; completions
    // are recorded on gRPC's threads. Nothing waits for a reply before sending the next query,
    // so a slow reply cannot hold back later sends. The dispatcher sleeps until just before each
    // due time and spins the rest: thread sleeps can overshoot by milliseconds under load (and
    // macOS coalesces timers), which would otherwise show up as server latency.
    struct Call {
      grpc::ClientContext context;
      v1::SearchResponse response;
      Clock::time_point due;
      Clock::time_point sent;
    };
    std::vector<Call> calls(total);
    std::vector<double> latency(total, -1.0);
    std::atomic<std::size_t> completed{0};
    for (std::size_t i = 0; i < total; ++i) {
      Call& call = calls[i];
      call.due = start + std::chrono::duration_cast<Clock::duration>(
                             std::chrono::duration<double>(static_cast<double>(i) / args.rate));
      std::this_thread::sleep_until(call.due - std::chrono::milliseconds(1));
      while (Clock::now() < call.due) {
      }
      call.sent = Clock::now();
      channel_stubs[i % channel_stubs.size()]->async()->Search(
          &call.context, &requests[i % num_queries], &call.response, [&, i](grpc::Status status) {
            if (status.ok()) {
              latency[i] =
                  std::chrono::duration<double, std::micro>(Clock::now() - calls[i].due).count();
            } else {
              ++errors;
            }
            completed.fetch_add(1, std::memory_order_release);
          });
    }
    while (completed.load(std::memory_order_acquire) < total) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    latencies.emplace_back();
    lateness.emplace_back();
    for (std::size_t i = args.warmup; i < total; ++i) {
      if (latency[i] >= 0) {
        latencies.back().push_back(latency[i]);
        lateness.back().push_back(
            std::chrono::duration<double, std::micro>(calls[i].sent - calls[i].due).count());
      }
    }
  } else {
    // Closed loop: each worker sends its next query as soon as the last one returns.
    const auto workers = static_cast<std::size_t>(args.concurrency);
    latencies.resize(workers);
    lateness.resize(workers);
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> threads;
    for (std::size_t w = 0; w < workers; ++w) {
      threads.emplace_back([&, w] {
        auto& stub = *channel_stubs[w % channel_stubs.size()];
        std::this_thread::sleep_until(start);  // every worker starts at the clock's zero
        for (std::size_t i = next++; i < total; i = next++) {
          const auto sent = Clock::now();
          grpc::ClientContext context;
          v1::SearchResponse response;
          if (!stub.Search(&context, requests[i % num_queries], &response).ok()) {
            ++errors;
            continue;
          }
          if (i >= args.warmup) {
            latencies[w].push_back(
                std::chrono::duration<double, std::micro>(Clock::now() - sent).count());
            lateness[w].push_back(0.0);
          }
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();

  std::vector<double> all;
  std::vector<double> all_late;
  for (std::size_t w = 0; w < latencies.size(); ++w) {
    all.insert(all.end(), latencies[w].begin(), latencies[w].end());
    all_late.insert(all_late.end(), lateness[w].begin(), lateness[w].end());
  }
  if (all.empty()) {
    throw std::runtime_error("no successful queries");
  }
  std::sort(all.begin(), all.end());
  std::sort(all_late.begin(), all_late.end());
  double sum = 0;
  for (double v : all) {
    sum += v;
  }
  std::cout << "{\"mode\": \"query\", \"loop\": \"" << (open_loop ? "open" : "closed") << "\""
            << ", \"concurrency\": " << args.concurrency << ", \"target_rate\": " << args.rate
            << ", \"queries\": " << all.size() << ", \"errors\": " << errors.load()
            << ", \"seconds\": " << seconds
            << ", \"qps\": " << static_cast<double>(total - errors.load()) / seconds
            << ", \"k\": " << args.k << ", \"ef_search\": " << args.ef_search
            << ", \"recall\": " << *recall << ", \"channels\": " << args.channels
            << ", \"mean_us\": " << sum / static_cast<double>(all.size())
            << ", \"p50_us\": " << percentile(all, 0.50)
            << ", \"p90_us\": " << percentile(all, 0.90)
            << ", \"p99_us\": " << percentile(all, 0.99)
            << ", \"p999_us\": " << percentile(all, 0.999) << ", \"max_us\": " << all.back()
            << ", \"send_lateness_p99_us\": " << percentile(all_late, 0.99) << "}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    return args.mode == "insert" ? run_insert(args) : run_query(args);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
