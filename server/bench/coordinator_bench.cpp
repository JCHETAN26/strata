// Coordinator search latency: async (callback) fan-out vs. the original thread per shard per query.
//
//   strata_coordinator_bench --data data/sift1m-200k-q1000 --shards 4
//       [--rounds 5] [--queries 2000] [--clients 8] [--k 10] [--ef-search 64]
//       [--M 16] [--ef-construction 200]
//
// Everything runs in this one process, over real gRPC on loopback: `--shards` shard servers (HNSW
// collections holding the base vectors dealt round-robin, built in parallel), and two coordinator
// servers over the same shards, one per fan-out mode. A client stub sends each query to a
// coordinator and waits for the answer, so a latency is the whole round trip: client ->
// coordinator -> every shard -> merge -> client.
//
// Each round measures both modes back to back, alternating which goes first, so slow drift (heat,
// background load) affects both equally. Two workloads per mode:
//   sequential   one client, one query at a time: per-query latency.
//   concurrent   --clients threads, each sending queries back to back: throughput and latency
//                under load, where creating threads per query competes with the searches.
// Writes one JSON object to stdout; bench/run_coordinator_bench.py runs and summarizes it.

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "coordinator_service.hpp"
#include "shard_service.hpp"
#include "strata/collection.hpp"
#include "strata/dataset.hpp"
#include "strata/recall.hpp"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using strata::server::SearchFanout;

struct Args {
  fs::path data;
  std::uint32_t shards = 2;
  int rounds = 5;
  std::size_t queries = 2000;
  int clients = 8;
  std::uint32_t k = 10;
  std::uint32_t ef_search = 64;
  std::size_t M = 16;
  std::size_t ef_construction = 200;
};

Args parse(int argc, char** argv) {
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const std::string value = argv[i + 1];
    if (flag == "--data") {
      args.data = value;
    } else if (flag == "--shards") {
      args.shards = static_cast<std::uint32_t>(std::stoul(value));
    } else if (flag == "--rounds") {
      args.rounds = std::stoi(value);
    } else if (flag == "--queries") {
      args.queries = std::stoul(value);
    } else if (flag == "--clients") {
      args.clients = std::stoi(value);
    } else if (flag == "--k") {
      args.k = static_cast<std::uint32_t>(std::stoul(value));
    } else if (flag == "--ef-search") {
      args.ef_search = static_cast<std::uint32_t>(std::stoul(value));
    } else if (flag == "--M") {
      args.M = std::stoul(value);
    } else if (flag == "--ef-construction") {
      args.ef_construction = std::stoul(value);
    } else {
      throw std::invalid_argument("unknown flag " + flag);
    }
  }
  if (args.data.empty() || args.shards == 0 || argc % 2 == 0) {
    throw std::invalid_argument(
        "usage: strata_coordinator_bench --data DIR --shards N [--rounds 5] [--queries 2000] "
        "[--clients 8] [--k 10] [--ef-search 64] [--M 16] [--ef-construction 200]");
  }
  return args;
}

struct LiveServer {
  std::unique_ptr<grpc::Server> server;
  int port = 0;
};

LiveServer serve(grpc::Service* service) {
  LiveServer live;
  grpc::ServerBuilder builder;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &live.port);
  builder.RegisterService(service);
  live.server = builder.BuildAndStart();
  if (!live.server) {
    throw std::runtime_error("failed to start a server");
  }
  return live;
}

strata::v1::SearchRequest make_request(std::span<const float> query, const Args& args) {
  strata::v1::SearchRequest request;
  request.mutable_values()->Add(query.begin(), query.end());
  request.set_k(args.k);
  request.set_ef_search(args.ef_search);
  return request;
}

double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(p * static_cast<double>(values.size() - 1) + 0.5);
  return values[std::min(index, values.size() - 1)];
}

double mean(const std::vector<double>& values) {
  double sum = 0;
  for (double v : values) {
    sum += v;
  }
  return sum / static_cast<double>(values.size());
}

// Sends `count` queries (cycling through `requests`) one at a time; returns each latency in us.
std::vector<double> sequential(strata::v1::VectorService::Stub& stub,
                               const std::vector<strata::v1::SearchRequest>& requests,
                               std::size_t count) {
  std::vector<double> latencies;
  latencies.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    grpc::ClientContext context;
    strata::v1::SearchResponse response;
    const auto start = Clock::now();
    const grpc::Status status = stub.Search(&context, requests[i % requests.size()], &response);
    const auto stop = Clock::now();
    if (!status.ok()) {
      throw std::runtime_error("search failed: " + status.error_message());
    }
    latencies.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
  }
  return latencies;
}

struct ConcurrentResult {
  double qps = 0;
  std::vector<double> latencies_us;
};

// `clients` threads share `count` queries, each sending its next query as soon as the last returns.
ConcurrentResult concurrent(strata::v1::VectorService::Stub& stub,
                            const std::vector<strata::v1::SearchRequest>& requests,
                            std::size_t count, int clients) {
  std::atomic<std::size_t> next{0};
  std::vector<std::vector<double>> per_client(static_cast<std::size_t>(clients));
  std::atomic<bool> failed{false};
  const auto start = Clock::now();
  std::vector<std::thread> threads;
  for (int c = 0; c < clients; ++c) {
    threads.emplace_back([&, c] {
      auto& mine = per_client[static_cast<std::size_t>(c)];
      for (std::size_t i = next++; i < count; i = next++) {
        grpc::ClientContext context;
        strata::v1::SearchResponse response;
        const auto t0 = Clock::now();
        if (!stub.Search(&context, requests[i % requests.size()], &response).ok()) {
          failed = true;
          return;
        }
        mine.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
  if (failed) {
    throw std::runtime_error("a concurrent search failed");
  }
  ConcurrentResult result;
  for (auto& mine : per_client) {
    result.latencies_us.insert(result.latencies_us.end(), mine.begin(), mine.end());
  }
  result.qps = static_cast<double>(count) / seconds;
  return result;
}

void print_latency_fields(const std::vector<double>& us) {
  std::cout << "\"mean_us\": " << mean(us) << ", \"p50_us\": " << percentile(us, 0.50)
            << ", \"p95_us\": " << percentile(us, 0.95) << ", \"p99_us\": " << percentile(us, 0.99);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    auto dataset = strata::load_dataset(args.data);
    if (!dataset) {
      throw std::runtime_error(dataset.error().message);
    }
    const strata::Matrix<float>& base = dataset->base;
    const std::size_t dim = base.cols();

    // Shards: vector i goes to shard i % S, so its local id is i / S and its global id is
    // (i / S) * S + i % S = i, the base index, which lets recall use the groundtruth directly.
    const fs::path root = fs::temp_directory_path() / "strata_coordinator_bench";
    fs::remove_all(root);
    std::vector<std::unique_ptr<strata::server::ShardService>> services(args.shards);
    const auto build_start = Clock::now();
    {
      std::vector<std::thread> builders;
      std::atomic<bool> failed{false};
      for (std::uint32_t s = 0; s < args.shards; ++s) {
        builders.emplace_back([&, s] {
          const fs::path dir = root / ("shard" + std::to_string(s));
          fs::create_directories(dir);
          strata::CollectionOptions options;
          options.sync = strata::SyncMode::kNone;  // durability is not what is measured here
          options.index = strata::IndexKind::kHnsw;
          options.hnsw.M = args.M;
          options.hnsw.ef_construction = args.ef_construction;
          auto collection = strata::Collection::open(dir, dim, strata::Metric::kL2, options);
          if (!collection) {
            failed = true;
            return;
          }
          for (std::size_t i = s; i < base.rows(); i += args.shards) {
            if (!collection->insert(base.row(i))) {
              failed = true;
              return;
            }
          }
          services[s] = std::make_unique<strata::server::ShardService>(std::move(*collection), dim,
                                                                       strata::Metric::kL2);
        });
      }
      for (auto& b : builders) {
        b.join();
      }
      if (failed) {
        throw std::runtime_error("failed to build a shard");
      }
    }
    const double build_seconds = std::chrono::duration<double>(Clock::now() - build_start).count();

    std::vector<LiveServer> shard_servers;
    std::vector<std::string> addresses;
    for (auto& service : services) {
      shard_servers.push_back(serve(service.get()));
      addresses.push_back("127.0.0.1:" + std::to_string(shard_servers.back().port));
    }

    struct Mode {
      const char* name;
      std::unique_ptr<strata::server::CoordinatorService> service;
      LiveServer server;
      std::unique_ptr<strata::v1::VectorService::Stub> stub;
    };
    std::vector<Mode> modes;
    for (auto [name, fanout] : {std::pair{"async", SearchFanout::kAsync},
                                std::pair{"thread_per_shard", SearchFanout::kThreadPerShard}}) {
      strata::server::CoordinatorOptions options;
      options.search_fanout = fanout;
      Mode mode;
      mode.name = name;
      mode.service = std::make_unique<strata::server::CoordinatorService>(
          addresses, dim, strata::Metric::kL2, options);
      mode.server = serve(mode.service.get());
      mode.stub = strata::v1::VectorService::NewStub(grpc::CreateChannel(
          "127.0.0.1:" + std::to_string(mode.server.port), grpc::InsecureChannelCredentials()));
      modes.push_back(std::move(mode));
    }

    std::vector<strata::v1::SearchRequest> requests;
    for (std::size_t q = 0; q < dataset->query.rows(); ++q) {
      requests.push_back(make_request(dataset->query.row(q), args));
    }

    // Recall through each coordinator (they must agree), also a warm-up for connections and caches.
    std::vector<double> recalls;
    for (Mode& mode : modes) {
      std::vector<std::vector<strata::Neighbor>> results(requests.size());
      for (std::size_t q = 0; q < requests.size(); ++q) {
        grpc::ClientContext context;
        strata::v1::SearchResponse response;
        if (!mode.stub->Search(&context, requests[q], &response).ok()) {
          throw std::runtime_error("warm-up search failed");
        }
        for (const auto& n : response.neighbors()) {
          results[q].push_back({n.id(), n.distance()});
        }
      }
      auto recall = strata::recall_at_k(results, dataset->groundtruth, args.k);
      if (!recall) {
        throw std::runtime_error(recall.error().message);
      }
      recalls.push_back(*recall);
    }

#ifdef NDEBUG
    const bool asserts = false;
#else
    const bool asserts = true;
#endif
    std::cout << "{\"shards\": " << args.shards << ", \"num_base\": " << base.rows()
              << ", \"dim\": " << dim << ", \"num_queries\": " << requests.size()
              << ", \"queries_per_pass\": " << args.queries << ", \"clients\": " << args.clients
              << ", \"k\": " << args.k << ", \"ef_search\": " << args.ef_search
              << ", \"M\": " << args.M << ", \"ef_construction\": " << args.ef_construction
              << ", \"build_seconds\": " << build_seconds
              << ", \"hardware_concurrency\": " << std::thread::hardware_concurrency()
              << ", \"asserts\": " << (asserts ? "true" : "false") << ", \"recall\": {\""
              << modes[0].name << "\": " << recalls[0] << ", \"" << modes[1].name
              << "\": " << recalls[1] << "}, \"rounds\": [";
    for (int r = 0; r < args.rounds; ++r) {
      std::cout << (r == 0 ? "" : ", ") << "{";
      // Alternate which mode goes first.
      for (std::size_t m = 0; m < modes.size(); ++m) {
        Mode& mode = modes[(m + static_cast<std::size_t>(r)) % modes.size()];
        const std::vector<double> seq = sequential(*mode.stub, requests, args.queries);
        const ConcurrentResult conc =
            concurrent(*mode.stub, requests, args.queries * 2, args.clients);
        std::cout << (m == 0 ? "" : ", ") << "\"" << mode.name << "\": {\"sequential\": {";
        print_latency_fields(seq);
        std::cout << "}, \"concurrent\": {\"qps\": " << conc.qps << ", ";
        print_latency_fields(conc.latencies_us);
        std::cout << "}}";
      }
      std::cout << "}";
    }
    std::cout << "]}\n";

    for (Mode& mode : modes) {
      mode.server.server->Shutdown();
    }
    for (LiveServer& live : shard_servers) {
      live.server->Shutdown();
    }
    services.clear();
    fs::remove_all(root);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
