// strata_shard: a standalone vector-search node, or one shard behind a coordinator.
//
//   strata_shard --dir <path> --dim <n> [--metric l2|inner_product|cosine]
//                [--index flat|hnsw] [--port 50051]
//                [--hnsw-m 16] [--hnsw-ef-construction 200]
//                [--listen 127.0.0.1] [--tls-cert <pem> --tls-key <pem>] [--token-file <path>]
//                [--insecure] [--max-threads <n>]
//
// Listens on 127.0.0.1 unless --listen says otherwise; see server/security.hpp for the rules on
// exposing it (TLS and a shared token, or an explicit --insecure).
//
// The collection is durable in --dir: it recovers from the snapshot and WAL on start, so a shard
// can be killed and restarted without losing acknowledged writes.

#include <grpcpp/grpcpp.h>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "build_info.hpp"
#include "security.hpp"
#include "shard_service.hpp"
#include "strata/collection.hpp"
#include "strata/distance.hpp"

namespace {

// Minimal --flag value parser: every flag takes one argument. Unknown flags are an error so a
// typo fails loudly instead of being silently ignored.
struct Args {
  std::string dir;
  std::size_t dim = 0;
  strata::Metric metric = strata::Metric::kL2;
  strata::IndexKind index = strata::IndexKind::kFlat;
  strata::server::ListenConfig listen{.port = 50051};
  std::size_t hnsw_m = 16;
  std::size_t hnsw_ef_construction = 200;
};

std::optional<Args> parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string_view flag = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + std::string(flag));
      }
      return argv[++i];
    };
    try {
      if (strata::server::parse_listen_flag(flag, next, args.listen)) {
        continue;
      }
      if (flag == "--dir") {
        args.dir = next();
      } else if (flag == "--dim") {
        args.dim = static_cast<std::size_t>(std::stoul(next()));
      } else if (flag == "--metric") {
        auto m = strata::parse_metric(next());
        if (!m) {
          std::cerr << "unknown metric\n";
          return std::nullopt;
        }
        args.metric = *m;
      } else if (flag == "--index") {
        std::string v = next();
        if (v == "flat") {
          args.index = strata::IndexKind::kFlat;
        } else if (v == "hnsw") {
          args.index = strata::IndexKind::kHnsw;
        } else {
          std::cerr << "unknown index kind: " << v << "\n";
          return std::nullopt;
        }
      } else if (flag == "--hnsw-m") {
        args.hnsw_m = static_cast<std::size_t>(std::stoul(next()));
      } else if (flag == "--hnsw-ef-construction") {
        args.hnsw_ef_construction = static_cast<std::size_t>(std::stoul(next()));
      } else {
        std::cerr << "unknown flag: " << flag << "\n";
        return std::nullopt;
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return std::nullopt;
    }
  }
  if (args.dir.empty() || args.dim == 0) {
    std::cerr << "usage: strata_shard --dir <path> --dim <n> [--metric l2|inner_product|cosine] "
                 "[--index flat|hnsw] [--port 50051] [--hnsw-m 16] [--hnsw-ef-construction 200]\n"
              << strata::server::kListenUsage;
    return std::nullopt;
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  auto parsed = parse_args(argc, argv);
  if (!parsed) {
    return 1;
  }
  const Args& args = *parsed;
  // Refuse an unsafe exposure before doing any work (opening may replay a long WAL).
  if (auto refused = strata::server::exposure_error(args.listen)) {
    std::cerr << *refused << "\n";
    return 1;
  }

  strata::CollectionOptions options;
  options.index = args.index;
  options.hnsw.M = args.hnsw_m;
  options.hnsw.ef_construction = args.hnsw_ef_construction;

  auto collection = strata::Collection::open(args.dir, args.dim, args.metric, options);
  if (!collection) {
    std::cerr << "failed to open collection: " << collection.error().message << "\n";
    return 1;
  }

  strata::server::ShardService service(std::move(*collection), args.dim, args.metric);

  int port = 0;
  auto server = strata::server::start_server(args.listen, &service, &port);
  if (!server) {
    std::cerr << server.error().message << "\n";
    return 1;
  }
  const std::string address = strata::server::host_port(args.listen.host, port);
  const char* security = !args.listen.token_file.empty()      ? "tls+token"
                         : !args.listen.tls_cert_file.empty() ? "tls"
                                                              : "plaintext";
  std::cout << "strata_shard listening on " << address << " (dir=" << args.dir
            << ", dim=" << args.dim
            << ", index=" << (args.index == strata::IndexKind::kHnsw ? "hnsw" : "flat") << ", "
            << security << ") [" << strata::server::rpc_versions() << "]"
            << std::endl;  // flush: stdout may be a log file
  (*server)->Wait();
  return 0;
}
