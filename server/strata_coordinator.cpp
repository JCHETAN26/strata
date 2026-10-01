// strata_coordinator: the sharding front end. Presents one VectorService endpoint and fans out to
// the shards named on the command line.
//
//   strata_coordinator --dim <n> [--metric l2|inner_product|cosine] [--port 50050]
//                       --shard host:port [--shard host:port ...]
//                       [--shard-ca <pem>] [--shard-token-file <path>] [--shard-timeout-ms 10000]
//                       [--listen 127.0.0.1] [--tls-cert <pem> --tls-key <pem>]
//                       [--token-file <path>] [--insecure] [--max-threads <n>]
//
// --shard-ca turns on TLS to the shards (their certificates must chain to it and name the host in
// --shard); --shard-token-file sends the shards' shared token. The --listen group configures the
// coordinator's own client-facing port, under the same rules as a shard (server/security.hpp).
//
// The order of --shard flags is significant and must be stable across restarts: a shard's position
// in the list is encoded into every global id it produced (see server/id_codec.hpp).

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "build_info.hpp"
#include "coordinator_service.hpp"
#include "security.hpp"
#include "strata/distance.hpp"

namespace {

struct Args {
  std::size_t dim = 0;
  strata::Metric metric = strata::Metric::kL2;
  strata::server::ListenConfig listen{.port = 50050};
  strata::server::ShardClientConfig shard_client;
  long shard_timeout_ms = 10'000;
  std::vector<std::string> shards;
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
      if (flag == "--dim") {
        args.dim = static_cast<std::size_t>(std::stoul(next()));
      } else if (flag == "--metric") {
        auto m = strata::parse_metric(next());
        if (!m) {
          std::cerr << "unknown metric\n";
          return std::nullopt;
        }
        args.metric = *m;
      } else if (flag == "--shard") {
        args.shards.push_back(next());
      } else if (flag == "--shard-ca") {
        args.shard_client.ca_file = next();
      } else if (flag == "--shard-token-file") {
        args.shard_client.token_file = next();
      } else if (flag == "--shard-timeout-ms") {
        args.shard_timeout_ms = std::stol(next());
      } else {
        std::cerr << "unknown flag: " << flag << "\n";
        return std::nullopt;
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return std::nullopt;
    }
  }
  if (args.dim == 0 || args.shards.empty()) {
    std::cerr << "usage: strata_coordinator --dim <n> [--metric l2|inner_product|cosine] "
                 "[--port 50050] --shard host:port [--shard host:port ...]\n"
                 "  --shard-ca <pem>         use TLS to the shards, trusting this CA\n"
                 "  --shard-token-file <p>   send this token to the shards (needs --shard-ca)\n"
                 "  --shard-timeout-ms <n>   deadline for each shard call (default 10000)\n"
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

  if (auto refused = strata::server::exposure_error(args.listen)) {
    std::cerr << *refused << "\n";
    return 1;
  }
  if (auto refused = strata::server::shard_client_error(args.shard_client)) {
    std::cerr << *refused << "\n";
    return 1;
  }
  if (args.shard_timeout_ms <= 0) {
    std::cerr << "--shard-timeout-ms must be positive\n";
    return 1;
  }
  auto shard_credentials = strata::server::make_shard_credentials(args.shard_client);
  if (!shard_credentials) {
    std::cerr << shard_credentials.error().message << "\n";
    return 1;
  }
  strata::server::CoordinatorOptions options;
  options.shard_credentials = *shard_credentials;
  options.shard_timeout = std::chrono::milliseconds(args.shard_timeout_ms);
  strata::server::CoordinatorService service(args.shards, args.dim, args.metric, options);

  int port = 0;
  auto server = strata::server::start_server(args.listen, &service, &port);
  if (!server) {
    std::cerr << server.error().message << "\n";
    return 1;
  }
  const std::string address = strata::server::host_port(args.listen.host, port);
  const char* to_shards = !args.shard_client.token_file.empty() ? "tls+token"
                          : !args.shard_client.ca_file.empty()  ? "tls"
                                                                : "plaintext";
  std::cout << "strata_coordinator listening on " << address << " over " << args.shards.size()
            << " shard(s), " << to_shards << " to shards [" << strata::server::rpc_versions() << "]"
            << std::endl;  // flush: may be a log file
  (*server)->Wait();
  return 0;
}
