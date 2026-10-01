// strata_coordinator: the sharding front end. Presents one VectorService endpoint and fans out to
// the shards named on the command line.
//
//   strata_coordinator --dim <n> [--metric l2|inner_product|cosine] [--port 50050]
//                       --shard host:port [--shard host:port ...]
//
// The order of --shard flags is significant and must be stable across restarts: a shard's position
// in the list is encoded into every global id it produced (see server/id_codec.hpp).

#include <grpcpp/grpcpp.h>

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
#include "strata/distance.hpp"

namespace {

struct Args {
  std::size_t dim = 0;
  strata::Metric metric = strata::Metric::kL2;
  int port = 50050;
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
      if (flag == "--dim") {
        args.dim = static_cast<std::size_t>(std::stoul(next()));
      } else if (flag == "--metric") {
        auto m = strata::parse_metric(next());
        if (!m) {
          std::cerr << "unknown metric\n";
          return std::nullopt;
        }
        args.metric = *m;
      } else if (flag == "--port") {
        args.port = std::stoi(next());
      } else if (flag == "--shard") {
        args.shards.push_back(next());
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
                 "[--port 50050] --shard host:port [--shard host:port ...]\n";
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

  strata::server::CoordinatorService service(args.shards, args.dim, args.metric);

  const std::string address = "0.0.0.0:" + std::to_string(args.port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "failed to bind " << address << "\n";
    return 1;
  }
  std::cout << "strata_coordinator listening on " << address << " over " << args.shards.size()
            << " shard(s) [" << strata::server::rpc_versions() << "]"
            << std::endl;  // flush: stdout may be a log file
  server->Wait();
  return 0;
}
