#pragma once

// Versions of the RPC stack this binary was compiled against, printed at startup so a benchmark
// log records them. On the Mac they come from Homebrew; on Linux from the vcpkg manifest, and the
// two can drift, so the log is the record of which one a result used.

#include <absl/base/config.h>
#include <google/protobuf/stubs/common.h>
#include <grpcpp/grpcpp.h>

#include <string>

namespace strata::server {

inline std::string rpc_versions() {
  // GOOGLE_PROTOBUF_VERSION packs the C++ runtime version as major*1e6 + minor*1e3 + patch.
  constexpr int kProtobuf = GOOGLE_PROTOBUF_VERSION;
  return "grpc " + grpc::Version() + ", protobuf " + std::to_string(kProtobuf / 1000000) + "." +
         std::to_string(kProtobuf / 1000 % 1000) + "." + std::to_string(kProtobuf % 1000) +
         ", abseil " + std::to_string(ABSL_LTS_RELEASE_VERSION);
}

}  // namespace strata::server
