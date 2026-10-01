#pragma once

#include <grpcpp/grpcpp.h>

#include "strata/error.hpp"

namespace strata::server {

// Translates a Strata Error into the closest gRPC status code, so a client sees a meaningful code
// (INVALID_ARGUMENT vs. INTERNAL) rather than everything as UNKNOWN.
inline grpc::Status to_status(const Error& error) {
  grpc::StatusCode code = grpc::StatusCode::UNKNOWN;
  switch (error.code) {
    case ErrorCode::kInvalidArgument:
    case ErrorCode::kDimensionMismatch:
      code = grpc::StatusCode::INVALID_ARGUMENT;
      break;
    case ErrorCode::kNotFound:
      code = grpc::StatusCode::NOT_FOUND;
      break;
    case ErrorCode::kIoError:
    case ErrorCode::kCorruptData:
      code = grpc::StatusCode::INTERNAL;
      break;
  }
  return grpc::Status(code, error.message);
}

}  // namespace strata::server
