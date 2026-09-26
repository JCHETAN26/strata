#pragma once

#include <string>
#include <tl/expected.hpp>
#include <utility>

namespace strata {

enum class ErrorCode {
  kInvalidArgument,
  kDimensionMismatch,
  kIoError,
  kCorruptData,
  kNotFound,
};

// Error returned across API boundaries. Hot paths do not throw; they return Expected<T>.
// Thread safety: value type, no shared state.
struct Error {
  ErrorCode code;
  std::string message;
};

// Drop-in for std::expected (C++23), which GCC 11 on Ubuntu 22.04 lacks.
template <typename T>
using Expected = tl::expected<T, Error>;

inline tl::unexpected<Error> make_error(ErrorCode code, std::string message) {
  return tl::unexpected<Error>(Error{code, std::move(message)});
}

}  // namespace strata
