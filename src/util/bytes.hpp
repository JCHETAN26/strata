#pragma once

// Internal byte-copy helper. Every raw byte copy in Strata (src and tests) goes through
// copy_bytes instead of calling std::memcpy or std::memmove directly. Not part of the public API.

#include <cstddef>
#include <cstring>

namespace strata::util {

// std::memcpy, except that a zero-length copy is a no-op. memcpy with a null pointer is undefined
// even when n == 0, and an empty std::vector or std::span may have a null data(). glibc declares
// memcpy's arguments nonnull, so UBSan reports this on Linux; macOS's libc does not, so it goes
// unnoticed there. The ranges must not overlap (as for memcpy). For a constant n (the common
// sizeof(T) case) the check folds away, so this costs nothing over memcpy.
inline void copy_bytes(void* dest, const void* src, std::size_t n) noexcept {
  if (n != 0) {
    std::memcpy(dest, src, n);
  }
}

}  // namespace strata::util
