#include "util/bytes.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace strata::util {
namespace {

TEST(CopyBytes, CopiesExactlyNBytes) {
  const std::vector<std::uint8_t> src{1, 2, 3, 4, 5};
  std::vector<std::uint8_t> dest(5, 0);
  copy_bytes(dest.data(), src.data(), 3);
  EXPECT_EQ(dest, (std::vector<std::uint8_t>{1, 2, 3, 0, 0}));
}

// The case the helper exists for: empty vectors have a null data(). Under the asan preset
// (UBSan), a bare memcpy here is reported on glibc. The length is a runtime value, as at the real
// call sites: GCC deletes a memcpy whose length is the constant 0 before UBSan can check it.
TEST(CopyBytes, ZeroLengthWithNullPointersIsANoOp) {
  std::vector<float> empty_src;
  std::vector<float> empty_dest;
  copy_bytes(empty_dest.data(), empty_src.data(), empty_src.size() * sizeof(float));
  EXPECT_TRUE(empty_dest.empty());
}

}  // namespace
}  // namespace strata::util
