#include "id_codec.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace strata::server {
namespace {

TEST(ShardIdCodec, SingleShardIsIdentity) {
  ShardIdCodec codec(1);
  EXPECT_EQ(codec.num_shards(), 1u);
  for (VectorId local = 0; local < 1000; local += 37) {
    const VectorId global = codec.to_global(0, local);
    EXPECT_EQ(global, local);
    EXPECT_EQ(codec.shard_of(global), 0u);
    EXPECT_EQ(codec.to_local(global), local);
  }
}

TEST(ShardIdCodec, ZeroShardsTreatedAsOne) {
  ShardIdCodec codec(0);
  EXPECT_EQ(codec.num_shards(), 1u);
}

TEST(ShardIdCodec, RoundTripsAcrossShards) {
  const std::uint32_t num_shards = 4;
  ShardIdCodec codec(num_shards);
  for (std::uint32_t shard = 0; shard < num_shards; ++shard) {
    for (VectorId local = 0; local < 500; ++local) {
      const VectorId global = codec.to_global(shard, local);
      EXPECT_EQ(codec.shard_of(global), shard);
      EXPECT_EQ(codec.to_local(global), local);
    }
  }
}

TEST(ShardIdCodec, GlobalIdsAreUniqueAndInterleaved) {
  const std::uint32_t num_shards = 3;
  ShardIdCodec codec(num_shards);
  // The first local id on each shard maps to a distinct small global id (interleaving).
  EXPECT_EQ(codec.to_global(0, 0), 0u);
  EXPECT_EQ(codec.to_global(1, 0), 1u);
  EXPECT_EQ(codec.to_global(2, 0), 2u);
  EXPECT_EQ(codec.to_global(0, 1), 3u);
  EXPECT_EQ(codec.to_global(1, 1), 4u);
}

TEST(ShardIdCodec, LocalLimitKeepsGlobalIdsInside32Bits) {
  for (std::uint32_t n : {1u, 2u, 3u, 4u, 7u, 64u, 1000u}) {
    const ShardIdCodec codec(n);
    for (std::uint32_t s = 0; s < n; ++s) {
      const std::uint64_t limit = codec.local_limit(s);
      ASSERT_GT(limit, 0u);
      // The last valid local id maps below the reserved id; the limit itself would not.
      EXPECT_LT(codec.to_global(s, static_cast<VectorId>(limit - 1)), kUnassignedId);
      EXPECT_GE(limit * n + s, std::uint64_t{kUnassignedId}) << "n=" << n << " s=" << s;
    }
  }
  EXPECT_EQ(ShardIdCodec(1).local_limit(0), kUnassignedId);
}

}  // namespace
}  // namespace strata::server
