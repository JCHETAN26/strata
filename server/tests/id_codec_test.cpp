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

}  // namespace
}  // namespace strata::server
