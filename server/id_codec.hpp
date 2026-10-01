#pragma once

#include <cstdint>

#include "strata/types.hpp"

namespace strata::server {

// Maps between a shard's dense local ids and the global ids clients see.
//
// Each shard's Collection assigns dense VectorIds (0, 1, 2, ...) in its own insertion order, so
// local ids collide across shards. The coordinator interleaves them into a single global space:
//
//     global_id = local_id * num_shards + shard_index
//
// This is a bijection for a fixed num_shards, so it decodes without any lookup table: the shard is
// the remainder and the local id is the quotient. Interleaving (rather than partitioning ranges)
// means global ids stay small from the first insert on every shard, and no shard needs to know how
// many vectors the others hold.
//
// num_shards == 1 (a lone shard, no coordinator) makes global and local ids identical.
class ShardIdCodec {
 public:
  explicit ShardIdCodec(std::uint32_t num_shards) : num_shards_(num_shards == 0 ? 1 : num_shards) {}

  [[nodiscard]] std::uint32_t num_shards() const noexcept { return num_shards_; }

  [[nodiscard]] VectorId to_global(std::uint32_t shard_index, VectorId local_id) const noexcept {
    return static_cast<VectorId>(local_id * num_shards_ + shard_index);
  }

  [[nodiscard]] std::uint32_t shard_of(VectorId global_id) const noexcept {
    return static_cast<std::uint32_t>(global_id % num_shards_);
  }

  [[nodiscard]] VectorId to_local(VectorId global_id) const noexcept {
    return static_cast<VectorId>(global_id / num_shards_);
  }

 private:
  std::uint32_t num_shards_;
};

}  // namespace strata::server
