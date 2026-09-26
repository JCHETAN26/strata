#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/matrix.hpp"

namespace strata {

// Point-in-time image of a collection: every vector (by id), tombstones, and the LSN of the last
// WAL record it includes. Index-independent: indexes rebuild from it (brute force trivially; an
// HNSW snapshot can add its graph later).
//
// On-disk format (little-endian):
//   char[8] magic "STRSNP\0\1", u32 version (1), u32 metric, u32 dim, u32 zero,
//   u64 count, u64 last_lsn, u64 num_deleted,
//   count x dim f32 vectors, ceil(count / 8) bytes tombstone bitmap (bit i = id i deleted),
//   u32 crc32c of everything before it.
//
// Written atomically (temp file + sync + rename + directory sync).
struct Snapshot {
  Metric metric = Metric::kL2;
  Matrix<float> vectors;
  std::vector<std::uint8_t> deleted;  // one byte per id, 1 = deleted
  std::uint64_t last_lsn = 0;         // 0 = includes no WAL records
};

// Thread safety: free functions with no shared state.
[[nodiscard]] Expected<void> write_snapshot(const std::filesystem::path& path,
                                            const Snapshot& snapshot);
[[nodiscard]] Expected<Snapshot> read_snapshot(const std::filesystem::path& path);

}  // namespace strata
