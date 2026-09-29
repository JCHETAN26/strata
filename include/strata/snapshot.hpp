#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/matrix.hpp"

namespace strata {

// Which index a snapshot belongs to. Stored in the file so a collection cannot be reopened as a
// different kind of index by mistake.
enum class IndexKind : std::uint32_t {
  kFlat = 0,  // brute force: vectors and tombstones are the whole index
  kHnsw = 1,  // plus the HNSW graph in the index section (format in include/strata/hnsw.hpp)
};

// Point-in-time image of an index: every vector (by id), tombstones, the LSN of the last WAL
// record it includes, and an index-specific section (the HNSW graph) that the snapshot layer
// stores and checksums without interpreting.
//
// On-disk format, version 2. All integers and floats are little-endian (IEEE 754 binary32 for
// floats); hosts must be little-endian, which is checked at compile time, and the byte-order
// field below is checked on load, so a file from a big-endian writer is rejected, not misread.
//
//   offset  field
//        0  char[8] magic "STRSNP\0\1"
//        8  u32 version (2)
//       12  u32 byte-order mark 0x01020304 (reads as 0x04030201 if the byte order differs)
//       16  u32 metric
//       20  u32 dim
//       24  u32 index kind (IndexKind)
//       28  u32 zero
//       32  u64 count
//       40  u64 last_lsn
//       48  u64 num_deleted
//       56  u64 index section size in bytes
//       64  count x dim f32 vectors, row-major
//           ceil(count / 8) bytes tombstone bitmap (bit i = id i deleted)
//           index section
//           u32 crc32c of everything before it
//
// Version 1 (48-byte header: magic, version, metric, dim, zero, count, last_lsn, num_deleted; no
// byte-order mark or index section) is still read, as an IndexKind::kFlat snapshot. Other
// versions are rejected.
//
// Written atomically (temp file + sync + rename + directory sync).
struct Snapshot {
  Metric metric = Metric::kL2;
  IndexKind index = IndexKind::kFlat;
  Matrix<float> vectors;
  std::vector<std::uint8_t> deleted;  // one byte per id, 1 = deleted
  std::uint64_t last_lsn = 0;         // 0 = includes no WAL records
  std::vector<std::byte> index_data;  // index section; empty for kFlat
};

// Thread safety: free functions with no shared state.
[[nodiscard]] Expected<void> write_snapshot(const std::filesystem::path& path,
                                            const Snapshot& snapshot);
[[nodiscard]] Expected<Snapshot> read_snapshot(const std::filesystem::path& path);

}  // namespace strata
