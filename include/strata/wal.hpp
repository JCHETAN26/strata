#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <utility>

#include "strata/error.hpp"
#include "strata/types.hpp"

namespace strata {

namespace io {
class File;
}

// On-disk format (little-endian):
//
//   File header, 32 bytes:
//     char[8]  magic "STRWAL\0\1"
//     u32      version (1)
//     u32      dim
//     u64      base_lsn   LSN of the first record in this file
//     u32      crc32c of the 24 bytes above
//     u32      zero
//
//   Record, 20-byte header + payload:
//     u32      crc32c of everything after this field (length, type, lsn, payload)
//     u32      payload length in bytes
//     u8       type (1 = insert, 2 = delete), then 3 zero bytes
//     u64      lsn, exactly base_lsn + record index
//     payload  insert: u32 id, dim x f32   delete: u32 id
//
// A record is acknowledged once append_*() returns. With SyncMode::kFsync that means it is on
// stable storage; with kNone it is in the OS page cache (survives a process crash, not power loss).
//
// Recovery (open) replays records in order and stops at the first bad one:
//   - If the bad record reaches end of file and the remaining bytes fit in one record, it is a
//     torn write from a crash mid-append. It was never acknowledged, so it is truncated away and
//     recovery succeeds.
//   - Otherwise (more data follows, or a corrupted length field claims to run past EOF over
//     more than one record's worth of bytes) the log is corrupt in the middle. Truncating would
//     drop acknowledged writes, so open fails with kCorruptData instead.

enum class WalRecordType : std::uint8_t {
  kInsert = 1,
  kDelete = 2,
};

struct WalRecord {
  WalRecordType type;
  std::uint64_t lsn;
  VectorId id;
  std::span<const float> vector;  // empty for deletes; valid only during the callback
};

enum class SyncMode {
  kNone,   // write() only: durable against process crashes, not against power loss
  kFsync,  // fsync (F_FULLFSYNC on macOS) after every append
};

struct WalReplayStats {
  std::size_t records = 0;
  std::uint64_t truncated_bytes = 0;  // torn tail removed during recovery
  std::uint64_t next_lsn = 0;
};

// Thread safety: not thread-safe. Callers serialize appends (Collection holds a lock).
class WriteAheadLog {
 public:
  using ApplyFn = std::function<Expected<void>(const WalRecord&)>;

  // Opens the log at `path`, creating it (with base_lsn = `first_lsn`) if missing. Every valid
  // record is passed to `apply` in LSN order; an error from `apply` aborts the open.
  [[nodiscard]] static Expected<std::pair<WriteAheadLog, WalReplayStats>> open(
      const std::filesystem::path& path, std::size_t dim, SyncMode sync, std::uint64_t first_lsn,
      const ApplyFn& apply);

  WriteAheadLog(WriteAheadLog&&) noexcept;
  WriteAheadLog& operator=(WriteAheadLog&&) noexcept;
  ~WriteAheadLog();

  // Append and (per SyncMode) sync one record. Returns its LSN. On failure the log is rolled back
  // to its previous end; if that also fails, every later append fails.
  [[nodiscard]] Expected<std::uint64_t> append_insert(VectorId id, std::span<const float> vector);
  [[nodiscard]] Expected<std::uint64_t> append_delete(VectorId id);

  // Atomically replaces the log with an empty one whose first record will have LSN `next_lsn`.
  // Used after a checkpoint has made every existing record redundant.
  [[nodiscard]] Expected<void> reset(std::uint64_t next_lsn);

  [[nodiscard]] std::uint64_t next_lsn() const noexcept { return next_lsn_; }
  [[nodiscard]] std::uint64_t size_bytes() const noexcept { return end_offset_; }

 private:
  WriteAheadLog(std::filesystem::path path, std::size_t dim, SyncMode sync,
                std::unique_ptr<io::File> file, std::uint64_t end_offset, std::uint64_t next_lsn);
  [[nodiscard]] Expected<std::uint64_t> append(WalRecordType type, VectorId id,
                                               std::span<const float> vector);

  std::filesystem::path path_;
  std::size_t dim_;
  SyncMode sync_;
  std::unique_ptr<io::File> file_;
  std::uint64_t end_offset_;
  std::uint64_t next_lsn_;
  bool failed_ = false;
};

}  // namespace strata
