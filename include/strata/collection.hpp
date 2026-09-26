#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <vector>

#include "strata/brute_force.hpp"
#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/wal.hpp"

namespace strata {

struct CollectionOptions {
  SyncMode sync = SyncMode::kFsync;
};

struct RecoveryInfo {
  bool loaded_snapshot = false;
  std::uint64_t snapshot_lsn = 0;
  std::size_t wal_records_replayed = 0;
  std::size_t wal_records_skipped = 0;  // already covered by the snapshot
  std::uint64_t wal_truncated_bytes = 0;
  double seconds = 0.0;
};

// A durable, searchable set of vectors in a directory:
//
//   <dir>/snapshot.bin   last checkpoint (optional)
//   <dir>/wal.log        every write since that checkpoint
//
// Write path: validate -> append to WAL (and sync, per options) -> apply in memory -> return.
// A write is acknowledged when insert()/remove() returns success; acknowledged writes survive a
// crash at any point (with SyncMode::kFsync, also power loss).
//
// checkpoint() writes a snapshot of the current state (atomically), then resets the WAL. A crash
// between the two leaves WAL records the snapshot already covers; recovery skips them by LSN.
//
// Currently backed by BruteForceIndex; HNSW plugs in once its implementation exists.
//
// Thread safety: all methods are safe to call concurrently. Writes and checkpoints take an
// exclusive lock; searches share a lock and run in parallel with each other.
class Collection {
 public:
  // Opens (recovering from snapshot + WAL) or creates the collection in `dir`. Fails with
  // kInvalidArgument if an existing collection has a different dimension or metric.
  [[nodiscard]] static Expected<Collection> open(const std::filesystem::path& dir, std::size_t dim,
                                                 Metric metric, CollectionOptions options = {});

  Collection(Collection&&) noexcept;
  Collection& operator=(Collection&&) noexcept;
  ~Collection();

  Expected<VectorId> insert(std::span<const float> vector);
  Expected<void> remove(VectorId id);
  [[nodiscard]] Expected<std::vector<Neighbor>> search(std::span<const float> query,
                                                       std::size_t k) const;
  Expected<void> checkpoint();

  [[nodiscard]] std::size_t size() const;       // ids assigned, including deleted
  [[nodiscard]] std::size_t live_size() const;  // excluding deleted
  // Copy of a stored vector, or nullopt if id is out of range or deleted.
  [[nodiscard]] std::optional<std::vector<float>> get(VectorId id) const;
  [[nodiscard]] const RecoveryInfo& recovery() const noexcept { return recovery_; }
  [[nodiscard]] std::uint64_t wal_size_bytes() const;

 private:
  Collection(std::filesystem::path dir, BruteForceIndex index, WriteAheadLog wal,
             RecoveryInfo recovery);

  std::filesystem::path dir_;
  std::unique_ptr<std::shared_mutex> mutex_;  // heap-allocated so Collection stays movable
  BruteForceIndex index_;
  WriteAheadLog wal_;
  RecoveryInfo recovery_;
};

}  // namespace strata
