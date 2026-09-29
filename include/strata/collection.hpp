#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <variant>
#include <vector>

#include "strata/brute_force.hpp"
#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/hnsw.hpp"
#include "strata/snapshot.hpp"
#include "strata/wal.hpp"

namespace strata {

struct CollectionOptions {
  SyncMode sync = SyncMode::kFsync;
  // Which index backs the collection. Fixed at creation: reopening with another kind fails.
  IndexKind index = IndexKind::kFlat;
  // Used when index == kHnsw. Fixed at creation: reopening with a different M, ef_construction,
  // selection, or seed fails, because the graph so far was built with the original values and the
  // WAL must replay into the same graph.
  HnswParams hnsw = {};
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
// Backed by a BruteForceIndex (IndexKind::kFlat) or an HnswIndex (IndexKind::kHnsw), chosen at
// creation. A new collection writes an initial, empty snapshot before anything else, so the index
// kind and HNSW parameters are on disk from the start and are checked on every reopen, even before
// the first checkpoint. (Directories from before this rule, with a WAL but no snapshot, can only
// be flat collections and are opened as such.)
//
// HNSW recovery: the snapshot holds the whole graph, including the level generator's state, and
// WAL replay re-applies later inserts and deletes in order, so the recovered graph is exactly the
// one the crashed process had built.
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
  // ef_search is the HNSW beam width (at least k is used); a flat collection ignores it.
  [[nodiscard]] Expected<std::vector<Neighbor>> search(std::span<const float> query, std::size_t k,
                                                       std::size_t ef_search = 64) const;
  Expected<void> checkpoint();

  [[nodiscard]] std::size_t size() const;       // ids assigned, including deleted
  [[nodiscard]] std::size_t live_size() const;  // excluding deleted
  // Copy of a stored vector, or nullopt if id is out of range or deleted.
  [[nodiscard]] std::optional<std::vector<float>> get(VectorId id) const;
  [[nodiscard]] const RecoveryInfo& recovery() const noexcept { return recovery_; }
  [[nodiscard]] std::uint64_t wal_size_bytes() const;
  [[nodiscard]] IndexKind index_kind() const noexcept;
  // The HNSW index, or nullptr for a flat collection. For tests and introspection: the caller must
  // not use it while another thread writes to the collection.
  [[nodiscard]] const HnswIndex* hnsw() const noexcept;

 private:
  using Index = std::variant<BruteForceIndex, HnswIndex>;

  Collection(std::filesystem::path dir, Index index, WriteAheadLog wal, RecoveryInfo recovery);

  std::filesystem::path dir_;
  std::unique_ptr<std::shared_mutex> mutex_;  // heap-allocated so Collection stays movable
  Index index_;
  WriteAheadLog wal_;
  RecoveryInfo recovery_;
};

}  // namespace strata
