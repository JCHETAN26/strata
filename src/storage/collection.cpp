#include "strata/collection.hpp"

#include <chrono>
#include <limits>
#include <mutex>
#include <string>

#include "strata/snapshot.hpp"

namespace strata {

namespace {

constexpr const char* kSnapshotFile = "snapshot.bin";
constexpr const char* kWalFile = "wal.log";

std::string kind_name(IndexKind kind) { return kind == IndexKind::kHnsw ? "hnsw" : "flat"; }

std::string selection_name(NeighborSelection selection) {
  return selection == NeighborSelection::kSimple ? "simple" : "heuristic";
}

// Every construction parameter that differs, as "name: on disk X, requested Y"; empty if none.
std::string hnsw_param_differences(const HnswParams& disk, const HnswParams& wanted) {
  std::string out;
  const auto add = [&](const std::string& name, const std::string& a, const std::string& b) {
    if (a != b) {
      out += (out.empty() ? "" : "; ") + name + ": on disk " + a + ", requested " + b;
    }
  };
  add("M", std::to_string(disk.M), std::to_string(wanted.M));
  add("ef_construction", std::to_string(disk.ef_construction),
      std::to_string(wanted.ef_construction));
  add("selection", selection_name(disk.selection), selection_name(wanted.selection));
  add("seed", std::to_string(disk.seed), std::to_string(wanted.seed));
  return out;
}

Snapshot flat_snapshot(const BruteForceIndex& index, std::uint64_t last_lsn) {
  Snapshot snapshot;
  snapshot.metric = index.metric();
  snapshot.index = IndexKind::kFlat;
  snapshot.last_lsn = last_lsn;
  snapshot.vectors = Matrix<float>(index.size(), index.dim());
  snapshot.deleted.resize(index.size());
  for (std::size_t i = 0; i < index.size(); ++i) {
    const auto id = static_cast<VectorId>(i);
    const auto v = index.vector(id);
    std::ranges::copy(v, snapshot.vectors.row(i).begin());
    snapshot.deleted[i] = index.is_deleted(id) ? 1 : 0;
  }
  return snapshot;
}

Snapshot snapshot_of(const std::variant<BruteForceIndex, HnswIndex>& index,
                     std::uint64_t last_lsn) {
  if (const auto* hnsw = std::get_if<HnswIndex>(&index)) {
    return hnsw->to_snapshot(last_lsn);
  }
  return flat_snapshot(std::get<BruteForceIndex>(index), last_lsn);
}

// The index a snapshot describes, after checking it matches what the caller asked for.
Expected<std::variant<BruteForceIndex, HnswIndex>> index_from_snapshot(
    const std::filesystem::path& dir, const Snapshot& snapshot, std::size_t dim, Metric metric,
    const CollectionOptions& options) {
  if (snapshot.vectors.cols() != dim || snapshot.metric != metric) {
    return make_error(ErrorCode::kInvalidArgument,
                      dir.string() + " holds a collection with a different dimension or metric");
  }
  if (snapshot.index != options.index) {
    return make_error(ErrorCode::kInvalidArgument,
                      dir.string() + " holds a " + kind_name(snapshot.index) + " collection, not " +
                          kind_name(options.index));
  }
  if (snapshot.index == IndexKind::kHnsw) {
    auto hnsw = HnswIndex::from_snapshot(snapshot);
    if (!hnsw) {
      return tl::unexpected(hnsw.error());
    }
    if (auto diff = hnsw_param_differences(hnsw->params(), options.hnsw); !diff.empty()) {
      return make_error(ErrorCode::kInvalidArgument,
                        dir.string() + " was built with different HNSW parameters (" + diff + ")");
    }
    return std::variant<BruteForceIndex, HnswIndex>(std::move(*hnsw));
  }
  auto flat = BruteForceIndex::create(dim, metric);
  if (!flat) {
    return tl::unexpected(flat.error());
  }
  if (auto r = flat->add_batch(snapshot.vectors); !r) {
    return tl::unexpected(r.error());
  }
  for (std::size_t i = 0; i < snapshot.deleted.size(); ++i) {
    if (snapshot.deleted[i] != 0) {
      (void)flat->remove(static_cast<VectorId>(i));
    }
  }
  return std::variant<BruteForceIndex, HnswIndex>(std::move(*flat));
}

Expected<std::variant<BruteForceIndex, HnswIndex>> empty_index(std::size_t dim, Metric metric,
                                                               const CollectionOptions& options) {
  if (options.index == IndexKind::kHnsw) {
    auto hnsw = HnswIndex::create(dim, metric, options.hnsw);
    if (!hnsw) {
      return tl::unexpected(hnsw.error());
    }
    return std::variant<BruteForceIndex, HnswIndex>(std::move(*hnsw));
  }
  auto flat = BruteForceIndex::create(dim, metric);
  if (!flat) {
    return tl::unexpected(flat.error());
  }
  return std::variant<BruteForceIndex, HnswIndex>(std::move(*flat));
}

// Applies one replayed WAL record to the index.
Expected<void> apply_record(std::variant<BruteForceIndex, HnswIndex>& index,
                            const WalRecord& record) {
  return std::visit(
      [&](auto& idx) -> Expected<void> {
        if (record.type == WalRecordType::kInsert) {
          if (record.id != idx.size()) {
            return make_error(ErrorCode::kCorruptData,
                              "WAL insert for id " + std::to_string(record.id) + ", expected " +
                                  std::to_string(idx.size()));
          }
          if (auto r = idx.add(record.vector); !r) {
            return tl::unexpected(r.error());
          }
          return {};
        }
        if (auto r = idx.remove(record.id); !r) {
          return make_error(ErrorCode::kCorruptData, "WAL delete: " + r.error().message);
        }
        return {};
      },
      index);
}

// The index as of the last snapshot (checked against the options), or a new empty one. Fills in
// the snapshot part of `info`. A new collection gets its initial snapshot here.
Expected<std::variant<BruteForceIndex, HnswIndex>> load_or_create(const std::filesystem::path& dir,
                                                                  std::size_t dim, Metric metric,
                                                                  const CollectionOptions& options,
                                                                  RecoveryInfo& info) {
  const auto snapshot_path = dir / kSnapshotFile;
  if (std::filesystem::exists(snapshot_path)) {
    auto snapshot = read_snapshot(snapshot_path);
    if (!snapshot) {
      return tl::unexpected(snapshot.error());
    }
    info.loaded_snapshot = true;
    info.snapshot_lsn = snapshot->last_lsn;
    return index_from_snapshot(dir, *snapshot, dim, metric, options);
  }
  auto index = empty_index(dim, metric, options);
  if (!index) {
    return index;
  }
  if (std::filesystem::exists(dir / kWalFile)) {
    // Created before collections wrote an initial snapshot: only flat collections existed.
    if (options.index != IndexKind::kFlat) {
      return make_error(ErrorCode::kInvalidArgument,
                        dir.string() + " holds a flat collection, not " + kind_name(options.index));
    }
    return index;
  }
  // A new collection: record its kind and parameters before the first write, so every later open
  // can check them.
  if (auto r = write_snapshot(snapshot_path, snapshot_of(*index, 0)); !r) {
    return tl::unexpected(r.error());
  }
  return index;
}

}  // namespace

Collection::Collection(std::filesystem::path dir, Index index, WriteAheadLog wal,
                       RecoveryInfo recovery)
    : dir_(std::move(dir)),
      mutex_(std::make_unique<std::shared_mutex>()),
      index_(std::move(index)),
      wal_(std::move(wal)),
      recovery_(recovery) {}

Collection::Collection(Collection&&) noexcept = default;
Collection& Collection::operator=(Collection&&) noexcept = default;
Collection::~Collection() = default;

Expected<Collection> Collection::open(const std::filesystem::path& dir, std::size_t dim,
                                      Metric metric, CollectionOptions options) {
  const auto start = std::chrono::steady_clock::now();
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    return make_error(ErrorCode::kIoError, "cannot create " + dir.string() + ": " + ec.message());
  }

  RecoveryInfo info;
  auto index = load_or_create(dir, dim, metric, options, info);
  if (!index) {
    return tl::unexpected(index.error());
  }

  // Replay the WAL on top. Records at or below the snapshot LSN are already applied.
  auto apply = [&](const WalRecord& record) -> Expected<void> {
    if (record.lsn <= info.snapshot_lsn) {
      ++info.wal_records_skipped;
      return {};
    }
    if (auto r = apply_record(*index, record); !r) {
      return r;
    }
    ++info.wal_records_replayed;
    return {};
  };
  auto opened =
      WriteAheadLog::open(dir / kWalFile, dim, options.sync, info.snapshot_lsn + 1, apply);
  if (!opened) {
    return tl::unexpected(opened.error());
  }
  info.wal_truncated_bytes = opened->second.truncated_bytes;
  info.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return Collection(dir, std::move(*index), std::move(opened->first), info);
}

Expected<VectorId> Collection::insert(std::span<const float> vector) {
  const std::unique_lock lock(*mutex_);
  return std::visit(
      [&](auto& index) -> Expected<VectorId> {
        if (vector.size() != index.dim()) {
          return make_error(ErrorCode::kDimensionMismatch,
                            "expected dimension " + std::to_string(index.dim()) + ", got " +
                                std::to_string(vector.size()));
        }
        // Check everything that could make the in-memory apply fail *before* logging, so the WAL
        // never holds a record the index rejected.
        if (index.size() >= std::numeric_limits<VectorId>::max()) {
          return make_error(ErrorCode::kInvalidArgument, "collection is full");
        }
        const auto id = static_cast<VectorId>(index.size());
        if (auto logged = wal_.append_insert(id, vector); !logged) {
          return tl::unexpected(logged.error());
        }
        return index.add(vector);
      },
      index_);
}

Expected<void> Collection::remove(VectorId id) {
  const std::unique_lock lock(*mutex_);
  return std::visit(
      [&](auto& index) -> Expected<void> {
        if (id >= index.size() || index.is_deleted(id)) {
          return make_error(ErrorCode::kNotFound, "no live vector with id " + std::to_string(id));
        }
        if (auto logged = wal_.append_delete(id); !logged) {
          return tl::unexpected(logged.error());
        }
        return index.remove(id);
      },
      index_);
}

Expected<std::vector<Neighbor>> Collection::search(std::span<const float> query, std::size_t k,
                                                   std::size_t ef_search) const {
  const std::shared_lock lock(*mutex_);
  if (const auto* hnsw = std::get_if<HnswIndex>(&index_)) {
    return hnsw->search(query, k, ef_search);
  }
  return std::get<BruteForceIndex>(index_).search(query, k);
}

Expected<void> Collection::checkpoint() {
  const std::unique_lock lock(*mutex_);
  const Snapshot snapshot = snapshot_of(index_, wal_.next_lsn() - 1);
  if (auto r = write_snapshot(dir_ / kSnapshotFile, snapshot); !r) {
    return r;
  }
  return wal_.reset(snapshot.last_lsn + 1);
}

std::size_t Collection::size() const {
  const std::shared_lock lock(*mutex_);
  return std::visit([](const auto& index) { return index.size(); }, index_);
}

std::size_t Collection::live_size() const {
  const std::shared_lock lock(*mutex_);
  return std::visit([](const auto& index) { return index.live_size(); }, index_);
}

std::optional<std::vector<float>> Collection::get(VectorId id) const {
  const std::shared_lock lock(*mutex_);
  return std::visit(
      [&](const auto& index) -> std::optional<std::vector<float>> {
        if (id >= index.size() || index.is_deleted(id)) {
          return std::nullopt;
        }
        const auto v = index.vector(id);
        return std::vector<float>(v.begin(), v.end());
      },
      index_);
}

std::uint64_t Collection::wal_size_bytes() const {
  const std::shared_lock lock(*mutex_);
  return wal_.size_bytes();
}

IndexKind Collection::index_kind() const noexcept {
  return std::holds_alternative<HnswIndex>(index_) ? IndexKind::kHnsw : IndexKind::kFlat;
}

const HnswIndex* Collection::hnsw() const noexcept { return std::get_if<HnswIndex>(&index_); }

}  // namespace strata
