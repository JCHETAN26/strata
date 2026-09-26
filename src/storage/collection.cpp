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

}  // namespace

Collection::Collection(std::filesystem::path dir, BruteForceIndex index, WriteAheadLog wal,
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
  auto index = BruteForceIndex::create(dim, metric);
  if (!index) {
    return tl::unexpected(index.error());
  }

  RecoveryInfo info;
  const auto snapshot_path = dir / kSnapshotFile;
  if (std::filesystem::exists(snapshot_path)) {
    auto snapshot = read_snapshot(snapshot_path);
    if (!snapshot) {
      return tl::unexpected(snapshot.error());
    }
    if (snapshot->vectors.cols() != dim || snapshot->metric != metric) {
      return make_error(ErrorCode::kInvalidArgument,
                        dir.string() + " holds a collection with a different dimension or metric");
    }
    if (auto r = index->add_batch(snapshot->vectors); !r) {
      return tl::unexpected(r.error());
    }
    for (std::size_t i = 0; i < snapshot->deleted.size(); ++i) {
      if (snapshot->deleted[i] != 0) {
        (void)index->remove(static_cast<VectorId>(i));
      }
    }
    info.loaded_snapshot = true;
    info.snapshot_lsn = snapshot->last_lsn;
  }

  // Replay the WAL on top. Records at or below the snapshot LSN are already applied.
  auto apply = [&](const WalRecord& record) -> Expected<void> {
    if (record.lsn <= info.snapshot_lsn) {
      ++info.wal_records_skipped;
      return {};
    }
    if (record.type == WalRecordType::kInsert) {
      if (record.id != index->size()) {
        return make_error(ErrorCode::kCorruptData, "WAL insert for id " +
                                                       std::to_string(record.id) + ", expected " +
                                                       std::to_string(index->size()));
      }
      if (auto r = index->add(record.vector); !r) {
        return tl::unexpected(r.error());
      }
    } else {
      if (auto r = index->remove(record.id); !r) {
        return make_error(ErrorCode::kCorruptData, "WAL delete: " + r.error().message);
      }
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
  if (vector.size() != index_.dim()) {
    return make_error(ErrorCode::kDimensionMismatch, "expected dimension " +
                                                         std::to_string(index_.dim()) + ", got " +
                                                         std::to_string(vector.size()));
  }
  // Check everything that could make the in-memory apply fail *before* logging, so the WAL never
  // holds a record the index rejected.
  if (index_.size() >= std::numeric_limits<VectorId>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "collection is full");
  }
  const auto id = static_cast<VectorId>(index_.size());
  if (auto logged = wal_.append_insert(id, vector); !logged) {
    return tl::unexpected(logged.error());
  }
  return index_.add(vector);
}

Expected<void> Collection::remove(VectorId id) {
  const std::unique_lock lock(*mutex_);
  if (id >= index_.size() || index_.is_deleted(id)) {
    return make_error(ErrorCode::kNotFound, "no live vector with id " + std::to_string(id));
  }
  if (auto logged = wal_.append_delete(id); !logged) {
    return tl::unexpected(logged.error());
  }
  return index_.remove(id);
}

Expected<std::vector<Neighbor>> Collection::search(std::span<const float> query,
                                                   std::size_t k) const {
  const std::shared_lock lock(*mutex_);
  return index_.search(query, k);
}

Expected<void> Collection::checkpoint() {
  const std::unique_lock lock(*mutex_);
  Snapshot snapshot;
  snapshot.metric = index_.metric();
  snapshot.last_lsn = wal_.next_lsn() - 1;
  snapshot.vectors = Matrix<float>(index_.size(), index_.dim());
  snapshot.deleted.resize(index_.size());
  for (std::size_t i = 0; i < index_.size(); ++i) {
    const auto id = static_cast<VectorId>(i);
    const auto v = index_.vector(id);
    std::copy(v.begin(), v.end(), snapshot.vectors.row(i).begin());
    snapshot.deleted[i] = index_.is_deleted(id) ? 1 : 0;
  }
  if (auto r = write_snapshot(dir_ / kSnapshotFile, snapshot); !r) {
    return r;
  }
  return wal_.reset(snapshot.last_lsn + 1);
}

std::size_t Collection::size() const {
  const std::shared_lock lock(*mutex_);
  return index_.size();
}

std::size_t Collection::live_size() const {
  const std::shared_lock lock(*mutex_);
  return index_.live_size();
}

std::optional<std::vector<float>> Collection::get(VectorId id) const {
  const std::shared_lock lock(*mutex_);
  if (id >= index_.size() || index_.is_deleted(id)) {
    return std::nullopt;
  }
  const auto v = index_.vector(id);
  return std::vector<float>(v.begin(), v.end());
}

std::uint64_t Collection::wal_size_bytes() const {
  const std::shared_lock lock(*mutex_);
  return wal_.size_bytes();
}

}  // namespace strata
