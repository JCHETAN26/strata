#include "strata/wal.hpp"

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "file.hpp"
#include "strata/crc32c.hpp"
#include "util/bytes.hpp"

namespace strata {

namespace {

constexpr std::array<char, 8> kMagic{'S', 'T', 'R', 'W', 'A', 'L', '\0', '\1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kFileHeaderSize = 32;
constexpr std::size_t kRecordHeaderSize = 20;

// Little-endian byte buffer builder/reader (the host is little-endian; see dataset.cpp).
template <typename T>
void put(std::vector<std::byte>& buf, const T& value) {
  const auto* p = reinterpret_cast<const std::byte*>(&value);
  buf.insert(buf.end(), p, p + sizeof(T));
}

template <typename T>
T get(std::span<const std::byte> buf, std::size_t offset) {
  T value;
  util::copy_bytes(&value, buf.data() + offset, sizeof(T));
  return value;
}

std::vector<std::byte> encode_file_header(std::size_t dim, std::uint64_t base_lsn) {
  std::vector<std::byte> buf;
  buf.reserve(kFileHeaderSize);
  for (char c : kMagic) {
    put(buf, c);
  }
  put(buf, kVersion);
  put(buf, static_cast<std::uint32_t>(dim));
  put(buf, base_lsn);
  put(buf, crc32c(buf));
  put(buf, std::uint32_t{0});
  return buf;
}

std::size_t payload_size(WalRecordType type, std::size_t dim) {
  return type == WalRecordType::kInsert ? sizeof(VectorId) + dim * sizeof(float) : sizeof(VectorId);
}

std::vector<std::byte> encode_record(WalRecordType type, std::uint64_t lsn, VectorId id,
                                     std::span<const float> vector) {
  std::vector<std::byte> buf;
  const std::size_t payload = sizeof(VectorId) + vector.size_bytes();
  buf.reserve(kRecordHeaderSize + payload);
  put(buf, std::uint32_t{0});  // crc placeholder
  put(buf, static_cast<std::uint32_t>(payload));
  put(buf, static_cast<std::uint8_t>(type));
  for (int i = 0; i < 3; ++i) {
    put(buf, std::uint8_t{0});
  }
  put(buf, lsn);
  put(buf, id);
  const auto bytes = std::as_bytes(vector);
  buf.insert(buf.end(), bytes.begin(), bytes.end());
  const std::uint32_t crc = crc32c(std::span(buf).subspan(sizeof(std::uint32_t)));
  util::copy_bytes(buf.data(), &crc, sizeof(crc));
  return buf;
}

tl::unexpected<Error> corrupt(const std::filesystem::path& path, const std::string& what) {
  return make_error(ErrorCode::kCorruptData, path.string() + ": " + what);
}

}  // namespace

WriteAheadLog::WriteAheadLog(std::filesystem::path path, std::size_t dim, SyncMode sync,
                             std::unique_ptr<io::File> file, std::uint64_t end_offset,
                             std::uint64_t next_lsn)
    : path_(std::move(path)),
      dim_(dim),
      sync_(sync),
      file_(std::move(file)),
      end_offset_(end_offset),
      next_lsn_(next_lsn) {}

WriteAheadLog::WriteAheadLog(WriteAheadLog&&) noexcept = default;
WriteAheadLog& WriteAheadLog::operator=(WriteAheadLog&&) noexcept = default;
WriteAheadLog::~WriteAheadLog() = default;

Expected<std::pair<WriteAheadLog, WalReplayStats>> WriteAheadLog::open(
    const std::filesystem::path& path, std::size_t dim, SyncMode sync, std::uint64_t first_lsn,
    const ApplyFn& apply) {
  if (dim == 0) {
    return make_error(ErrorCode::kInvalidArgument, "dimension must be positive");
  }
  if (!std::filesystem::exists(path)) {
    const auto header = encode_file_header(dim, first_lsn);
    if (auto r = io::write_file_atomic(path, header); !r) {
      return tl::unexpected(r.error());
    }
  }

  auto file = io::File::open(path, io::OpenMode::kReadWrite);
  if (!file) {
    return tl::unexpected(file.error());
  }
  auto file_size = file->size();
  if (!file_size) {
    return tl::unexpected(file_size.error());
  }

  // Read the whole log. Logs are bounded by checkpointing, so this is a few hundred MB at most.
  std::vector<std::byte> data(*file_size);
  if (auto r = file->read_exact(data); !r) {
    return tl::unexpected(r.error());
  }
  const std::span<const std::byte> bytes(data);

  if (bytes.size() < kFileHeaderSize) {
    return corrupt(path, "file shorter than header");
  }
  if (std::memcmp(bytes.data(), kMagic.data(), kMagic.size()) != 0) {
    return corrupt(path, "bad magic; not a Strata WAL");
  }
  if (crc32c(bytes.first(24)) != get<std::uint32_t>(bytes, 24)) {
    return corrupt(path, "header checksum mismatch");
  }
  if (get<std::uint32_t>(bytes, 8) != kVersion) {
    return corrupt(path, "unsupported version");
  }
  if (get<std::uint32_t>(bytes, 12) != dim) {
    return make_error(ErrorCode::kDimensionMismatch,
                      path.string() + ": log has dimension " +
                          std::to_string(get<std::uint32_t>(bytes, 12)) + ", expected " +
                          std::to_string(dim));
  }
  const std::uint64_t base_lsn = get<std::uint64_t>(bytes, 16);

  WalReplayStats stats;
  const std::size_t max_record = kRecordHeaderSize + payload_size(WalRecordType::kInsert, dim);
  std::uint64_t expected_lsn = base_lsn;
  std::size_t offset = kFileHeaderSize;
  std::vector<float> vector(dim);
  while (offset < bytes.size()) {
    const std::size_t remaining = bytes.size() - offset;
    // A record is "bad" if its header is incomplete, it runs past EOF, or it fails validation.
    // Bad records that reach EOF are torn writes; anything else is mid-log corruption.
    bool bad = remaining < kRecordHeaderSize;
    std::size_t record_end = bytes.size();
    WalRecordType type{};
    std::uint64_t lsn = 0;
    if (!bad) {
      const auto payload_len = get<std::uint32_t>(bytes, offset + 4);
      record_end = offset + kRecordHeaderSize + payload_len;
      const auto type_byte = get<std::uint8_t>(bytes, offset + 8);
      type = static_cast<WalRecordType>(type_byte);
      lsn = get<std::uint64_t>(bytes, offset + 12);
      const bool known_type = type == WalRecordType::kInsert || type == WalRecordType::kDelete;
      bad = record_end > bytes.size() || !known_type || payload_len != payload_size(type, dim) ||
            crc32c(bytes.subspan(offset + 4, record_end - offset - 4)) !=
                get<std::uint32_t>(bytes, offset);
      if (bad && record_end > bytes.size()) {
        record_end = bytes.size();
      }
    }
    if (bad) {
      // A torn write damages only the last record, so the bytes from here to EOF must fit in
      // one maximum-size record. Otherwise a corrupted length field in the middle of the log
      // would masquerade as a torn tail and silently drop the acknowledged records after it.
      if (record_end >= bytes.size() && remaining <= max_record) {
        stats.truncated_bytes = remaining;
        break;  // torn tail
      }
      return corrupt(path, "corrupt record at offset " + std::to_string(offset) +
                               " followed by more data; refusing to drop acknowledged writes");
    }
    if (lsn != expected_lsn) {
      return corrupt(path, "record at offset " + std::to_string(offset) + " has LSN " +
                               std::to_string(lsn) + ", expected " + std::to_string(expected_lsn));
    }

    const std::size_t payload = offset + kRecordHeaderSize;
    const auto id = get<VectorId>(bytes, payload);
    WalRecord record{.type = type, .lsn = lsn, .id = id, .vector = {}};
    if (type == WalRecordType::kInsert) {
      util::copy_bytes(vector.data(), bytes.data() + payload + sizeof(VectorId),
                       dim * sizeof(float));
      record.vector = vector;
    }
    if (auto r = apply(record); !r) {
      return tl::unexpected(r.error());
    }
    ++stats.records;
    ++expected_lsn;
    offset = record_end;
  }

  if (stats.truncated_bytes > 0) {
    if (auto r = file->truncate(offset); !r) {
      return tl::unexpected(r.error());
    }
    if (auto r = file->sync(); !r) {
      return tl::unexpected(r.error());
    }
  }
  if (auto r = file->seek(offset); !r) {
    return tl::unexpected(r.error());
  }
  stats.next_lsn = expected_lsn;
  auto owned = std::make_unique<io::File>(std::move(*file));
  return std::pair{WriteAheadLog(path, dim, sync, std::move(owned), offset, expected_lsn), stats};
}

Expected<std::uint64_t> WriteAheadLog::append(WalRecordType type, VectorId id,
                                              std::span<const float> vector) {
  if (failed_) {
    return make_error(ErrorCode::kIoError, path_.string() + ": log is in a failed state");
  }
  const std::uint64_t lsn = next_lsn_;
  const auto record = encode_record(type, lsn, id, vector);
  auto written = file_->write_all(record);
  if (written && sync_ == SyncMode::kFsync) {
    written = file_->sync();
  }
  if (!written) {
    // Roll back a partial record so later appends don't follow garbage.
    if (!file_->truncate(end_offset_) || !file_->seek(end_offset_)) {
      failed_ = true;
    }
    return tl::unexpected(written.error());
  }
  end_offset_ += record.size();
  ++next_lsn_;
  return lsn;
}

Expected<std::uint64_t> WriteAheadLog::append_insert(VectorId id, std::span<const float> vector) {
  if (vector.size() != dim_) {
    return make_error(ErrorCode::kDimensionMismatch, "WAL insert has wrong dimension");
  }
  return append(WalRecordType::kInsert, id, vector);
}

Expected<std::uint64_t> WriteAheadLog::append_delete(VectorId id) {
  return append(WalRecordType::kDelete, id, {});
}

Expected<void> WriteAheadLog::reset(std::uint64_t next_lsn) {
  const auto header = encode_file_header(dim_, next_lsn);
  if (auto r = io::write_file_atomic(path_, header); !r) {
    failed_ = true;  // the old file may be gone or replaced; don't append to a stale handle
    return r;
  }
  auto file = io::File::open(path_, io::OpenMode::kReadWrite);
  if (!file || !file->seek_end()) {
    failed_ = true;
    return make_error(ErrorCode::kIoError, "cannot reopen " + path_.string() + " after reset");
  }
  file_ = std::make_unique<io::File>(std::move(*file));
  end_offset_ = kFileHeaderSize;
  next_lsn_ = next_lsn;
  failed_ = false;
  return {};
}

}  // namespace strata
