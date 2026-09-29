#include "strata/snapshot.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <string>

#include "file.hpp"
#include "strata/crc32c.hpp"
#include "util/bytes.hpp"

namespace strata {

static_assert(std::endian::native == std::endian::little,
              "the snapshot format is little-endian and written with native byte order");
static_assert(std::numeric_limits<float>::is_iec559, "the snapshot format stores IEEE 754 floats");

namespace {

constexpr std::array<char, 8> kMagic{'S', 'T', 'R', 'S', 'N', 'P', '\0', '\1'};
// Version written. 3 and 2 share the header layout; they differ only in the HNSW index section,
// which the index interprets (its section carries its own version).
constexpr std::uint32_t kVersion = 3;
constexpr std::uint32_t kVersionTextRng = 2;
constexpr std::uint32_t kByteOrderMark = 0x01020304;
constexpr std::uint32_t kByteOrderMarkSwapped = 0x04030201;  // as read on the other byte order
constexpr std::size_t kHeaderSize = 64;
constexpr std::size_t kHeaderSizeV1 = 48;

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

tl::unexpected<Error> corrupt(const std::filesystem::path& path, const std::string& what) {
  return make_error(ErrorCode::kCorruptData, path.string() + ": " + what);
}

// Fields common to both versions, as read from the header.
struct Header {
  std::uint32_t metric = 0;
  std::size_t dim = 0;
  std::uint32_t index = 0;
  std::uint64_t count = 0;
  std::uint64_t last_lsn = 0;
  std::uint64_t num_deleted = 0;
  std::uint64_t index_bytes = 0;
  std::size_t header_size = 0;
};

Expected<Header> parse_header(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  Header h;
  const auto version = get<std::uint32_t>(bytes, 8);
  if (version == 1) {
    if (bytes.size() < kHeaderSizeV1 + 4) {
      return corrupt(path, "file too short");
    }
    h.header_size = kHeaderSizeV1;
    h.metric = get<std::uint32_t>(bytes, 12);
    h.dim = get<std::uint32_t>(bytes, 16);
    h.index = static_cast<std::uint32_t>(IndexKind::kFlat);
    h.count = get<std::uint64_t>(bytes, 24);
    h.last_lsn = get<std::uint64_t>(bytes, 32);
    h.num_deleted = get<std::uint64_t>(bytes, 40);
    return h;
  }
  if (version != kVersion && version != kVersionTextRng) {
    return corrupt(path, "unsupported version " + std::to_string(version));
  }
  if (bytes.size() < kHeaderSize + 4) {
    return corrupt(path, "file too short");
  }
  if (const auto mark = get<std::uint32_t>(bytes, 12); mark != kByteOrderMark) {
    return corrupt(path, mark == kByteOrderMarkSwapped ? "written with the opposite byte order"
                                                       : "bad byte-order mark");
  }
  h.header_size = kHeaderSize;
  h.metric = get<std::uint32_t>(bytes, 16);
  h.dim = get<std::uint32_t>(bytes, 20);
  h.index = get<std::uint32_t>(bytes, 24);
  h.count = get<std::uint64_t>(bytes, 32);
  h.last_lsn = get<std::uint64_t>(bytes, 40);
  h.num_deleted = get<std::uint64_t>(bytes, 48);
  h.index_bytes = get<std::uint64_t>(bytes, 56);
  return h;
}

}  // namespace

Expected<void> write_snapshot(const std::filesystem::path& path, const Snapshot& snapshot) {
  const std::size_t count = snapshot.vectors.rows();
  if (snapshot.deleted.size() != count) {
    return make_error(ErrorCode::kInvalidArgument, "tombstone count differs from vector count");
  }
  if (snapshot.index == IndexKind::kFlat && !snapshot.index_data.empty()) {
    return make_error(ErrorCode::kInvalidArgument, "a flat snapshot has no index section");
  }
  std::uint64_t num_deleted = 0;
  std::vector<std::uint8_t> bitmap((count + 7) / 8, 0);
  for (std::size_t i = 0; i < count; ++i) {
    if (snapshot.deleted[i] != 0) {
      bitmap[i / 8] |= static_cast<std::uint8_t>(1U << (i % 8));
      ++num_deleted;
    }
  }

  std::vector<std::byte> buf;
  buf.reserve(kHeaderSize + snapshot.vectors.data().size_bytes() + bitmap.size() +
              snapshot.index_data.size() + 4);
  for (char c : kMagic) {
    put(buf, c);
  }
  put(buf, kVersion);
  put(buf, kByteOrderMark);
  put(buf, static_cast<std::uint32_t>(snapshot.metric));
  put(buf, static_cast<std::uint32_t>(snapshot.vectors.cols()));
  put(buf, static_cast<std::uint32_t>(snapshot.index));
  put(buf, std::uint32_t{0});
  put(buf, static_cast<std::uint64_t>(count));
  put(buf, snapshot.last_lsn);
  put(buf, num_deleted);
  put(buf, static_cast<std::uint64_t>(snapshot.index_data.size()));
  const auto vector_bytes = std::as_bytes(snapshot.vectors.data());
  buf.insert(buf.end(), vector_bytes.begin(), vector_bytes.end());
  const auto bitmap_bytes = std::as_bytes(std::span(bitmap));
  buf.insert(buf.end(), bitmap_bytes.begin(), bitmap_bytes.end());
  buf.insert(buf.end(), snapshot.index_data.begin(), snapshot.index_data.end());
  put(buf, crc32c(buf));
  return io::write_file_atomic(path, buf);
}

Expected<Snapshot> read_snapshot(const std::filesystem::path& path) {
  auto file = io::File::open(path, io::OpenMode::kRead);
  if (!file) {
    return tl::unexpected(file.error());
  }
  auto size = file->size();
  if (!size) {
    return tl::unexpected(size.error());
  }
  std::vector<std::byte> data(*size);
  if (auto r = file->read_exact(data); !r) {
    return tl::unexpected(r.error());
  }
  const std::span<const std::byte> bytes(data);
  if (bytes.size() < 12 + 4) {
    return corrupt(path, "file too short");
  }
  if (std::memcmp(bytes.data(), kMagic.data(), kMagic.size()) != 0) {
    return corrupt(path, "bad magic; not a Strata snapshot");
  }
  if (crc32c(bytes.first(bytes.size() - 4)) != get<std::uint32_t>(bytes, bytes.size() - 4)) {
    return corrupt(path, "checksum mismatch");
  }
  auto header = parse_header(path, bytes);
  if (!header) {
    return tl::unexpected(header.error());
  }
  const Header& h = *header;
  if (h.metric > static_cast<std::uint32_t>(Metric::kCosine) || h.dim == 0 ||
      h.index > static_cast<std::uint32_t>(IndexKind::kHnsw)) {
    return corrupt(path, "bad header");
  }
  // Sizes come from the file: check for overflow before trusting them.
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (h.count > max / h.dim / sizeof(float) || h.index_bytes > bytes.size()) {
    return corrupt(path, "size does not match header");
  }
  const std::size_t vector_bytes = h.count * h.dim * sizeof(float);
  const std::size_t bitmap_bytes = (h.count + 7) / 8;
  if (bytes.size() != h.header_size + vector_bytes + bitmap_bytes + h.index_bytes + 4) {
    return corrupt(path, "size does not match header");
  }

  Snapshot snapshot;
  snapshot.metric = static_cast<Metric>(h.metric);
  snapshot.index = static_cast<IndexKind>(h.index);
  snapshot.last_lsn = h.last_lsn;
  snapshot.vectors = Matrix<float>(h.count, h.dim);
  // An empty snapshot (count == 0) has a null vectors.data(); copy_bytes makes that safe.
  util::copy_bytes(snapshot.vectors.data().data(), bytes.data() + h.header_size, vector_bytes);
  const std::size_t bitmap_offset = h.header_size + vector_bytes;
  snapshot.deleted.resize(h.count);
  std::uint64_t deleted = 0;
  for (std::size_t i = 0; i < h.count; ++i) {
    const auto byte = get<std::uint8_t>(bytes, bitmap_offset + i / 8);
    snapshot.deleted[i] = (byte >> (i % 8)) & 1U;
    deleted += snapshot.deleted[i];
  }
  if (deleted != h.num_deleted) {
    return corrupt(path, "tombstone count mismatch");
  }
  const auto index_section = bytes.subspan(bitmap_offset + bitmap_bytes, h.index_bytes);
  if (snapshot.index == IndexKind::kFlat && !index_section.empty()) {
    return corrupt(path, "flat snapshot with an index section");
  }
  snapshot.index_data.assign(index_section.begin(), index_section.end());
  return snapshot;
}

}  // namespace strata
