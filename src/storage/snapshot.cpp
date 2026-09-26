#include "strata/snapshot.hpp"

#include <array>
#include <cstring>
#include <string>

#include "file.hpp"
#include "strata/crc32c.hpp"

namespace strata {

namespace {

constexpr std::array<char, 8> kMagic{'S', 'T', 'R', 'S', 'N', 'P', '\0', '\1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderSize = 48;

template <typename T>
void put(std::vector<std::byte>& buf, const T& value) {
  const auto* p = reinterpret_cast<const std::byte*>(&value);
  buf.insert(buf.end(), p, p + sizeof(T));
}

template <typename T>
T get(std::span<const std::byte> buf, std::size_t offset) {
  T value;
  std::memcpy(&value, buf.data() + offset, sizeof(T));
  return value;
}

tl::unexpected<Error> corrupt(const std::filesystem::path& path, const std::string& what) {
  return make_error(ErrorCode::kCorruptData, path.string() + ": " + what);
}

}  // namespace

Expected<void> write_snapshot(const std::filesystem::path& path, const Snapshot& snapshot) {
  const std::size_t count = snapshot.vectors.rows();
  if (snapshot.deleted.size() != count) {
    return make_error(ErrorCode::kInvalidArgument, "tombstone count differs from vector count");
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
  buf.reserve(kHeaderSize + snapshot.vectors.data().size_bytes() + bitmap.size() + 4);
  for (char c : kMagic) {
    put(buf, c);
  }
  put(buf, kVersion);
  put(buf, static_cast<std::uint32_t>(snapshot.metric));
  put(buf, static_cast<std::uint32_t>(snapshot.vectors.cols()));
  put(buf, std::uint32_t{0});
  put(buf, static_cast<std::uint64_t>(count));
  put(buf, snapshot.last_lsn);
  put(buf, num_deleted);
  const auto vector_bytes = std::as_bytes(snapshot.vectors.data());
  buf.insert(buf.end(), vector_bytes.begin(), vector_bytes.end());
  const auto bitmap_bytes = std::as_bytes(std::span(bitmap));
  buf.insert(buf.end(), bitmap_bytes.begin(), bitmap_bytes.end());
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
  if (bytes.size() < kHeaderSize + 4) {
    return corrupt(path, "file too short");
  }
  if (std::memcmp(bytes.data(), kMagic.data(), kMagic.size()) != 0) {
    return corrupt(path, "bad magic; not a Strata snapshot");
  }
  if (crc32c(bytes.first(bytes.size() - 4)) != get<std::uint32_t>(bytes, bytes.size() - 4)) {
    return corrupt(path, "checksum mismatch");
  }
  if (get<std::uint32_t>(bytes, 8) != kVersion) {
    return corrupt(path, "unsupported version");
  }
  const auto metric = get<std::uint32_t>(bytes, 12);
  const std::size_t dim = get<std::uint32_t>(bytes, 16);
  const auto count = get<std::uint64_t>(bytes, 24);
  const auto last_lsn = get<std::uint64_t>(bytes, 32);
  const auto num_deleted = get<std::uint64_t>(bytes, 40);
  if (metric > static_cast<std::uint32_t>(Metric::kCosine) || dim == 0) {
    return corrupt(path, "bad header");
  }
  const std::size_t vector_bytes = count * dim * sizeof(float);
  const std::size_t bitmap_bytes = (count + 7) / 8;
  if (bytes.size() != kHeaderSize + vector_bytes + bitmap_bytes + 4) {
    return corrupt(path, "size does not match header");
  }

  Snapshot snapshot;
  snapshot.metric = static_cast<Metric>(metric);
  snapshot.last_lsn = last_lsn;
  snapshot.vectors = Matrix<float>(count, dim);
  std::memcpy(snapshot.vectors.data().data(), bytes.data() + kHeaderSize, vector_bytes);
  snapshot.deleted.resize(count);
  std::uint64_t deleted = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto byte = get<std::uint8_t>(bytes, kHeaderSize + vector_bytes + i / 8);
    snapshot.deleted[i] = (byte >> (i % 8)) & 1U;
    deleted += snapshot.deleted[i];
  }
  if (deleted != num_deleted) {
    return corrupt(path, "tombstone count mismatch");
  }
  return snapshot;
}

}  // namespace strata
