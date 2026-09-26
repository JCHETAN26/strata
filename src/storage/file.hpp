#pragma once

// Internal RAII wrapper over a POSIX file descriptor, for the WAL and snapshots.
// Not part of the public API.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include "strata/error.hpp"

namespace strata::io {

enum class OpenMode {
  kRead,         // must exist
  kReadWrite,    // created if missing
  kCreateTrunc,  // created or truncated, write-only
};

// Thread safety: not thread-safe; one owner at a time.
class File {
 public:
  [[nodiscard]] static Expected<File> open(const std::filesystem::path& path, OpenMode mode);

  File(File&& other) noexcept;
  File& operator=(File&& other) noexcept;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  ~File();

  // Writes every byte at the current offset (retrying short writes and EINTR).
  [[nodiscard]] Expected<void> write_all(std::span<const std::byte> data);
  // Reads exactly data.size() bytes at the current offset; kCorruptData if EOF comes first.
  [[nodiscard]] Expected<void> read_exact(std::span<std::byte> data);
  // Reads up to data.size() bytes; returns the count (0 at EOF).
  [[nodiscard]] Expected<std::size_t> read_some(std::span<std::byte> data);

  [[nodiscard]] Expected<std::uint64_t> size() const;
  [[nodiscard]] Expected<void> seek(std::uint64_t offset);
  [[nodiscard]] Expected<void> seek_end();
  [[nodiscard]] Expected<void> truncate(std::uint64_t size);

  // Durably flush file contents. On macOS this is fcntl(F_FULLFSYNC): plain fsync there only
  // reaches the drive's volatile cache.
  [[nodiscard]] Expected<void> sync();

 private:
  File(int fd, std::filesystem::path path) : fd_(fd), path_(std::move(path)) {}
  [[nodiscard]] tl::unexpected<Error> errno_error(const char* what) const;

  int fd_ = -1;
  std::filesystem::path path_;
};

// fsync a directory so a rename or file creation inside it survives power loss.
[[nodiscard]] Expected<void> sync_directory(const std::filesystem::path& dir);

// Write `data` to `path` atomically: temp file in the same directory, sync, rename, sync dir.
// Readers see either the old file or the complete new one, never a partial write.
[[nodiscard]] Expected<void> write_file_atomic(const std::filesystem::path& path,
                                               std::span<const std::byte> data);

}  // namespace strata::io
