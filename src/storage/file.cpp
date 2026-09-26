#include "file.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

namespace strata::io {

namespace {

tl::unexpected<Error> path_error(const std::filesystem::path& path, const char* what) {
  return make_error(ErrorCode::kIoError,
                    std::string(what) + " " + path.string() + ": " + std::strerror(errno));
}

int full_sync(int fd) {
#if defined(__APPLE__)
  // F_FULLFSYNC asks the drive to flush its cache. Some filesystems don't support it; fall back.
  if (::fcntl(fd, F_FULLFSYNC) == 0) {
    return 0;
  }
#endif
  return ::fsync(fd);
}

}  // namespace

Expected<File> File::open(const std::filesystem::path& path, OpenMode mode) {
  int flags = O_CLOEXEC;
  switch (mode) {
    case OpenMode::kRead:
      flags |= O_RDONLY;
      break;
    case OpenMode::kReadWrite:
      flags |= O_RDWR | O_CREAT;
      break;
    case OpenMode::kCreateTrunc:
      flags |= O_WRONLY | O_CREAT | O_TRUNC;
      break;
  }
  const int fd = ::open(path.c_str(), flags, 0644);
  if (fd < 0) {
    return path_error(path, "cannot open");
  }
  return File(fd, path);
}

File::File(File&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), path_(std::move(other.path_)) {}

File& File::operator=(File&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    fd_ = std::exchange(other.fd_, -1);
    path_ = std::move(other.path_);
  }
  return *this;
}

File::~File() {
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

tl::unexpected<Error> File::errno_error(const char* what) const { return path_error(path_, what); }

Expected<void> File::write_all(std::span<const std::byte> data) {
  while (!data.empty()) {
    const ssize_t n = ::write(fd_, data.data(), data.size());
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return errno_error("write failed on");
    }
    data = data.subspan(static_cast<std::size_t>(n));
  }
  return {};
}

Expected<std::size_t> File::read_some(std::span<std::byte> data) {
  while (true) {
    const ssize_t n = ::read(fd_, data.data(), data.size());
    if (n >= 0) {
      return static_cast<std::size_t>(n);
    }
    if (errno != EINTR) {
      return errno_error("read failed on");
    }
  }
}

Expected<void> File::read_exact(std::span<std::byte> data) {
  while (!data.empty()) {
    auto n = read_some(data);
    if (!n) {
      return tl::unexpected(n.error());
    }
    if (*n == 0) {
      return make_error(ErrorCode::kCorruptData, "unexpected end of file in " + path_.string());
    }
    data = data.subspan(*n);
  }
  return {};
}

Expected<std::uint64_t> File::size() const {
  struct stat st{};
  if (::fstat(fd_, &st) != 0) {
    return errno_error("fstat failed on");
  }
  return static_cast<std::uint64_t>(st.st_size);
}

Expected<void> File::seek(std::uint64_t offset) {
  if (::lseek(fd_, static_cast<off_t>(offset), SEEK_SET) < 0) {
    return errno_error("seek failed on");
  }
  return {};
}

Expected<void> File::seek_end() {
  if (::lseek(fd_, 0, SEEK_END) < 0) {
    return errno_error("seek failed on");
  }
  return {};
}

Expected<void> File::truncate(std::uint64_t size) {
  if (::ftruncate(fd_, static_cast<off_t>(size)) != 0) {
    return errno_error("truncate failed on");
  }
  return {};
}

Expected<void> File::sync() {
  if (full_sync(fd_) != 0) {
    return errno_error("sync failed on");
  }
  return {};
}

Expected<void> sync_directory(const std::filesystem::path& dir) {
  const int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return path_error(dir, "cannot open directory");
  }
  const int rc = full_sync(fd);
  ::close(fd);
  if (rc != 0) {
    return path_error(dir, "sync failed on directory");
  }
  return {};
}

Expected<void> write_file_atomic(const std::filesystem::path& path,
                                 std::span<const std::byte> data) {
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  {
    auto file = File::open(tmp, OpenMode::kCreateTrunc);
    if (!file) {
      return tl::unexpected(file.error());
    }
    if (auto r = file->write_all(data); !r) {
      return r;
    }
    if (auto r = file->sync(); !r) {
      return r;
    }
  }
  if (::rename(tmp.c_str(), path.c_str()) != 0) {
    return path_error(path, "cannot rename temp file to");
  }
  return sync_directory(path.has_parent_path() ? path.parent_path() : std::filesystem::path("."));
}

}  // namespace strata::io
