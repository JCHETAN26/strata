// Storage benchmark: insert throughput and latency per sync mode, checkpoint (snapshot) time, and
// recovery time from a WAL-only directory vs. a snapshot. Prints one JSON object.
//
//   strata_storage_bench --data data/siftsmall --dir /tmp/strata_storage_bench
//                        --fsync-inserts 500 --runs 3

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "strata/collection.hpp"
#include "strata/dataset.hpp"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Options {
  std::string data;
  std::string dir = (fs::temp_directory_path() / "strata_storage_bench").string();
  std::size_t fsync_inserts = 500;  // fsync'd inserts are slow; don't do the whole dataset
  std::size_t runs = 3;
};

[[noreturn]] void usage(const std::string& error) {
  std::cerr << "error: " << error << "\nusage: strata_storage_bench --data DIR [--dir TMPDIR]"
            << " [--fsync-inserts 500] [--runs 3]\n";
  std::exit(2);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const std::string value = argv[i + 1];
    if (flag == "--data") {
      opt.data = value;
    } else if (flag == "--dir") {
      opt.dir = value;
    } else if (flag == "--fsync-inserts") {
      opt.fsync_inserts = std::stoul(value);
    } else if (flag == "--runs") {
      opt.runs = std::stoul(value);
    } else {
      usage("unknown flag " + flag);
    }
  }
  if (opt.data.empty() || argc % 2 == 0) {
    usage("--data is required and every flag takes a value");
  }
  return opt;
}

double percentile(std::vector<double> v, double p) {
  std::ranges::sort(v);
  const auto rank = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(v.size())));
  return v[std::clamp<std::size_t>(rank, 1, v.size()) - 1];
}

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename T>
T check(strata::Expected<T> result) {
  if (!result) {
    std::cerr << "error: " << result.error().message << "\n";
    std::exit(1);
  }
  if constexpr (!std::is_void_v<T>) {
    return std::move(*result);
  }
}

struct InsertStats {
  double ops_per_second;
  double p50_us;
  double p99_us;
};

InsertStats insert_run(const fs::path& dir, const strata::Matrix<float>& vectors, std::size_t n,
                       strata::SyncMode sync) {
  fs::remove_all(dir);
  auto collection =
      check(strata::Collection::open(dir, vectors.cols(), strata::Metric::kL2, {.sync = sync}));
  std::vector<double> latencies(n);
  const auto start = Clock::now();
  for (std::size_t i = 0; i < n; ++i) {
    const auto t0 = Clock::now();
    check(collection.insert(vectors.row(i)));
    latencies[i] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
  }
  const double total = seconds(start);
  return {static_cast<double>(n) / total, percentile(latencies, 50), percentile(latencies, 99)};
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);
  auto dataset = check(strata::load_dataset(opt.data));
  const auto& base = dataset.base;
  const fs::path dir = opt.dir;
  const std::size_t n = base.rows();
  const std::size_t n_fsync = std::min(opt.fsync_inserts, n);

  std::ostringstream runs;
  runs.precision(9);
  for (std::size_t run = 0; run < opt.runs; ++run) {
    const auto no_sync = insert_run(dir, base, n, strata::SyncMode::kNone);
    const auto fsync = insert_run(dir, base, n_fsync, strata::SyncMode::kFsync);

    // WAL-only directory with all n vectors, then recover from it.
    insert_run(dir, base, n, strata::SyncMode::kNone);
    const std::uint64_t wal_bytes = fs::file_size(dir / "wal.log");
    auto from_wal = check(strata::Collection::open(dir, base.cols(), strata::Metric::kL2));
    const double wal_recovery = from_wal.recovery().seconds;

    const auto t_checkpoint = Clock::now();
    check(from_wal.checkpoint());
    const double checkpoint_seconds = seconds(t_checkpoint);
    const std::uint64_t snapshot_bytes = fs::file_size(dir / "snapshot.bin");

    auto from_snapshot = check(strata::Collection::open(dir, base.cols(), strata::Metric::kL2));
    const double snapshot_recovery = from_snapshot.recovery().seconds;

    std::cerr << "run " << run + 1 << "/" << opt.runs << ": " << no_sync.ops_per_second
              << " inserts/s (no sync), " << fsync.ops_per_second << " inserts/s (fsync), "
              << "recovery " << wal_recovery << " s (WAL) / " << snapshot_recovery
              << " s (snapshot)\n";
    runs << (run == 0 ? "" : ",\n") << "    {\"insert_nosync_ops\": " << no_sync.ops_per_second
         << ", \"insert_nosync_p50_us\": " << no_sync.p50_us
         << ", \"insert_nosync_p99_us\": " << no_sync.p99_us
         << ", \"insert_fsync_ops\": " << fsync.ops_per_second
         << ", \"insert_fsync_p50_us\": " << fsync.p50_us
         << ", \"insert_fsync_p99_us\": " << fsync.p99_us
         << ", \"checkpoint_seconds\": " << checkpoint_seconds
         << ", \"recovery_wal_seconds\": " << wal_recovery
         << ", \"recovery_snapshot_seconds\": " << snapshot_recovery
         << ", \"wal_bytes\": " << wal_bytes << ", \"snapshot_bytes\": " << snapshot_bytes << "}";
  }
  fs::remove_all(dir);

  std::cout << "{\n  \"harness\": \"strata_storage_bench\",\n"
#ifdef NDEBUG
            << "  \"asserts\": false,\n"
#else
            << "  \"asserts\": true,\n"
#endif
            << "  \"num_vectors\": " << n << ",\n  \"fsync_inserts\": " << n_fsync
            << ",\n  \"dim\": " << base.cols() << ",\n  \"dir\": \"" << dir.string()
            << "\",\n  \"runs\": [\n"
            << runs.str() << "\n  ]\n}\n";
  return 0;
}
