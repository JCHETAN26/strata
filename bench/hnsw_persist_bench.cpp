// HNSW persistence: how long a build takes versus saving and loading the finished graph.
//
//   strata_hnsw_persist_bench --data data/sift1m-200k-q1000 --out /tmp/hnsw.snap \
//       [--M 16] [--ef-construction 200] [--ef-search 64]
//
// One run per process (bench/run_hnsw_persist_bench.py runs several, with cool-downs between):
// build the index from the base vectors, save it (a durable, atomic snapshot write), load it
// back, and check that the loaded index answers every query identically (ids and float bits).
// Prints one JSON object. Save time includes the fsyncs that make the snapshot durable.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "strata/dataset.hpp"
#include "strata/hnsw.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

[[noreturn]] void fail(const std::string& message) {
  std::cerr << "error: " << message << "\n";
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  std::string data;
  std::filesystem::path out;
  strata::HnswParams params;
  std::size_t ef_search = 64;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const std::string value = argv[i + 1];
    if (flag == "--data") {
      data = value;
    } else if (flag == "--out") {
      out = value;
    } else if (flag == "--M") {
      params.M = std::stoul(value);
    } else if (flag == "--ef-construction") {
      params.ef_construction = std::stoul(value);
    } else if (flag == "--ef-search") {
      ef_search = std::stoul(value);
    } else {
      fail("unknown flag " + flag);
    }
  }
  if (data.empty() || out.empty()) {
    fail(
        "usage: strata_hnsw_persist_bench --data DIR --out FILE [--M 16] "
        "[--ef-construction 200] [--ef-search 64]");
  }
  auto dataset = strata::load_dataset(data);
  if (!dataset) {
    fail(dataset.error().message);
  }
  auto index = strata::HnswIndex::create(dataset->base.cols(), strata::Metric::kL2, params);
  if (!index) {
    fail(index.error().message);
  }

  auto start = Clock::now();
  if (auto r = index->add_batch(dataset->base); !r) {
    fail(r.error().message);
  }
  const double build_seconds = seconds_since(start);

  start = Clock::now();
  if (auto r = index->save(out); !r) {
    fail(r.error().message);
  }
  const double save_seconds = seconds_since(start);
  const auto file_bytes = std::filesystem::file_size(out);

  start = Clock::now();
  auto loaded = strata::HnswIndex::load(out);
  if (!loaded) {
    fail(loaded.error().message);
  }
  const double load_seconds = seconds_since(start);

  bool identical = true;
  for (std::size_t q = 0; q < dataset->query.rows(); ++q) {
    auto a = index->search(dataset->query.row(q), 10, ef_search);
    auto b = loaded->search(dataset->query.row(q), 10, ef_search);
    identical = identical && a && b && *a == *b;
  }
  std::filesystem::remove(out);

#ifdef NDEBUG
  const bool asserts = false;
#else
  const bool asserts = true;
#endif
  std::cout << "{\"num_base\": " << dataset->base.rows() << ", \"dim\": " << dataset->base.cols()
            << ", \"M\": " << params.M << ", \"ef_construction\": " << params.ef_construction
            << ", \"asserts\": " << (asserts ? "true" : "false")
            << ", \"build_seconds\": " << build_seconds << ", \"save_seconds\": " << save_seconds
            << ", \"load_seconds\": " << load_seconds << ", \"file_bytes\": " << file_bytes
            << ", \"results_identical\": " << (identical ? "true" : "false") << "}\n";
  return identical ? 0 : 1;
}
