// Search benchmark harness: builds an index over a dataset, runs every query, and prints one JSON
// object with build time, QPS, latency percentiles, and recall@k for each run.
//
// Runs are single-threaded and sequential: one query at a time, timed individually, so latency
// percentiles are meaningful. The Python driver (bench/run_search_bench.py) adds commit, hardware,
// and dataset metadata and saves the result under results/.
//
//   strata_search --data data/siftsmall --metric l2 --index brute_force --k 10 --runs 5

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "strata/brute_force.hpp"
#include "strata/dataset.hpp"
#include "strata/recall.hpp"

#ifndef STRATA_BUILD_TYPE
#define STRATA_BUILD_TYPE "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string data;
  std::string metric = "l2";
  std::string index = "brute_force";
  std::size_t k = 10;
  std::size_t runs = 5;
  std::size_t max_queries = 0;  // 0 = all
  std::size_t warmup = 1;       // untimed passes over the queries before each timed run
};

[[noreturn]] void usage(std::string_view error) {
  std::cerr << "error: " << error << "\n\n"
            << "usage: strata_search --data DIR [--metric l2|ip|cosine|angular]\n"
            << "                     [--index brute_force] [--k 10] [--runs 5]\n"
            << "                     [--max-queries N] [--warmup 1]\n";
  std::exit(2);
}

std::size_t parse_size(std::string_view flag, const std::string& value) {
  try {
    std::size_t pos = 0;
    const auto n = std::stoull(value, &pos);
    if (pos != value.size()) {
      throw std::invalid_argument(value);
    }
    return static_cast<std::size_t>(n);
  } catch (const std::exception&) {
    usage(std::string(flag) + " expects a non-negative integer, got '" + value + "'");
  }
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string_view flag = argv[i];
    if (i + 1 >= argc) {
      usage("missing value for " + std::string(flag));
    }
    const std::string value = argv[++i];
    if (flag == "--data") {
      opt.data = value;
    } else if (flag == "--metric") {
      opt.metric = value;
    } else if (flag == "--index") {
      opt.index = value;
    } else if (flag == "--k") {
      opt.k = parse_size(flag, value);
    } else if (flag == "--runs") {
      opt.runs = parse_size(flag, value);
    } else if (flag == "--max-queries") {
      opt.max_queries = parse_size(flag, value);
    } else if (flag == "--warmup") {
      opt.warmup = parse_size(flag, value);
    } else {
      usage("unknown flag " + std::string(flag));
    }
  }
  if (opt.data.empty()) {
    usage("--data is required");
  }
  if (opt.k == 0 || opt.runs == 0) {
    usage("--k and --runs must be positive");
  }
  return opt;
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// Nearest-rank percentile of sorted values.
double percentile(const std::vector<double>& sorted, double p) {
  const auto rank =
      static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(sorted.size())));
  return sorted[std::clamp<std::size_t>(rank, 1, sorted.size()) - 1];
}

std::string json_string(std::string_view s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    out += c;
  }
  return out + "\"";
}

struct RunResult {
  double build_seconds;
  double search_seconds;
  double qps;
  double recall;
  double latency_mean_us;
  double latency_p50_us;
  double latency_p95_us;
  double latency_p99_us;
  double latency_max_us;
};

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);

  const auto metric = strata::parse_metric(opt.metric);
  if (!metric) {
    usage("unknown metric " + opt.metric);
  }
  if (opt.index != "brute_force") {
    usage("unknown index " + opt.index);
  }

  auto dataset = strata::load_dataset(opt.data);
  if (!dataset) {
    std::cerr << "error: " << dataset.error().message << "\n";
    return 1;
  }
  const std::size_t num_queries = opt.max_queries == 0
                                      ? dataset->query.rows()
                                      : std::min(opt.max_queries, dataset->query.rows());
  if (opt.k > dataset->groundtruth.cols()) {
    usage("--k exceeds groundtruth width " + std::to_string(dataset->groundtruth.cols()));
  }
  strata::Matrix<std::int32_t> groundtruth(num_queries, dataset->groundtruth.cols());
  for (std::size_t q = 0; q < num_queries; ++q) {
    std::ranges::copy(dataset->groundtruth.row(q), groundtruth.row(q).begin());
  }

  std::vector<RunResult> runs;
  for (std::size_t run = 0; run < opt.runs; ++run) {
    const auto build_start = Clock::now();
    auto index = strata::BruteForceIndex::create(dataset->base.cols(), *metric);
    if (!index || !index->add_batch(dataset->base)) {
      std::cerr << "error: failed to build index\n";
      return 1;
    }
    const double build_seconds = seconds_since(build_start);

    for (std::size_t pass = 0; pass < opt.warmup; ++pass) {
      for (std::size_t q = 0; q < num_queries; ++q) {
        (void)index->search(dataset->query.row(q), opt.k);
      }
    }

    std::vector<std::vector<strata::Neighbor>> results(num_queries);
    std::vector<double> latencies_us(num_queries);
    const auto search_start = Clock::now();
    for (std::size_t q = 0; q < num_queries; ++q) {
      const auto t0 = Clock::now();
      auto r = index->search(dataset->query.row(q), opt.k);
      latencies_us[q] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
      if (!r) {
        std::cerr << "error: " << r.error().message << "\n";
        return 1;
      }
      results[q] = std::move(*r);
    }
    const double search_seconds = seconds_since(search_start);

    auto recall = strata::recall_at_k(results, groundtruth, opt.k);
    if (!recall) {
      std::cerr << "error: " << recall.error().message << "\n";
      return 1;
    }
    std::ranges::sort(latencies_us);
    double total_us = 0;
    for (double l : latencies_us) {
      total_us += l;
    }
    runs.push_back(
        {build_seconds, search_seconds, static_cast<double>(num_queries) / search_seconds, *recall,
         total_us / static_cast<double>(num_queries), percentile(latencies_us, 50),
         percentile(latencies_us, 95), percentile(latencies_us, 99), latencies_us.back()});
    std::cerr << "run " << run + 1 << "/" << opt.runs << ": qps=" << runs.back().qps << " recall@"
              << opt.k << "=" << runs.back().recall << "\n";
  }

  std::ostringstream out;
  out.precision(9);
  out << "{\n"
      << "  \"harness\": \"strata_search\",\n"
      << "  \"build_type\": " << json_string(STRATA_BUILD_TYPE) << ",\n"
#ifdef NDEBUG
      << "  \"asserts\": false,\n"
#else
      << "  \"asserts\": true,\n"
#endif
      << "  \"compiler\": " << json_string(__VERSION__) << ",\n"
      << "  \"index\": " << json_string(opt.index) << ",\n"
      << "  \"params\": {},\n"
      << "  \"metric\": " << json_string(strata::to_string(*metric)) << ",\n"
      << "  \"k\": " << opt.k << ",\n"
      << "  \"threads\": 1,\n"
      << "  \"warmup_passes\": " << opt.warmup << ",\n"
      << "  \"num_base\": " << dataset->base.rows() << ",\n"
      << "  \"num_queries\": " << num_queries << ",\n"
      << "  \"dim\": " << dataset->base.cols() << ",\n"
      << "  \"runs\": [\n";
  for (std::size_t i = 0; i < runs.size(); ++i) {
    const auto& r = runs[i];
    out << "    {\"build_seconds\": " << r.build_seconds
        << ", \"search_seconds\": " << r.search_seconds << ", \"qps\": " << r.qps
        << ", \"recall\": " << r.recall << ", \"latency_mean_us\": " << r.latency_mean_us
        << ", \"latency_p50_us\": " << r.latency_p50_us
        << ", \"latency_p95_us\": " << r.latency_p95_us
        << ", \"latency_p99_us\": " << r.latency_p99_us
        << ", \"latency_max_us\": " << r.latency_max_us << "}" << (i + 1 < runs.size() ? "," : "")
        << "\n";
  }
  out << "  ]\n}\n";
  std::cout << out.str();
  return 0;
}
