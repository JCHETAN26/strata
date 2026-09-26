// Search benchmark harness: builds an index over a dataset once, then for each search-parameter
// point runs every query `--runs` times and prints one JSON object with build time, QPS, latency
// percentiles, and recall@k per run.
//
// Queries run one at a time on a single thread and are timed individually, so latency
// percentiles are meaningful. The Python driver (bench/run_search_bench.py) adds commit,
// hardware, and dataset metadata and saves the result under results/.
//
//   strata_search --data data/siftsmall --metric l2 --index brute_force
//   strata_search --data data/siftsmall --metric l2 --index hnsw --M 16 --ef-construction 200 \
//                 --ef-search 10,20,40,80,160

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "strata/brute_force.hpp"
#include "strata/dataset.hpp"
#include "strata/recall.hpp"
#ifdef STRATA_HAS_HNSW
#include "strata/hnsw.hpp"
#endif

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
  std::string kernel = "best";  // "best" (SIMD for this build) or "scalar"
  // HNSW
  std::size_t m = 16;
  std::size_t ef_construction = 200;
  std::vector<std::size_t> ef_search{10, 20, 40, 80, 160, 320};
};

[[noreturn]] void usage(std::string_view error) {
  std::cerr << "error: " << error << "\n\n"
            << "usage: strata_search --data DIR [--metric l2|ip|cosine|angular]\n"
            << "                     [--index brute_force|hnsw] [--k 10] [--runs 5]\n"
            << "                     [--max-queries N] [--warmup 1] [--kernel best|scalar]\n"
            << "       hnsw only:    [--M 16] [--ef-construction 200]\n"
            << "                     [--ef-search 10,20,40,80,160,320]\n";
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

std::vector<std::size_t> parse_list(std::string_view flag, const std::string& value) {
  std::vector<std::size_t> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(parse_size(flag, item));
  }
  if (out.empty()) {
    usage(std::string(flag) + " expects a comma-separated list");
  }
  return out;
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
    } else if (flag == "--kernel") {
      if (value != "best" && value != "scalar") {
        usage("--kernel must be best or scalar");
      }
      opt.kernel = value;
    } else if (flag == "--M") {
      opt.m = parse_size(flag, value);
    } else if (flag == "--ef-construction") {
      opt.ef_construction = parse_size(flag, value);
    } else if (flag == "--ef-search") {
      opt.ef_search = parse_list(flag, value);
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
  double search_seconds;
  double qps;
  double recall;        // tie-aware (ann-benchmarks definition); the headline number
  double recall_by_id;  // strict id matching against groundtruth
  double latency_mean_us;
  double latency_p50_us;
  double latency_p95_us;
  double latency_p99_us;
  double latency_max_us;
};

struct Point {
  std::string search_params_json;  // e.g. {"ef_search": 40}
  std::vector<RunResult> runs;
};

// Uniform view over every index type: search(query, k) for one point of the parameter sweep.
using SearchFn = std::function<strata::Expected<std::vector<strata::Neighbor>>(
    std::span<const float>, std::size_t)>;

struct SweepPoint {
  std::string search_params_json;
  SearchFn search;
};

// Times one sweep point: `opt.runs` runs, each after `opt.warmup` untimed passes.
strata::Expected<Point> run_point(const SweepPoint& point, const strata::Matrix<float>& queries,
                                  const strata::Matrix<std::int32_t>& groundtruth,
                                  std::span<const float> kth_distances, const Options& opt) {
  const std::size_t num_queries = queries.rows();
  Point result{.search_params_json = point.search_params_json, .runs = {}};
  for (std::size_t run = 0; run < opt.runs; ++run) {
    for (std::size_t pass = 0; pass < opt.warmup; ++pass) {
      for (std::size_t q = 0; q < num_queries; ++q) {
        (void)point.search(queries.row(q), opt.k);
      }
    }

    std::vector<std::vector<strata::Neighbor>> results(num_queries);
    std::vector<double> latencies_us(num_queries);
    const auto search_start = Clock::now();
    for (std::size_t q = 0; q < num_queries; ++q) {
      const auto t0 = Clock::now();
      auto r = point.search(queries.row(q), opt.k);
      latencies_us[q] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
      if (!r) {
        return tl::unexpected(r.error());
      }
      results[q] = std::move(*r);
    }
    const double search_seconds = seconds_since(search_start);

    auto recall = strata::recall_at_k_with_ties(results, kth_distances, opt.k);
    if (!recall) {
      return tl::unexpected(recall.error());
    }
    auto recall_by_id = strata::recall_at_k(results, groundtruth, opt.k);
    if (!recall_by_id) {
      return tl::unexpected(recall_by_id.error());
    }
    std::ranges::sort(latencies_us);
    double total_us = 0;
    for (double l : latencies_us) {
      total_us += l;
    }
    result.runs.push_back({.search_seconds = search_seconds,
                           .qps = static_cast<double>(num_queries) / search_seconds,
                           .recall = *recall,
                           .recall_by_id = *recall_by_id,
                           .latency_mean_us = total_us / static_cast<double>(num_queries),
                           .latency_p50_us = percentile(latencies_us, 50),
                           .latency_p95_us = percentile(latencies_us, 95),
                           .latency_p99_us = percentile(latencies_us, 99),
                           .latency_max_us = latencies_us.back()});
    std::cerr << "  " << point.search_params_json << " run " << run + 1 << "/" << opt.runs
              << ": qps=" << result.runs.back().qps << " recall@" << opt.k << "="
              << result.runs.back().recall << "\n";
  }
  return result;
}

struct RunInfo {
  const Options& opt;
  std::string_view metric;
  std::string_view kernel;
  std::string_view build_params_json;
  double build_seconds;
  std::size_t num_base;
  std::size_t num_queries;
  std::size_t dim;
};

void write_json(std::ostream& os, const RunInfo& info, const std::vector<Point>& points) {
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
      << "  \"index\": " << json_string(info.opt.index) << ",\n"
      << "  \"build_params\": " << info.build_params_json << ",\n"
      << "  \"build_seconds\": " << info.build_seconds << ",\n"
      << "  \"metric\": " << json_string(info.metric) << ",\n"
      << "  \"kernel\": " << json_string(info.kernel) << ",\n"
      << "  \"k\": " << info.opt.k << ",\n"
      << "  \"threads\": 1,\n"
      << "  \"warmup_passes\": " << info.opt.warmup << ",\n"
      << "  \"num_base\": " << info.num_base << ",\n"
      << "  \"num_queries\": " << info.num_queries << ",\n"
      << "  \"dim\": " << info.dim << ",\n"
      << "  \"points\": [\n";
  for (std::size_t p = 0; p < points.size(); ++p) {
    out << "    {\"search_params\": " << points[p].search_params_json << ", \"runs\": [\n";
    for (std::size_t i = 0; i < points[p].runs.size(); ++i) {
      const auto& r = points[p].runs[i];
      out << "      {\"search_seconds\": " << r.search_seconds << ", \"qps\": " << r.qps
          << ", \"recall\": " << r.recall << ", \"recall_by_id\": " << r.recall_by_id
          << ", \"latency_mean_us\": " << r.latency_mean_us
          << ", \"latency_p50_us\": " << r.latency_p50_us
          << ", \"latency_p95_us\": " << r.latency_p95_us
          << ", \"latency_p99_us\": " << r.latency_p99_us
          << ", \"latency_max_us\": " << r.latency_max_us << "}"
          << (i + 1 < points[p].runs.size() ? "," : "") << "\n";
    }
    out << "    ]}" << (p + 1 < points.size() ? "," : "") << "\n";
  }
  out << "  ]\n}\n";
  os << out.str();
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);
#if defined(__APPLE__)
  // Ask for the highest QoS so macOS keeps the thread on performance cores. Without this, Apple
  // Silicon may schedule a long-running CLI process on efficiency cores mid-run.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  const auto metric = strata::parse_metric(opt.metric);
  if (!metric) {
    usage("unknown metric " + opt.metric);
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
  const std::size_t dim = dataset->base.cols();
  strata::Matrix<float> queries(num_queries, dim);
  strata::Matrix<std::int32_t> groundtruth(num_queries, dataset->groundtruth.cols());
  for (std::size_t q = 0; q < num_queries; ++q) {
    std::ranges::copy(dataset->query.row(q), queries.row(q).begin());
    std::ranges::copy(dataset->groundtruth.row(q), groundtruth.row(q).begin());
  }
  auto kth_distances =
      strata::kth_neighbor_distances(dataset->base, queries, groundtruth, *metric, opt.k);
  if (!kth_distances) {
    std::cerr << "error: " << kth_distances.error().message << "\n";
    return 1;
  }

  const auto kernels =
      opt.kernel == "scalar" ? strata::KernelSet::kScalar : strata::KernelSet::kBest;
  const std::string_view kernel_name =
      opt.kernel == "scalar" ? std::string_view("scalar") : strata::best_kernel_name();

  // Build the index once.
  std::string build_params_json = "{}";
  std::vector<SweepPoint> sweep;
  std::optional<strata::BruteForceIndex> brute_force;
#ifdef STRATA_HAS_HNSW
  std::optional<strata::HnswIndex> hnsw;
#endif
  const auto build_start = Clock::now();
  if (opt.index == "brute_force") {
    auto index = strata::BruteForceIndex::create(dim, *metric, kernels);
    if (!index || !index->add_batch(dataset->base)) {
      std::cerr << "error: failed to build index\n";
      return 1;
    }
    brute_force.emplace(std::move(*index));
    sweep.push_back(
        {"{}", [&](std::span<const float> q, std::size_t k) { return brute_force->search(q, k); }});
#ifdef STRATA_HAS_HNSW
  } else if (opt.index == "hnsw") {
    auto index = strata::HnswIndex::create(dim, *metric,
                                           {.M = opt.m, .ef_construction = opt.ef_construction});
    if (!index) {
      std::cerr << "error: " << index.error().message << "\n";
      return 1;
    }
    if (auto added = index->add_batch(dataset->base); !added) {
      std::cerr << "error: " << added.error().message << "\n";
      return 1;
    }
    hnsw.emplace(std::move(*index));
    build_params_json = "{\"M\": " + std::to_string(opt.m) +
                        ", \"ef_construction\": " + std::to_string(opt.ef_construction) + "}";
    for (std::size_t ef : opt.ef_search) {
      sweep.push_back(
          {"{\"ef_search\": " + std::to_string(ef) + "}",
           [&, ef](std::span<const float> q, std::size_t k) { return hnsw->search(q, k, ef); }});
    }
#endif
  } else {
    usage("unknown or unavailable index " + opt.index);
  }
  const double build_seconds = seconds_since(build_start);
  std::cerr << opt.index << ": built in " << build_seconds << " s\n";

  std::vector<Point> points;
  for (const auto& point : sweep) {
    auto result = run_point(point, queries, groundtruth, *kth_distances, opt);
    if (!result) {
      std::cerr << "error: " << result.error().message << "\n";
      return 1;
    }
    points.push_back(std::move(*result));
  }

  const RunInfo info{.opt = opt,
                     .metric = strata::to_string(*metric),
                     .kernel = kernel_name,
                     .build_params_json = build_params_json,
                     .build_seconds = build_seconds,
                     .num_base = dataset->base.rows(),
                     .num_queries = num_queries,
                     .dim = dim};
  write_json(std::cout, info, points);
  return 0;
}
