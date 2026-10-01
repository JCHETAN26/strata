// Search benchmark harness: builds an index over a dataset once, then for each search-parameter
// point runs every query `--runs` times and prints one JSON object with build time, QPS, latency
// percentiles, and recall@k per run.
//
// Each query is timed individually. With --threads N the queries are spread over a thread pool
// (throughput mode): QPS is total queries / wall time, and latency percentiles describe queries
// running concurrently with N - 1 others. The Python driver (bench/run_search_bench.py) adds
// commit, hardware, and dataset metadata and saves the result under results/.
//
//   strata_search --data data/siftsmall --metric l2 --index brute_force
//   strata_search --data data/siftsmall --metric l2 --index hnsw --M 16 --ef-construction 200 \
//                 --ef-search 10,20,40,80,160

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
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
#include "strata/pq.hpp"
#include "strata/recall.hpp"
#include "strata/thread_pool.hpp"
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
  std::size_t threads = 1;
  // HNSW
  std::size_t m = 16;
  std::size_t ef_construction = 200;
  std::string selection = "heuristic";  // "heuristic" (paper Algorithm 4) or "simple" (closest M)
  // Threads for building the HNSW graph. 1 = the sequential, deterministic build; more = the
  // parallel add_batch. (--threads is for searching.)
  std::size_t build_threads = 1;
  // If set: load the HNSW index from this snapshot when it exists, otherwise build and save it
  // there. For runs that measure search only (thread scaling, perf profiles), so they do not pay
  // for, or profile, a build. Recorded in build_params, so such records never match build-time
  // comparisons.
  std::string snapshot;
  std::vector<std::size_t> ef_search{10, 20, 40, 80, 160, 320};
  // PQ
  std::size_t pq_m = 16;
  std::vector<std::size_t> rerank{0, 10, 20, 50, 100, 200, 500};
  // HNSW only: after building, delete these cumulative fractions of the ids in turn and sweep
  // ef_search at each, with ground truth recomputed over the live vectors. Empty = no deletes.
  std::vector<double> delete_fractions;
  // With --delete-fractions: at each fraction above 0, also build a fresh index over only the live
  // vectors and sweep it, to show what a rebuild would buy.
  bool compare_rebuild = false;
};

[[noreturn]] void usage(std::string_view error) {
  std::cerr
      << "error: " << error << "\n\n"
      << "usage: strata_search --data DIR [--metric l2|ip|cosine|angular]\n"
      << "                     [--index brute_force|hnsw] [--k 10] [--runs 5]\n"
      << "                     [--max-queries N] [--warmup 1] [--kernel best|scalar]\n"
      << "                     [--threads 1]\n"
      << "       hnsw only:    [--M 16] [--ef-construction 200] [--selection heuristic|simple]\n"
      << "                     [--build-threads 1] [--snapshot PATH]\n"
      << "                     [--ef-search 10,20,40,80,160,320]\n"
      << "       pq only:      [--pq-m 16] [--rerank 0,10,20,50,100,200,500]\n"
      << "       hnsw deletes: [--delete-fractions 0,0.25,0.5,0.9] [--compare-rebuild 1]\n";
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
    } else if (flag == "--threads") {
      opt.threads = parse_size(flag, value);
    } else if (flag == "--kernel") {
      if (value != "best" && value != "scalar") {
        usage("--kernel must be best or scalar");
      }
      opt.kernel = value;
    } else if (flag == "--M") {
      opt.m = parse_size(flag, value);
    } else if (flag == "--ef-construction") {
      opt.ef_construction = parse_size(flag, value);
    } else if (flag == "--snapshot") {
      opt.snapshot = value;
    } else if (flag == "--build-threads") {
      opt.build_threads = parse_size(flag, value);
      if (opt.build_threads == 0) {
        usage("--build-threads must be positive");
      }
    } else if (flag == "--selection") {
      if (value != "heuristic" && value != "simple") {
        usage("--selection must be heuristic or simple");
      }
      opt.selection = value;
    } else if (flag == "--ef-search") {
      opt.ef_search = parse_list(flag, value);
    } else if (flag == "--pq-m") {
      opt.pq_m = parse_size(flag, value);
    } else if (flag == "--compare-rebuild") {
      opt.compare_rebuild = value == "1" || value == "true";
    } else if (flag == "--delete-fractions") {
      opt.delete_fractions.clear();
      std::stringstream list(value);
      for (std::string item; std::getline(list, item, ',');) {
        char* end = nullptr;
        const double f = std::strtod(item.c_str(), &end);
        if (end == item.c_str() || *end != '\0' || f < 0 || f >= 1 ||
            (!opt.delete_fractions.empty() && f < opt.delete_fractions.back())) {
          usage("--delete-fractions must be ascending values in [0, 1)");
        }
        opt.delete_fractions.push_back(f);
      }
    } else if (flag == "--rerank") {
      opt.rerank = parse_list(flag, value);
    } else {
      usage("unknown flag " + std::string(flag));
    }
  }
  if (opt.data.empty()) {
    usage("--data is required");
  }
  if (opt.k == 0 || opt.runs == 0 || opt.threads == 0) {
    usage("--k, --runs, and --threads must be positive");
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
strata::Expected<Point> run_point(const SweepPoint& point, const strata::Matrix<float>& base,
                                  strata::DistanceFn exact_distance,
                                  const strata::Matrix<float>& queries,
                                  const strata::Matrix<std::int32_t>& groundtruth,
                                  std::span<const float> kth_distances, const Options& opt,
                                  strata::ThreadPool& pool) {
  const std::size_t num_queries = queries.rows();
  Point result{.search_params_json = point.search_params_json, .runs = {}};
  for (std::size_t run = 0; run < opt.runs; ++run) {
    for (std::size_t pass = 0; pass < opt.warmup; ++pass) {
      pool.parallel_for(num_queries,
                        [&](std::size_t q) { (void)point.search(queries.row(q), opt.k); });
    }

    std::vector<strata::Expected<std::vector<strata::Neighbor>>> answers(num_queries);
    std::vector<double> latencies_us(num_queries);
    const auto search_start = Clock::now();
    pool.parallel_for(num_queries, [&](std::size_t q) {
      const auto t0 = Clock::now();
      answers[q] = point.search(queries.row(q), opt.k);
      latencies_us[q] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    });
    const double search_seconds = seconds_since(search_start);

    // Recall is judged on exact distances to the returned ids, since some indexes (PQ without
    // reranking) return estimated distances. Outside the timed region.
    std::vector<std::vector<strata::Neighbor>> results(num_queries);
    for (std::size_t q = 0; q < num_queries; ++q) {
      if (!answers[q]) {
        return tl::unexpected(answers[q].error());
      }
      results[q] = std::move(*answers[q]);
      for (auto& n : results[q]) {
        n.distance = exact_distance(queries.row(q), base.row(n.id));
      }
    }

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

#ifdef STRATA_HAS_HNSW
// Graph statistics that explain recall: a node that no search can reach on layer 0 can never be
// returned, whatever ef_search is. Uses only the index's public introspection accessors.
std::string hnsw_graph_json(const strata::HnswIndex& index) {
  if (!index.entry_point()) {
    return "null";
  }
  const std::size_t n = index.size();
  std::vector<std::size_t> nodes_per_layer(static_cast<std::size_t>(index.max_level()) + 1, 0);
  std::size_t layer0_edges = 0;
  for (strata::VectorId id = 0; id < n; ++id) {
    for (int layer = 0; layer <= index.level(id); ++layer) {
      ++nodes_per_layer[static_cast<std::size_t>(layer)];
    }
    layer0_edges += index.neighbors(id, 0).size();
  }
  std::vector<char> seen(n, 0);
  std::vector<strata::VectorId> stack{*index.entry_point()};
  seen[stack.front()] = 1;
  std::size_t reached = 1;
  while (!stack.empty()) {
    const strata::VectorId id = stack.back();
    stack.pop_back();
    for (strata::VectorId nbr : index.neighbors(id, 0)) {
      if (seen[nbr] == 0) {
        seen[nbr] = 1;
        ++reached;
        stack.push_back(nbr);
      }
    }
  }
  std::ostringstream out;
  out.precision(9);
  out << "{\"max_level\": " << index.max_level() << ", \"nodes_per_layer\": [";
  for (std::size_t l = 0; l < nodes_per_layer.size(); ++l) {
    out << (l > 0 ? ", " : "") << nodes_per_layer[l];
  }
  out << "], \"layer0_mean_degree\": " << static_cast<double>(layer0_edges) / static_cast<double>(n)
      << ", \"layer0_reachable_fraction\": "
      << static_cast<double>(reached) / static_cast<double>(n) << "}";
  return out.str();
}
#endif

#ifdef STRATA_HAS_HNSW
// Which ids a delete sweep removes: those whose hash falls below the fraction, so the sets are
// nested (everything deleted at 25% is also deleted at 50%) and spread evenly over the ids.
std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}
bool deleted_at(std::size_t id, double fraction) {
  return static_cast<double>(splitmix64(id) % 1'000'000) < fraction * 1'000'000;
}

// Exact top-k over the live vectors, as ground truth ids and k-th distances for recall.
struct LiveTruth {
  strata::Matrix<std::int32_t> ids;
  std::vector<float> kth;
};
strata::Expected<LiveTruth> live_truth(const strata::BruteForceIndex& exact,
                                       const strata::Matrix<float>& queries, std::size_t k) {
  LiveTruth truth{strata::Matrix<std::int32_t>(queries.rows(), k), {}};
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto found = exact.search(queries.row(q), k);
    if (!found) {
      return tl::unexpected(found.error());
    }
    if (found->size() < k) {
      return strata::make_error(strata::ErrorCode::kInvalidArgument, "fewer live vectors than k");
    }
    for (std::size_t i = 0; i < k; ++i) {
      truth.ids.row(q)[i] = static_cast<std::int32_t>((*found)[i].id);
    }
    truth.kth.push_back(found->back().distance);
  }
  return truth;
}
#endif

struct RunInfo {
  const Options& opt;
  std::string_view metric;
  std::string_view kernel;
  std::string_view build_params_json;
  std::string_view graph_json;  // HNSW graph statistics, or null
  double build_seconds;
  std::size_t num_base;
  std::size_t index_bytes;
  std::size_t rerank_bytes;
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
      << "  \"graph\": " << info.graph_json << ",\n"
      << "  \"metric\": " << json_string(info.metric) << ",\n"
      << "  \"kernel\": " << json_string(info.kernel) << ",\n"
      << "  \"k\": " << info.opt.k << ",\n"
      << "  \"threads\": " << info.opt.threads << ",\n"
      << "  \"warmup_passes\": " << info.opt.warmup << ",\n"
      << "  \"num_base\": " << info.num_base << ",\n"
      << "  \"index_bytes\": " << info.index_bytes << ",\n"
      << "  \"rerank_bytes\": " << info.rerank_bytes << ",\n"
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

  // Used for the timed searches and (for PQ) for parallel training/encoding.
  strata::ThreadPool pool(opt.threads);

  // Build the index once.
  std::string build_params_json = "{}";
  std::string graph_json = "null";
  // Bytes the index needs to answer queries (excluding originals kept only for reranking).
  std::size_t index_bytes = dataset->base.data().size_bytes();
  std::size_t rerank_bytes = 0;
  std::vector<SweepPoint> sweep;
  std::optional<strata::BruteForceIndex> brute_force;
  std::optional<strata::PqIndex> pq_index;
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
    const auto selection = opt.selection == "simple" ? strata::NeighborSelection::kSimple
                                                     : strata::NeighborSelection::kHeuristic;
    const bool load = !opt.snapshot.empty() && std::filesystem::exists(opt.snapshot);
    if (load) {
      auto loaded = strata::HnswIndex::load(opt.snapshot);
      if (!loaded) {
        std::cerr << "error: " << loaded.error().message << "\n";
        return 1;
      }
      if (loaded->size() != dataset->base.rows() || loaded->dim() != dim ||
          loaded->params().M != opt.m || loaded->params().ef_construction != opt.ef_construction) {
        std::cerr << "error: snapshot " << opt.snapshot << " does not match the dataset or M / "
                  << "ef_construction; delete it to rebuild\n";
        return 1;
      }
      hnsw.emplace(std::move(*loaded));
    } else {
      auto index = strata::HnswIndex::create(
          dim, *metric,
          {.M = opt.m, .ef_construction = opt.ef_construction, .selection = selection});
      if (!index) {
        std::cerr << "error: " << index.error().message << "\n";
        return 1;
      }
      std::optional<strata::ThreadPool> build_pool;
      if (opt.build_threads > 1) {
        build_pool.emplace(opt.build_threads);
      }
      auto added = build_pool ? index->add_batch(dataset->base, *build_pool)
                              : index->add_batch(dataset->base);
      if (!added) {
        std::cerr << "error: " << added.error().message << "\n";
        return 1;
      }
      if (!opt.snapshot.empty()) {
        if (auto saved = index->save(opt.snapshot); !saved) {
          std::cerr << "error: " << saved.error().message << "\n";
          return 1;
        }
      }
      hnsw.emplace(std::move(*index));
    }
    graph_json = hnsw_graph_json(*hnsw);
    // build_threads is recorded only for parallel builds, so sequential records keep matching
    // earlier ones.
    build_params_json =
        "{\"M\": " + std::to_string(opt.m) +
        ", \"ef_construction\": " + std::to_string(opt.ef_construction) + ", \"selection\": \"" +
        opt.selection + "\"" +
        (opt.build_threads > 1 ? ", \"build_threads\": " + std::to_string(opt.build_threads)
                               : std::string()) +
        (opt.snapshot.empty()
             ? std::string()
             : std::string(", \"snapshot\": ") + (load ? "\"loaded\"" : "\"built\"")) +
        "}";
    for (std::size_t ef : opt.ef_search) {
      sweep.push_back(
          {"{\"ef_search\": " + std::to_string(ef) + "}",
           [&, ef](std::span<const float> q, std::size_t k) { return hnsw->search(q, k, ef); }});
    }
#endif
  } else if (opt.index == "pq") {
    auto pq = strata::ProductQuantizer::train(dataset->base, *metric, {.m = opt.pq_m}, &pool);
    if (!pq) {
      std::cerr << "error: " << pq.error().message << "\n";
      return 1;
    }
    auto index = strata::PqIndex::create(std::move(*pq), /*keep_originals=*/true);
    if (!index || !index->add_batch(dataset->base, &pool)) {
      std::cerr << "error: failed to build PQ index\n";
      return 1;
    }
    pq_index.emplace(std::move(*index));
    index_bytes = pq_index->code_bytes() + pq_index->codebook_bytes();
    rerank_bytes = pq_index->original_bytes();
    build_params_json = "{\"m\": " + std::to_string(opt.pq_m) + "}";
    for (std::size_t rerank : opt.rerank) {
      sweep.push_back({"{\"rerank\": " + std::to_string(rerank) + "}",
                       [&, rerank](std::span<const float> q, std::size_t k) {
                         return pq_index->search(q, k, {.rerank = rerank});
                       }});
    }
  } else {
    usage("unknown or unavailable index " + opt.index);
  }
  const double build_seconds = seconds_since(build_start);
  std::cerr << opt.index << ": built in " << build_seconds << " s\n";

  std::vector<Point> points;
  if (opt.delete_fractions.empty()) {
    for (const auto& point : sweep) {
      auto result = run_point(point, dataset->base, strata::distance_function(*metric, kernels),
                              queries, groundtruth, *kth_distances, opt, pool);
      if (!result) {
        std::cerr << "error: " << result.error().message << "\n";
        return 1;
      }
      points.push_back(std::move(*result));
    }
  } else {
#ifdef STRATA_HAS_HNSW
    if (!hnsw) {
      usage("--delete-fractions needs --index hnsw");
    }
    // The same tombstones in an exact index give the live ground truth at each fraction.
    auto exact = strata::BruteForceIndex::create(dim, *metric, kernels);
    if (!exact || !exact->add_batch(dataset->base)) {
      std::cerr << "error: failed to build the exact index\n";
      return 1;
    }
    for (double fraction : opt.delete_fractions) {
      for (std::size_t id = 0; id < hnsw->size(); ++id) {
        const auto vid = static_cast<strata::VectorId>(id);
        if (deleted_at(id, fraction) && !hnsw->is_deleted(vid)) {
          (void)hnsw->remove(vid);
          (void)exact->remove(vid);
        }
      }
      auto truth = live_truth(*exact, queries, opt.k);
      if (!truth) {
        std::cerr << "error: " << truth.error().message << "\n";
        return 1;
      }
      const double live =
          static_cast<double>(hnsw->live_size()) / static_cast<double>(hnsw->size());
      std::cerr << "deleted fraction " << fraction << ": " << hnsw->live_size() << " live\n";
      for (const auto& point : sweep) {
        std::string params = point.search_params_json;
        params.pop_back();  // the closing brace
        params += ", \"deleted_fraction\": " + std::to_string(fraction) +
                  ", \"live_fraction\": " + std::to_string(live) + "}";
        const SweepPoint labeled{params, point.search};
        auto result = run_point(labeled, dataset->base, strata::distance_function(*metric, kernels),
                                queries, truth->ids, truth->kth, opt, pool);
        if (!result) {
          std::cerr << "error: " << result.error().message << "\n";
          return 1;
        }
        points.push_back(std::move(*result));
      }
      if (!opt.compare_rebuild || fraction == 0) {
        continue;
      }
      // The alternative to living with tombstones: a new index over the live vectors (in id
      // order, same parameters). Its ids are positions in `live_ids`; they are mapped back so
      // recall is measured against the same ground truth.
      std::vector<strata::VectorId> live_ids;
      for (std::size_t id = 0; id < hnsw->size(); ++id) {
        if (!hnsw->is_deleted(static_cast<strata::VectorId>(id))) {
          live_ids.push_back(static_cast<strata::VectorId>(id));
        }
      }
      auto rebuilt = strata::HnswIndex::create(dim, *metric, hnsw->params());
      const auto rebuild_start = Clock::now();
      for (strata::VectorId id : live_ids) {
        if (!rebuilt || !rebuilt->add(dataset->base.row(id))) {
          std::cerr << "error: rebuild failed\n";
          return 1;
        }
      }
      const double rebuild_seconds = seconds_since(rebuild_start);
      std::cerr << "  rebuilt " << live_ids.size() << " live vectors in " << rebuild_seconds
                << " s\n";
      for (std::size_t ef : opt.ef_search) {
        const std::string params =
            "{\"ef_search\": " + std::to_string(ef) +
            ", \"deleted_fraction\": " + std::to_string(fraction) +
            ", \"live_fraction\": " + std::to_string(live) +
            ", \"rebuilt\": true, \"rebuild_seconds\": " + std::to_string(rebuild_seconds) + "}";
        const SweepPoint point{
            params,
            [&, ef](std::span<const float> q,
                    std::size_t k) -> strata::Expected<std::vector<strata::Neighbor>> {
              auto found = rebuilt->search(q, k, ef);
              if (found) {
                for (auto& n : *found) {
                  n.id = live_ids[n.id];
                }
              }
              return found;
            }};
        auto result = run_point(point, dataset->base, strata::distance_function(*metric, kernels),
                                queries, truth->ids, truth->kth, opt, pool);
        if (!result) {
          std::cerr << "error: " << result.error().message << "\n";
          return 1;
        }
        points.push_back(std::move(*result));
      }
    }
#else
    usage("--delete-fractions needs HNSW");
#endif
  }

  const RunInfo info{.opt = opt,
                     .metric = strata::to_string(*metric),
                     .kernel = kernel_name,
                     .build_params_json = build_params_json,
                     .graph_json = graph_json,
                     .build_seconds = build_seconds,
                     .num_base = dataset->base.rows(),
                     .index_bytes = index_bytes,
                     .rerank_bytes = rerank_bytes,
                     .num_queries = num_queries,
                     .dim = dim};
  write_json(std::cout, info, points);
  return 0;
}
