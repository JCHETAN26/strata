// Filtered HNSW search: recall and QPS per (filter kind, selectivity, strategy, ef_search).
//
//   strata_hnsw_filter_bench --data data/sift1m-200k-q1000 --snapshot /tmp/idx.snap \
//       --clusters-cache /tmp/clusters.bin --kind random|correlated --phase graph|auto \
//       [--selectivities 0.001,0.01,0.1,0.5] [--ef-search 10,20,40,80,160,320] [--runs 3]
//       [--max-queries 500] [--threshold 0.02] [--build-threads 4]
//
// The index is loaded from --snapshot if it exists, otherwise built (with --build-threads) and
// saved there, so every run of this tool measures the same graph. Two attributes per vector:
//   bucket  = hash(id) % 100000                       random filters: bucket < s * 100000
//   cluster = nearest of 1000 k-means centroids       correlated filters: a set of clusters,
//             (cached in --clusters-cache)            taken in a fixed shuffled order until they
//                                                      cover a fraction s of the vectors
// Filters are CompiledFilters evaluated inside the timed region (each query brings its own
// filter), so the pre-filter pays its per-query evaluation of every id.
//
// Phases: graph = the pre-filter (exact) and the graph strategy at each ef_search (never falls
// back); auto = the auto strategy with --threshold at each ef_search, recording how often it chose
// the pre-filter and how often the graph fell back. Recall@k is tie-aware against exact filtered
// search; for correlated filters it is also split by whether the query's own cluster matches.
// Prints one JSON object.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "strata/brute_force.hpp"
#include "strata/dataset.hpp"
#include "strata/filter.hpp"
#include "strata/hnsw.hpp"
#include "strata/kmeans.hpp"
#include "strata/thread_pool.hpp"

namespace {

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

constexpr std::size_t kK = 10;
constexpr std::size_t kClusters = 1000;
constexpr std::int64_t kBuckets = 100000;

[[noreturn]] void fail(const std::string& message) {
  std::cerr << "error: " << message << "\n";
  std::exit(1);
}

template <typename T>
T check(strata::Expected<T> r) {
  if (!r) {
    fail(r.error().message);
  }
  if constexpr (!std::is_void_v<T>) {
    return std::move(*r);
  }
}

std::vector<double> parse_doubles(const std::string& s) {
  std::vector<double> out;
  std::stringstream in(s);
  for (std::string item; std::getline(in, item, ',');) {
    out.push_back(std::stod(item));
  }
  return out;
}

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

struct Options {
  std::string data;
  fs::path snapshot;
  fs::path clusters_cache;
  std::string kind = "random";
  std::string phase = "graph";
  std::vector<double> selectivities{0.001, 0.01, 0.1, 0.5};
  std::vector<double> ef_search{10, 20, 40, 80, 160, 320};
  std::size_t runs = 3;
  std::size_t max_queries = 500;
  double threshold = strata::kDefaultPrefilterBelow;
  std::size_t build_threads = 4;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const std::string value = argv[i + 1];
    if (flag == "--data") {
      o.data = value;
    } else if (flag == "--snapshot") {
      o.snapshot = value;
    } else if (flag == "--clusters-cache") {
      o.clusters_cache = value;
    } else if (flag == "--kind") {
      o.kind = value;
    } else if (flag == "--phase") {
      o.phase = value;
    } else if (flag == "--selectivities") {
      o.selectivities = parse_doubles(value);
    } else if (flag == "--ef-search") {
      o.ef_search = parse_doubles(value);
    } else if (flag == "--runs") {
      o.runs = std::stoul(value);
    } else if (flag == "--max-queries") {
      o.max_queries = std::stoul(value);
    } else if (flag == "--threshold") {
      o.threshold = std::stod(value);
    } else if (flag == "--build-threads") {
      o.build_threads = std::stoul(value);
    } else {
      fail("unknown flag " + flag);
    }
  }
  if (o.data.empty() || o.snapshot.empty() || o.clusters_cache.empty() ||
      (o.kind != "random" && o.kind != "correlated") || (o.phase != "graph" && o.phase != "auto")) {
    fail("usage: see the comment at the top of bench/hnsw_filter_bench.cpp");
  }
  return o;
}

// Each base vector's and each query's nearest k-means centroid, computed once and cached (the
// k-means is deterministic for a seed and pool size, but slow enough to be worth caching).
struct Clusters {
  std::vector<std::uint32_t> base;
  std::vector<std::uint32_t> queries;
};

Clusters load_or_compute_clusters(const Options& opt, const strata::Dataset& ds,
                                  std::size_t num_queries, strata::ThreadPool& pool) {
  Clusters c;
  if (std::ifstream in(opt.clusters_cache, std::ios::binary); in) {
    c.base.resize(ds.base.rows());
    c.queries.resize(num_queries);
    in.read(reinterpret_cast<char*>(c.base.data()),
            static_cast<std::streamsize>(c.base.size() * 4));
    in.read(reinterpret_cast<char*>(c.queries.data()),
            static_cast<std::streamsize>(c.queries.size() * 4));
    if (in) {
      return c;
    }
  }
  // Train on a 50k sample (deterministic stride), then assign every vector to its nearest
  // centroid with an exact search over the 1000 centroids.
  const std::size_t sample_rows = std::min<std::size_t>(50000, ds.base.rows());
  strata::Matrix<float> sample(sample_rows, ds.base.cols());
  const std::size_t stride = ds.base.rows() / sample_rows;
  for (std::size_t i = 0; i < sample_rows; ++i) {
    std::ranges::copy(ds.base.row(i * stride), sample.row(i).begin());
  }
  auto km = check(strata::kmeans(sample, {.k = kClusters, .max_iterations = 10, .seed = 7}, &pool));
  auto centroids = check(strata::BruteForceIndex::create(ds.base.cols(), strata::Metric::kL2));
  check(centroids.add_batch(km.centroids));
  const auto nearest = [&](const strata::Matrix<float>& m, std::size_t rows) {
    std::vector<std::uint32_t> out(rows);
    const strata::MatrixView<const float> view(m.data().data(), rows, m.cols());
    auto found = check(centroids.search_batch(view, 1, pool));
    for (std::size_t i = 0; i < rows; ++i) {
      out[i] = found[i][0].id;
    }
    return out;
  };
  c.base = nearest(ds.base, ds.base.rows());
  c.queries = nearest(ds.query, num_queries);
  std::ofstream out(opt.clusters_cache, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(c.base.data()),
            static_cast<std::streamsize>(c.base.size() * 4));
  out.write(reinterpret_cast<const char*>(c.queries.data()),
            static_cast<std::streamsize>(c.queries.size() * 4));
  return c;
}

// Clusters in a fixed shuffled order; the correlated filter at selectivity s takes a prefix of
// this order until it covers s of the vectors (so filters at larger s contain those at smaller s).
std::vector<std::int64_t> clusters_for(double selectivity, const std::vector<std::uint32_t>& base) {
  std::vector<std::size_t> sizes(kClusters, 0);
  for (std::uint32_t c : base) {
    ++sizes[c];
  }
  std::vector<std::int64_t> order(kClusters);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937_64(11));
  std::vector<std::int64_t> chosen;
  std::size_t covered = 0;
  const auto target = static_cast<std::size_t>(selectivity * static_cast<double>(base.size()));
  for (std::int64_t c : order) {
    if (covered >= std::max<std::size_t>(target, 1)) {
      break;
    }
    chosen.push_back(c);
    covered += sizes[static_cast<std::size_t>(c)];
  }
  return chosen;
}

struct QueryTruth {
  std::size_t expected;  // min(k, matching live vectors)
  float kth;             // distance of the k-th exact result (ties count as hits)
};

// Everything one (filter, selectivity) measurement needs.
struct FilterCase {
  const strata::HnswIndex* index;
  const strata::Dataset* ds;
  std::size_t num_queries;
  bool correlated;
  const strata::CompiledFilter* filter;
  std::vector<bool> own_match;    // correlated: the query's own cluster is in the filter
  std::vector<QueryTruth> truth;  // exact filtered answers
};

strata::FilterStrategy strategy_of(const std::string& name) {
  if (name == "prefilter") {
    return strata::FilterStrategy::kPreFilter;
  }
  if (name == "graph") {
    return strata::FilterStrategy::kGraph;
  }
  return strata::FilterStrategy::kAuto;
}

// Tie-aware recall@k of one query's results.
double query_recall(const FilterCase& c, std::size_t q, const std::vector<strata::Neighbor>& got) {
  const QueryTruth& t = c.truth[q];
  if (t.expected == 0) {
    return 1.0;
  }
  std::size_t hits = 0;
  for (const auto& nb : got) {
    hits += (nb.distance <= t.kth && c.filter->matches(nb.id)) ? 1 : 0;
  }
  return static_cast<double>(std::min(hits, t.expected)) / static_cast<double>(t.expected);
}

// One timed pass over the queries, as a JSON object.
std::string timed_pass(const FilterCase& c, const strata::FilteredSearchOptions& options) {
  std::vector<std::vector<strata::Neighbor>> results(c.num_queries);
  std::vector<strata::FilteredSearchStats> stats(c.num_queries);
  const auto start = Clock::now();
  for (std::size_t q = 0; q < c.num_queries; ++q) {
    results[q] =
        check(c.index->search_filtered(c.ds->query.row(q), kK, *c.filter, options, &stats[q]));
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();

  double recall = 0;
  double recall_match = 0;
  std::size_t n_match = 0;
  double prefilter = 0;
  double fallback = 0;
  double distances = 0;
  std::size_t graph_searches = 0;
  for (std::size_t q = 0; q < c.num_queries; ++q) {
    const double r = query_recall(c, q, results[q]);
    recall += r;
    if (c.own_match[q]) {
      recall_match += r;
      ++n_match;
    }
    prefilter += stats[q].used == strata::FilterStrategy::kPreFilter ? 1 : 0;
    fallback += stats[q].fell_back ? 1 : 0;
    if (stats[q].graph_distances > 0) {
      ++graph_searches;
      distances += static_cast<double>(stats[q].graph_distances);
    }
  }
  const auto nq = static_cast<double>(c.num_queries);
  const double match =
      (!c.correlated || n_match == 0) ? -1 : recall_match / static_cast<double>(n_match);
  const double nomatch =
      (!c.correlated || n_match == c.num_queries)
          ? -1
          : (recall - recall_match) / static_cast<double>(c.num_queries - n_match);
  std::ostringstream out;
  out.precision(9);
  out << "{\"qps\": " << nq / seconds << ", \"recall\": " << recall / nq
      << ", \"recall_own_match\": " << match << ", \"recall_own_nomatch\": " << nomatch
      << ", \"prefilter_fraction\": " << prefilter / nq
      << ", \"fallback_fraction\": " << fallback / nq << ", \"mean_graph_distances\": "
      << (graph_searches == 0 ? 0 : distances / static_cast<double>(graph_searches)) << "}";
  return out.str();
}

// The runs of one (strategy, ef) point after an untimed warmup pass, as a JSON array body.
std::string measure(const FilterCase& c, const strata::FilteredSearchOptions& options,
                    std::size_t runs) {
  (void)timed_pass(c, options);  // warmup
  std::string out;
  for (std::size_t run = 0; run < runs; ++run) {
    out += (run == 0 ? "" : ", ") + timed_pass(c, options);
  }
  return out;
}

// The filter for one selectivity, and the clusters it covers (correlated) or none (random).
std::pair<strata::Filter, std::vector<std::int64_t>> make_filter(const Options& opt, double s,
                                                                 const Clusters& clusters) {
  if (opt.kind == "random") {
    return {strata::Filter::range("bucket", 0,
                                  static_cast<std::int64_t>(s * static_cast<double>(kBuckets)) - 1),
            {}};
  }
  auto chosen = clusters_for(s, clusters.base);
  std::vector<strata::AttributeValue> values(chosen.begin(), chosen.end());
  return {strata::Filter::in("cluster", values), chosen};
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  auto ds = check(strata::load_dataset(opt.data));
  const std::size_t n = ds.base.rows();
  const std::size_t num_queries = std::min(opt.max_queries, ds.query.rows());
  strata::ThreadPool pool(opt.build_threads);

  double build_seconds = 0;
  std::optional<strata::HnswIndex> index;
  if (fs::exists(opt.snapshot)) {
    index.emplace(check(strata::HnswIndex::load(opt.snapshot)));
  } else {
    index.emplace(check(strata::HnswIndex::create(ds.base.cols(), strata::Metric::kL2, {})));
    const auto start = Clock::now();
    check(index->add_batch(ds.base, pool));
    build_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    check(index->save(opt.snapshot));
  }
  if (index->size() != n) {
    fail("snapshot does not match the dataset");
  }

  auto table = check(strata::AttributeTable::create(
      {{"bucket", strata::ColumnType::kInt}, {"cluster", strata::ColumnType::kInt}}));
  const Clusters clusters = load_or_compute_clusters(opt, ds, num_queries, pool);
  for (std::size_t id = 0; id < n; ++id) {
    check(table.append({static_cast<std::int64_t>(splitmix64(id) % kBuckets),
                        static_cast<std::int64_t>(clusters.base[id])}));
  }

  std::string points;
  for (double s : opt.selectivities) {
    const auto [filter, chosen] = make_filter(opt, s, clusters);
    const auto compiled = check(strata::CompiledFilter::compile(filter, table));
    const auto matching = compiled.evaluate().count();
    FilterCase c{.index = &*index,
                 .ds = &ds,
                 .num_queries = num_queries,
                 .correlated = opt.kind == "correlated",
                 .filter = &compiled,
                 .own_match = std::vector<bool>(num_queries, false),
                 .truth = std::vector<QueryTruth>(num_queries)};
    std::size_t n_match = 0;
    for (std::size_t q = 0; q < num_queries; ++q) {
      c.own_match[q] =
          std::ranges::find(chosen, static_cast<std::int64_t>(clusters.queries[q])) != chosen.end();
      n_match += c.own_match[q] ? 1 : 0;
      auto exact = check(index->search_filtered(ds.query.row(q), kK, compiled,
                                                {.strategy = strata::FilterStrategy::kPreFilter}));
      c.truth[q] = {.expected = exact.size(), .kth = exact.empty() ? 0.0F : exact.back().distance};
    }
    const double actual = static_cast<double>(matching) / static_cast<double>(n);
    std::cerr << opt.kind << " s=" << s << ": " << matching << " matching (" << actual << ")\n";

    std::vector<std::pair<std::string, std::size_t>> configs;
    if (opt.phase == "graph") {
      configs.emplace_back("prefilter", 0);
    }
    for (double ef : opt.ef_search) {
      configs.emplace_back(opt.phase == "graph" ? "graph" : "auto", static_cast<std::size_t>(ef));
    }
    for (const auto& [strategy, ef] : configs) {
      const strata::FilteredSearchOptions options{.ef_search = ef == 0 ? 64 : ef,
                                                  .strategy = strategy_of(strategy),
                                                  .prefilter_below = opt.threshold};
      std::ostringstream point;
      point << (points.empty() ? "" : ",\n    ") << "{\"kind\": \"" << opt.kind
            << "\", \"target_selectivity\": " << s << ", \"selectivity\": " << actual
            << ", \"matching\": " << matching << ", \"queries_own_match\": " << n_match
            << ", \"strategy\": \"" << strategy << "\", \"ef_search\": " << ef
            << ", \"threshold\": " << opt.threshold << ", \"runs\": ["
            << measure(c, options, opt.runs) << "]}";
      points += point.str();
      std::cerr << "  " << strategy << " ef=" << ef << " done\n";
    }
  }
#ifdef NDEBUG
  const bool asserts = false;
#else
  const bool asserts = true;
#endif
  std::cout.precision(9);
  std::cout << "{\"n\": " << n << ", \"dim\": " << ds.base.cols() << ", \"k\": " << kK
            << ", \"queries\": " << num_queries << ", \"asserts\": " << (asserts ? "true" : "false")
            << ", \"build_seconds\": " << build_seconds << ", \"kind\": \"" << opt.kind
            << "\", \"phase\": \"" << opt.phase << "\", \"clusters\": " << kClusters
            << ",\n  \"points\": [\n    " << points << "\n  ]}\n";
  return 0;
}
