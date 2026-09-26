// Filtered-search benchmark: QPS and recall per (strategy, selectivity), single thread.
//
// Each base vector gets a synthetic int attribute bucket = hash(id) % 1000, so the filter
// `bucket in [0, p * 1000)` selects a fraction p of the data, spread uniformly (no correlation
// with the vectors). Ground truth is the exact filtered top-k. Filter evaluation is inside the
// timed region: in practice each query brings its own filter.
//
// Strategies:
//   prefilter   evaluate the filter to a bitset, score only matching ids (bitset iteration)
//   scan_check  visit every id, test the compiled filter, score matches
// (An in-graph strategy for HNSW is added with the HNSW implementation.)
//
//   strata_filter_bench --data data/siftsmall --metric l2 --selectivity 0.01,0.1,0.5

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "strata/brute_force.hpp"
#include "strata/dataset.hpp"
#include "strata/filter.hpp"
#include "strata/recall.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string data;
  std::string metric = "l2";
  std::size_t k = 10;
  std::size_t runs = 5;
  std::size_t max_queries = 0;
  std::vector<double> selectivity{0.01, 0.1, 0.5};
};

[[noreturn]] void usage(const std::string& error) {
  std::cerr << "error: " << error << "\nusage: strata_filter_bench --data DIR [--metric l2]"
            << " [--k 10] [--runs 5] [--max-queries N] [--selectivity 0.01,0.1,0.5]\n";
  std::exit(2);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  if (argc % 2 == 0) {
    usage("every flag takes a value");
  }
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const std::string value = argv[i + 1];
    if (flag == "--data") {
      opt.data = value;
    } else if (flag == "--metric") {
      opt.metric = value;
    } else if (flag == "--k") {
      opt.k = std::stoul(value);
    } else if (flag == "--runs") {
      opt.runs = std::stoul(value);
    } else if (flag == "--max-queries") {
      opt.max_queries = std::stoul(value);
    } else if (flag == "--selectivity") {
      opt.selectivity.clear();
      std::stringstream ss(value);
      std::string item;
      while (std::getline(ss, item, ',')) {
        opt.selectivity.push_back(std::stod(item));
      }
    } else {
      usage("unknown flag " + flag);
    }
  }
  if (opt.data.empty()) {
    usage("--data is required");
  }
  return opt;
}

// splitmix64: a well-mixed deterministic hash, so buckets don't correlate with id order.
std::uint64_t mix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

double percentile(std::vector<double> v, double p) {
  std::ranges::sort(v);
  const auto rank = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(v.size())));
  return v[std::clamp<std::size_t>(rank, 1, v.size()) - 1];
}

template <typename T>
T check(strata::Expected<T> r) {
  if (!r) {
    std::cerr << "error: " << r.error().message << "\n";
    std::exit(1);
  }
  return std::move(*r);
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  const auto metric = strata::parse_metric(opt.metric);
  if (!metric) {
    usage("unknown metric " + opt.metric);
  }
  auto dataset = check(strata::load_dataset(opt.data));
  const auto& base = dataset.base;
  const std::size_t nq =
      opt.max_queries == 0 ? dataset.query.rows() : std::min(opt.max_queries, dataset.query.rows());

  auto index = check(strata::BruteForceIndex::create(base.cols(), *metric));
  if (!index.add_batch(base)) {
    return 1;
  }
  auto table = check(strata::AttributeTable::create({{"bucket", strata::ColumnType::kInt}}));
  for (std::size_t i = 0; i < base.rows(); ++i) {
    if (!table.append({static_cast<std::int64_t>(mix(i) % 1000)})) {
      return 1;
    }
  }

  std::ostringstream points;
  points.precision(9);
  bool first_point = true;
  for (double p : opt.selectivity) {
    const auto upper = static_cast<std::int64_t>(std::llround(p * 1000.0)) - 1;
    const strata::Filter filter = strata::Filter::range("bucket", 0, upper);
    auto compiled = check(strata::CompiledFilter::compile(filter, table));
    const strata::Bitset allowed = compiled.evaluate();
    const double actual = static_cast<double>(allowed.count()) / static_cast<double>(base.rows());

    // Exact filtered ground truth (k-th distance per query, for tie-aware recall).
    std::vector<float> kth(nq);
    for (std::size_t q = 0; q < nq; ++q) {
      auto truth = check(index.search_filtered(dataset.query.row(q), opt.k, allowed));
      kth[q] = truth.empty() ? 0.0F : truth.back().distance;
    }

    using Strategy =
        std::function<strata::Expected<std::vector<strata::Neighbor>>(std::span<const float>)>;
    const std::vector<std::pair<std::string, Strategy>> strategies = {
        {"prefilter",
         [&](std::span<const float> q) {
           const auto compiled_q = strata::CompiledFilter::compile(filter, table);
           return index.search_filtered(q, opt.k, compiled_q->evaluate());
         }},
        {"scan_check",
         [&](std::span<const float> q) {
           const auto compiled_q = strata::CompiledFilter::compile(filter, table);
           return index.search_predicate(
               q, opt.k, [&](strata::VectorId id) { return compiled_q->matches(id); });
         }},
    };
    for (const auto& [name, search] : strategies) {
      std::ostringstream runs;
      runs.precision(9);
      for (std::size_t run = 0; run < opt.runs; ++run) {
        for (std::size_t q = 0; q < nq; ++q) {
          (void)search(dataset.query.row(q));  // warmup
        }
        std::vector<std::vector<strata::Neighbor>> results(nq);
        std::vector<double> latencies(nq);
        const auto start = Clock::now();
        for (std::size_t q = 0; q < nq; ++q) {
          const auto t0 = Clock::now();
          results[q] = check(search(dataset.query.row(q)));
          latencies[q] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        const double recall = check(strata::recall_at_k_with_ties(results, kth, opt.k));
        double mean = 0;
        for (double l : latencies) {
          mean += l;
        }
        runs << (run == 0 ? "" : ", ") << "{\"search_seconds\": " << seconds
             << ", \"qps\": " << static_cast<double>(nq) / seconds << ", \"recall\": " << recall
             << ", \"latency_mean_us\": " << mean / static_cast<double>(nq)
             << ", \"latency_p50_us\": " << percentile(latencies, 50)
             << ", \"latency_p95_us\": " << percentile(latencies, 95)
             << ", \"latency_p99_us\": " << percentile(latencies, 99) << "}";
        std::cerr << name << " selectivity=" << actual << " run " << run + 1
                  << ": qps=" << static_cast<double>(nq) / seconds << " recall=" << recall << "\n";
      }
      points << (first_point ? "" : ",\n") << "    {\"search_params\": {\"strategy\": \"" << name
             << "\", \"selectivity\": " << p << ", \"actual_selectivity\": " << actual
             << "}, \"runs\": [" << runs.str() << "]}";
      first_point = false;
    }
  }

  std::cout << "{\n  \"harness\": \"strata_filter_bench\",\n  \"build_type\": \"" <<
#ifdef STRATA_BUILD_TYPE
      STRATA_BUILD_TYPE
#else
      "unknown"
#endif
            << "\",\n"
#ifdef NDEBUG
            << "  \"asserts\": false,\n"
#else
            << "  \"asserts\": true,\n"
#endif
            << "  \"kernel\": \"" << strata::best_kernel_name() << "\",\n  \"metric\": \""
            << strata::to_string(*metric) << "\",\n  \"k\": " << opt.k
            << ",\n  \"num_base\": " << base.rows() << ",\n  \"num_queries\": " << nq
            << ",\n  \"dim\": " << base.cols() << ",\n  \"points\": [\n"
            << points.str() << "\n  ]\n}\n";
  return 0;
}
