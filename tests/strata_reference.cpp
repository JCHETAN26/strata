// Writes C++ results on a dataset for the Python bindings to be compared against
// (tests/python/test_bindings.py). Also records how this build computes floating point, so the
// test can require bit-identical distances only when the Python module was built the same way.
//
//   strata_reference --data data/siftsmall --out /tmp/strata_ref
//
// Output (little-endian .ibin/.fbin as in scripts/prepare_datasets.py):
//   build_info.json                     kernel, compiler, fp_flags, target, build_type
//   brute_ids.ibin, brute_dist.fbin     exact top-10 for every query (L2)
//   pq_codes.ibin                       PQ codes of the first 1000 base vectors (m=16, seed 42)
//   pq_r{0,100}_ids.ibin, _dist.fbin    PQ search, rerank 0 (ADC) and 100 (exact)
//   filt_{10,100}_ids.ibin, _dist.fbin  pre-filtered search, bucket in [0, 10) and [0, 100)
//                                       where bucket = splitmix64(id) % 1000 (1% and 10%)

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "strata/brute_force.hpp"
#include "strata/build_info.hpp"
#include "strata/dataset.hpp"
#include "strata/filter.hpp"
#include "strata/pq.hpp"

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kK = 10;

template <typename T>
T check(strata::Expected<T> r) {
  if (!r) {
    std::cerr << "error: " << r.error().message << "\n";
    std::exit(1);
  }
  if constexpr (!std::is_void_v<T>) {
    return std::move(*r);
  }
}

// Same hash as bench/filter_bench.cpp and the Python test.
std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

void write_results(const fs::path& out, const std::string& name,
                   const std::vector<std::vector<strata::Neighbor>>& results) {
  strata::Matrix<std::int32_t> ids(results.size(), kK);
  strata::Matrix<float> dist(results.size(), kK);
  for (std::size_t q = 0; q < results.size(); ++q) {
    for (std::size_t i = 0; i < kK; ++i) {
      const bool present = i < results[q].size();
      ids.row(q)[i] = present ? static_cast<std::int32_t>(results[q][i].id) : -1;
      dist.row(q)[i] = present ? results[q][i].distance : std::numeric_limits<float>::infinity();
    }
  }
  check(strata::write_ibin(out / (name + "_ids.ibin"), ids));
  check(strata::write_fbin(out / (name + "_dist.fbin"), dist));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 || std::string(argv[1]) != "--data" || std::string(argv[3]) != "--out") {
    std::cerr << "usage: strata_reference --data DIR --out DIR\n";
    return 2;
  }
  const fs::path out = argv[4];
  fs::create_directories(out);
  const auto ds = check(strata::load_dataset(argv[2]));
  const auto& q = ds.query;

  std::ofstream(out / "build_info.json")
      << "{\"kernel\": \"" << strata::best_kernel_name() << "\", \"compiler\": \""
      << strata::build::kCompiler << "\", \"fp_flags\": \"" << strata::build::kFpFlags
      << "\", \"target\": \"" << strata::build::kTarget << "\", \"build_type\": \""
      << strata::build::kBuildType << "\"}\n";

  // Brute force.
  auto brute = check(strata::BruteForceIndex::create(ds.base.cols(), strata::Metric::kL2));
  check(brute.add_batch(ds.base));
  std::vector<std::vector<strata::Neighbor>> results(q.rows());
  for (std::size_t i = 0; i < q.rows(); ++i) {
    results[i] = check(brute.search(q.row(i), kK));
  }
  write_results(out, "brute", results);

  // Product quantization.
  auto pq = check(strata::ProductQuantizer::train(ds.base, strata::Metric::kL2,
                                                  {.m = 16, .kmeans_iterations = 25, .seed = 42}));
  strata::Matrix<std::int32_t> codes(1000, pq.code_size());
  std::vector<std::uint8_t> code(pq.code_size());
  for (std::size_t i = 0; i < 1000; ++i) {
    pq.encode(ds.base.row(i), code);
    std::copy(code.begin(), code.end(), codes.row(i).begin());
  }
  check(strata::write_ibin(out / "pq_codes.ibin", codes));
  auto pq_index = check(strata::PqIndex::create(pq, /*keep_originals=*/true));
  check(pq_index.add_batch(ds.base));
  for (std::size_t rerank : {0U, 100U}) {
    for (std::size_t i = 0; i < q.rows(); ++i) {
      results[i] = check(pq_index.search(q.row(i), kK, {.rerank = rerank}));
    }
    write_results(out, "pq_r" + std::to_string(rerank), results);
  }

  // Filtered search.
  auto table = check(strata::AttributeTable::create({{"bucket", strata::ColumnType::kInt}}));
  for (std::size_t i = 0; i < ds.base.rows(); ++i) {
    check(table.append({static_cast<std::int64_t>(splitmix64(i) % 1000)}));
  }
  for (std::int64_t upper : {10, 100}) {
    const auto filter = check(
        strata::CompiledFilter::compile(strata::Filter::range("bucket", 0, upper - 1), table));
    const auto allowed = filter.evaluate();
    for (std::size_t i = 0; i < q.rows(); ++i) {
      results[i] = check(brute.search_filtered(q.row(i), kK, allowed));
    }
    write_results(out, "filt_" + std::to_string(upper), results);
  }
  std::cerr << "wrote " << out << "\n";
  return 0;
}
