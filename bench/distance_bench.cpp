#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <vector>

#include "strata/distance.hpp"

namespace {

std::vector<float> random_vector(std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> v(dim);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

template <strata::DistanceFn Fn>
void BM_Distance(benchmark::State& state) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const auto a = random_vector(dim, 1);
  const auto b = random_vector(dim, 2);
  for (auto _ : state) {
    benchmark::DoNotOptimize(Fn(a, b));
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(2 * dim * sizeof(float)));
}

// 100 = GloVe, 128 = SIFT, 384/768 = common sentence-embedding sizes.
#define STRATA_DIMS Arg(100)->Arg(128)->Arg(384)->Arg(768)

BENCHMARK(BM_Distance<&strata::scalar::l2_squared>)->Name("scalar/l2")->STRATA_DIMS;
BENCHMARK(BM_Distance<&strata::scalar::inner_product>)->Name("scalar/ip")->STRATA_DIMS;
BENCHMARK(BM_Distance<&strata::scalar::cosine_distance>)->Name("scalar/cosine")->STRATA_DIMS;

#if defined(STRATA_HAS_NEON)
BENCHMARK(BM_Distance<&strata::neon::l2_squared>)->Name("neon/l2")->STRATA_DIMS;
BENCHMARK(BM_Distance<&strata::neon::inner_product>)->Name("neon/ip")->STRATA_DIMS;
BENCHMARK(BM_Distance<&strata::neon::cosine_distance>)->Name("neon/cosine")->STRATA_DIMS;
#endif

#if defined(STRATA_HAS_AVX2)
BENCHMARK(BM_Distance<&strata::avx2::l2_squared>)->Name("avx2/l2")->STRATA_DIMS;
BENCHMARK(BM_Distance<&strata::avx2::inner_product>)->Name("avx2/ip")->STRATA_DIMS;
BENCHMARK(BM_Distance<&strata::avx2::cosine_distance>)->Name("avx2/cosine")->STRATA_DIMS;
#endif

}  // namespace
