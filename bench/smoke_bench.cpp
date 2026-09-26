#include <benchmark/benchmark.h>

#include <numeric>
#include <vector>

// Placeholder benchmark: verifies Google Benchmark is wired up.
// Replaced by real distance and search benchmarks in Phase 1.
static void BM_SumFloats(benchmark::State& state) {
  std::vector<float> values(static_cast<std::size_t>(state.range(0)), 1.0F);
  for (auto _ : state) {
    float sum = std::accumulate(values.begin(), values.end(), 0.0F);
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_SumFloats)->Arg(128)->Arg(1024);
