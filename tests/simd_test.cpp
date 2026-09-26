// Every SIMD kernel must match the scalar reference within kSimdTolerance (see distance.hpp),
// and exactly on integer-valued inputs.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "strata/distance.hpp"

namespace strata {
namespace {

struct KernelCase {
  std::string name;
  DistanceFn simd;
  DistanceFn reference;
  // Sum of |term_i|: the scale that rounding error is proportional to.
  float (*magnitude)(std::span<const float>, std::span<const float>);
};

float l2_magnitude(std::span<const float> a, std::span<const float> b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    s += d * d;
  }
  return static_cast<float>(s);
}

float dot_magnitude(std::span<const float> a, std::span<const float> b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    s += std::abs(static_cast<double>(a[i]) * b[i]);
  }
  return static_cast<float>(s);
}

// Cosine: error in 1 - ab/sqrt(aa*bb) is relative, so scale is 1 (plus the ratio's magnitude).
float cosine_magnitude(std::span<const float> /*a*/, std::span<const float> /*b*/) { return 1.0F; }

std::vector<KernelCase> simd_kernels() {
  std::vector<KernelCase> cases;
#if defined(STRATA_HAS_NEON)
  cases.push_back({"neon_l2", &neon::l2_squared, &scalar::l2_squared, &l2_magnitude});
  cases.push_back({"neon_ip", &neon::inner_product, &scalar::inner_product, &dot_magnitude});
  cases.push_back(
      {"neon_cosine", &neon::cosine_distance, &scalar::cosine_distance, &cosine_magnitude});
#endif
#if defined(STRATA_HAS_AVX2)
  cases.push_back({"avx2_l2", &avx2::l2_squared, &scalar::l2_squared, &l2_magnitude});
  cases.push_back({"avx2_ip", &avx2::inner_product, &scalar::inner_product, &dot_magnitude});
  cases.push_back(
      {"avx2_cosine", &avx2::cosine_distance, &scalar::cosine_distance, &cosine_magnitude});
#endif
  return cases;
}

class SimdKernel : public ::testing::TestWithParam<KernelCase> {};

// Every dimension from 0 to 130 exercises each unrolled / 4-or-8-lane / tail combination;
// the larger ones cover real embedding sizes.
std::vector<std::size_t> test_dims() {
  std::vector<std::size_t> dims;
  for (std::size_t d = 0; d <= 130; ++d) {
    dims.push_back(d);
  }
  for (std::size_t d : {255U, 256U, 257U, 384U, 768U, 960U, 1536U, 4096U}) {
    dims.push_back(d);
  }
  return dims;
}

TEST_P(SimdKernel, MatchesScalarWithinTolerance) {
  const auto& kc = GetParam();
  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.0F, 10.0F);
  float worst_ratio = 0;
  for (std::size_t dim : test_dims()) {
    for (int trial = 0; trial < 20; ++trial) {
      std::vector<float> a(dim);
      std::vector<float> b(dim);
      for (std::size_t i = 0; i < dim; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
      }
      const float got = kc.simd(a, b);
      const float want = kc.reference(a, b);
      const float bound = kSimdTolerance * kc.magnitude(a, b) + 1e-6F;
      ASSERT_LE(std::abs(got - want), bound)
          << kc.name << " dim=" << dim << " simd=" << got << " scalar=" << want;
      worst_ratio = std::max(worst_ratio, std::abs(got - want) / bound);
    }
  }
  // How much of the tolerance was used; visible with --gtest_also_run_disabled_tests -v style runs.
  RecordProperty("worst_error_over_tolerance", std::to_string(worst_ratio));
}

// SIFT-like data: small non-negative integers. Every partial sum is an integer below 2^24, so it
// is exactly representable and summation order cannot matter.
TEST_P(SimdKernel, ExactOnIntegerData) {
  const auto& kc = GetParam();
  if (kc.name.find("cosine") != std::string::npos) {
    GTEST_SKIP() << "cosine divides and takes a square root; not exact";
  }
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> dist(0, 255);
  for (std::size_t dim : {1U, 7U, 16U, 100U, 128U, 129U, 200U}) {
    std::vector<float> a(dim);
    std::vector<float> b(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      a[i] = static_cast<float>(dist(rng));
      b[i] = static_cast<float>(dist(rng));
    }
    EXPECT_EQ(kc.simd(a, b), kc.reference(a, b)) << kc.name << " dim=" << dim;
  }
}

TEST_P(SimdKernel, IdenticalVectors) {
  const auto& kc = GetParam();
  std::vector<float> a(100);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(i) * 0.25F - 12.0F;
  }
  if (kc.name.find("l2") != std::string::npos) {
    EXPECT_EQ(kc.simd(a, a), 0.0F);
  } else if (kc.name.find("cosine") != std::string::npos) {
    EXPECT_NEAR(kc.simd(a, a), 0.0F, 1e-6F);
  }
}

TEST_P(SimdKernel, CosineZeroVectorIsOne) {
  const auto& kc = GetParam();
  if (kc.name.find("cosine") == std::string::npos) {
    GTEST_SKIP();
  }
  const std::vector<float> zero(37, 0.0F);
  const std::vector<float> b(37, 1.0F);
  EXPECT_EQ(kc.simd(zero, b), 1.0F);
  EXPECT_EQ(kc.simd(zero, zero), 1.0F);
}

// Kernels must not read past the end: place the vector at the very end of an allocation.
// (ASan catches any overrun in the asan preset.)
TEST_P(SimdKernel, NoReadPastEnd) {
  const auto& kc = GetParam();
  for (std::size_t dim : {1U, 3U, 5U, 9U, 17U, 33U}) {
    std::vector<float> a(dim, 1.0F);
    std::vector<float> b(dim, 2.0F);
    EXPECT_TRUE(std::isfinite(kc.simd(a, b))) << kc.name << " dim=" << dim;
  }
}

INSTANTIATE_TEST_SUITE_P(Kernels, SimdKernel, ::testing::ValuesIn(simd_kernels()),
                         [](const auto& info) { return info.param.name; });
// No SIMD kernels on this architecture is fine: the scalar path is the only one.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SimdKernel);

TEST(KernelDispatch, ScalarSetReturnsReference) {
  EXPECT_EQ(distance_function(Metric::kL2, KernelSet::kScalar), &scalar::l2_squared);
  EXPECT_EQ(distance_function(Metric::kInnerProduct, KernelSet::kScalar), &scalar::inner_product);
  EXPECT_EQ(distance_function(Metric::kCosine, KernelSet::kScalar), &scalar::cosine_distance);
}

TEST(KernelDispatch, BestMatchesBuildArchitecture) {
#if defined(STRATA_HAS_NEON)
  EXPECT_EQ(best_kernel_name(), "neon");
  EXPECT_EQ(distance_function(Metric::kL2), &neon::l2_squared);
#elif defined(STRATA_HAS_AVX2)
  EXPECT_EQ(best_kernel_name(), "avx2");
  EXPECT_EQ(distance_function(Metric::kL2), &avx2::l2_squared);
#else
  EXPECT_EQ(best_kernel_name(), "scalar");
  EXPECT_EQ(distance_function(Metric::kL2), &scalar::l2_squared);
#endif
}

}  // namespace
}  // namespace strata
