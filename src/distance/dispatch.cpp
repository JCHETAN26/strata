#include "strata/distance.hpp"

namespace strata {

namespace {

DistanceFn scalar_kernel(Metric metric) noexcept {
  switch (metric) {
    case Metric::kL2:
      return &scalar::l2_squared;
    case Metric::kInnerProduct:
      return &scalar::inner_product;
    case Metric::kCosine:
      return &scalar::cosine_distance;
  }
  return &scalar::l2_squared;
}

}  // namespace

std::string_view best_kernel_name() noexcept {
#if defined(STRATA_HAS_NEON)
  return "neon";
#elif defined(STRATA_HAS_AVX2)
  return "avx2";
#else
  return "scalar";
#endif
}

DistanceFn distance_function(Metric metric, KernelSet kernels) noexcept {
  if (kernels == KernelSet::kScalar) {
    return scalar_kernel(metric);
  }
#if defined(STRATA_HAS_NEON)
  namespace simd = neon;
#elif defined(STRATA_HAS_AVX2)
  namespace simd = avx2;
#endif
#if defined(STRATA_HAS_NEON) || defined(STRATA_HAS_AVX2)
  switch (metric) {
    case Metric::kL2:
      return &simd::l2_squared;
    case Metric::kInnerProduct:
      return &simd::inner_product;
    case Metric::kCosine:
      return &simd::cosine_distance;
  }
#endif
  return scalar_kernel(metric);
}

}  // namespace strata
