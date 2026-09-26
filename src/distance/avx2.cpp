// AVX2 + FMA distance kernels for x86_64. Compiled only when __AVX2__ and __FMA__ are defined
// (CMake option STRATA_AVX2 adds -mavx2 -mfma).
//
// Structure (same for all three): 2 independent __m256 accumulators, so each iteration processes
// 16 floats and the two FMA dependency chains overlap. Then one 8-lane loop, then a scalar tail
// for dim % 8. Unaligned loads: vectors are not guaranteed 32-byte aligned, and on Zen 3 an
// unaligned load that doesn't cross a cache line costs the same as an aligned one.

#include "strata/distance.hpp"

#if defined(STRATA_HAS_AVX2)

#include <immintrin.h>

#include <cassert>
#include <cmath>
#include <cstddef>

namespace strata::avx2 {

namespace {

// Sum of the 8 lanes.
inline float horizontal_sum(__m256 v) noexcept {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(lo, hi);                 // 4 lanes
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));  // 2 lanes
  sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));     // 1 lane
  return _mm_cvtss_f32(sum);
}

float dot(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa + i + 8), _mm256_loadu_ps(pb + i + 8), acc1);
  }
  for (; i + 8 <= n; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i), acc0);
  }
  float sum = horizontal_sum(_mm256_add_ps(acc0, acc1));
  for (; i < n; ++i) {
    sum += pa[i] * pb[i];
  }
  return sum;
}

}  // namespace

float l2_squared(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(pa + i + 8), _mm256_loadu_ps(pb + i + 8));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  for (; i + 8 <= n; i += 8) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i));
    acc0 = _mm256_fmadd_ps(d, d, acc0);
  }
  float sum = horizontal_sum(_mm256_add_ps(acc0, acc1));
  for (; i < n; ++i) {
    const float d = pa[i] - pb[i];
    sum += d * d;
  }
  return sum;
}

float inner_product(std::span<const float> a, std::span<const float> b) noexcept {
  return -dot(a, b);
}

float cosine_distance(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  __m256 ab = _mm256_setzero_ps();
  __m256 aa = _mm256_setzero_ps();
  __m256 bb = _mm256_setzero_ps();
  std::size_t i = 0;
  // Three independent chains already hide FMA latency; no second set of accumulators needed.
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_loadu_ps(pa + i);
    const __m256 y = _mm256_loadu_ps(pb + i);
    ab = _mm256_fmadd_ps(x, y, ab);
    aa = _mm256_fmadd_ps(x, x, aa);
    bb = _mm256_fmadd_ps(y, y, bb);
  }
  float sab = horizontal_sum(ab);
  float saa = horizontal_sum(aa);
  float sbb = horizontal_sum(bb);
  for (; i < n; ++i) {
    sab += pa[i] * pb[i];
    saa += pa[i] * pa[i];
    sbb += pb[i] * pb[i];
  }
  if (saa == 0.0F || sbb == 0.0F) {
    return 1.0F;
  }
  return 1.0F - sab / std::sqrt(saa * sbb);
}

}  // namespace strata::avx2

#endif  // STRATA_HAS_AVX2
