// NEON distance kernels for arm64. Compiled only when __ARM_NEON is defined.
//
// Structure (same for all three): 4 independent float32x4 accumulators, so each iteration
// processes 16 floats and the four FMA dependency chains overlap in the pipeline (a single
// accumulator would stall on FMA latency every iteration). Then one 4-lane loop, then a scalar
// tail for dim % 4.

#include "strata/distance.hpp"

#if defined(STRATA_HAS_NEON)

#include <arm_neon.h>

#include <cassert>
#include <cmath>
#include <cstddef>

namespace strata::neon {

float l2_squared(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  float32x4_t acc2 = vdupq_n_f32(0.0F);
  float32x4_t acc3 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(pa + i), vld1q_f32(pb + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(pa + i + 4), vld1q_f32(pb + i + 4));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(pa + i + 8), vld1q_f32(pb + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(pa + i + 12), vld1q_f32(pb + i + 12));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
    acc2 = vfmaq_f32(acc2, d2, d2);
    acc3 = vfmaq_f32(acc3, d3, d3);
  }
  for (; i + 4 <= n; i += 4) {
    const float32x4_t d = vsubq_f32(vld1q_f32(pa + i), vld1q_f32(pb + i));
    acc0 = vfmaq_f32(acc0, d, d);
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; i < n; ++i) {
    const float d = pa[i] - pb[i];
    sum += d * d;
  }
  return sum;
}

namespace {
float dot(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  float32x4_t acc2 = vdupq_n_f32(0.0F);
  float32x4_t acc3 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(pa + i), vld1q_f32(pb + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(pa + i + 4), vld1q_f32(pb + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(pa + i + 8), vld1q_f32(pb + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(pa + i + 12), vld1q_f32(pb + i + 12));
  }
  for (; i + 4 <= n; i += 4) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(pa + i), vld1q_f32(pb + i));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; i < n; ++i) {
    sum += pa[i] * pb[i];
  }
  return sum;
}
}  // namespace

float inner_product(std::span<const float> a, std::span<const float> b) noexcept {
  return -dot(a, b);
}

float cosine_distance(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  // Three sums (a.b, a.a, b.b) with two accumulators each: 6 independent chains.
  float32x4_t ab0 = vdupq_n_f32(0.0F);
  float32x4_t ab1 = vdupq_n_f32(0.0F);
  float32x4_t aa0 = vdupq_n_f32(0.0F);
  float32x4_t aa1 = vdupq_n_f32(0.0F);
  float32x4_t bb0 = vdupq_n_f32(0.0F);
  float32x4_t bb1 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const float32x4_t x0 = vld1q_f32(pa + i);
    const float32x4_t y0 = vld1q_f32(pb + i);
    const float32x4_t x1 = vld1q_f32(pa + i + 4);
    const float32x4_t y1 = vld1q_f32(pb + i + 4);
    ab0 = vfmaq_f32(ab0, x0, y0);
    ab1 = vfmaq_f32(ab1, x1, y1);
    aa0 = vfmaq_f32(aa0, x0, x0);
    aa1 = vfmaq_f32(aa1, x1, x1);
    bb0 = vfmaq_f32(bb0, y0, y0);
    bb1 = vfmaq_f32(bb1, y1, y1);
  }
  for (; i + 4 <= n; i += 4) {
    const float32x4_t x = vld1q_f32(pa + i);
    const float32x4_t y = vld1q_f32(pb + i);
    ab0 = vfmaq_f32(ab0, x, y);
    aa0 = vfmaq_f32(aa0, x, x);
    bb0 = vfmaq_f32(bb0, y, y);
  }
  float ab = vaddvq_f32(vaddq_f32(ab0, ab1));
  float aa = vaddvq_f32(vaddq_f32(aa0, aa1));
  float bb = vaddvq_f32(vaddq_f32(bb0, bb1));
  for (; i < n; ++i) {
    ab += pa[i] * pb[i];
    aa += pa[i] * pa[i];
    bb += pb[i] * pb[i];
  }
  if (aa == 0.0F || bb == 0.0F) {
    return 1.0F;
  }
  return 1.0F - ab / std::sqrt(aa * bb);
}

}  // namespace strata::neon

#endif  // STRATA_HAS_NEON
