#pragma once

#include <optional>
#include <span>
#include <string_view>

namespace strata {

// Every metric is expressed as a distance: lower means more similar.
enum class Metric {
  kL2,            // squared Euclidean distance: sum (a_i - b_i)^2
  kInnerProduct,  // negated dot product: -sum a_i * b_i
  kCosine,        // 1 - cos(a, b); defined as 1 when either vector has zero norm
};

[[nodiscard]] std::string_view to_string(Metric metric) noexcept;
[[nodiscard]] std::optional<Metric> parse_metric(std::string_view name) noexcept;

// Distance kernel signature. Precondition: a.size() == b.size() (checked by assert only;
// callers validate dimensions at API boundaries so the hot path has no branches for it).
using DistanceFn = float (*)(std::span<const float> a, std::span<const float> b) noexcept;

// Scalar reference kernels. Accumulate in float, left to right. SIMD kernels are tested against
// these within a documented tolerance.
// Thread safety: pure functions.
namespace scalar {
[[nodiscard]] float l2_squared(std::span<const float> a, std::span<const float> b) noexcept;
[[nodiscard]] float inner_product(std::span<const float> a, std::span<const float> b) noexcept;
[[nodiscard]] float cosine_distance(std::span<const float> a, std::span<const float> b) noexcept;
}  // namespace scalar

#if defined(__ARM_NEON)
#define STRATA_HAS_NEON 1
// NEON kernels (arm64). 4 independent 4-lane FMA accumulators, scalar tail.
namespace neon {
[[nodiscard]] float l2_squared(std::span<const float> a, std::span<const float> b) noexcept;
[[nodiscard]] float inner_product(std::span<const float> a, std::span<const float> b) noexcept;
[[nodiscard]] float cosine_distance(std::span<const float> a, std::span<const float> b) noexcept;
}  // namespace neon
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define STRATA_HAS_AVX2 1
// AVX2 + FMA kernels (x86_64). 2 independent 8-lane FMA accumulators, scalar tail.
namespace avx2 {
[[nodiscard]] float l2_squared(std::span<const float> a, std::span<const float> b) noexcept;
[[nodiscard]] float inner_product(std::span<const float> a, std::span<const float> b) noexcept;
[[nodiscard]] float cosine_distance(std::span<const float> a, std::span<const float> b) noexcept;
}  // namespace avx2
#endif

// SIMD kernels sum in a different order than the scalar reference, so results differ by
// rounding. Tested bound: |simd - scalar| <= kSimdTolerance * sum_i |term_i|, where term_i is
// (a_i - b_i)^2 or a_i * b_i. On integer-valued inputs whose sums stay below 2^24 (e.g. SIFT)
// every partial sum is exact, so results are bit-identical.
inline constexpr float kSimdTolerance = 1e-5F;

// Which kernel family distance_function returns.
enum class KernelSet {
  kBest,    // the SIMD family compiled for this architecture, else scalar
  kScalar,  // always the scalar reference (for with/without-SIMD comparisons)
};

// Name of the family kBest resolves to in this build: "neon", "avx2", or "scalar".
[[nodiscard]] std::string_view best_kernel_name() noexcept;

// Kernel for the metric. Look it up once, outside the loop.
[[nodiscard]] DistanceFn distance_function(Metric metric,
                                           KernelSet kernels = KernelSet::kBest) noexcept;

// Convenience for non-hot-path code.
[[nodiscard]] inline float distance(Metric metric, std::span<const float> a,
                                    std::span<const float> b) noexcept {
  return distance_function(metric)(a, b);
}

}  // namespace strata
