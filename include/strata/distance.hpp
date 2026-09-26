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

// Kernel for the metric. Look it up once, outside the loop.
[[nodiscard]] DistanceFn distance_function(Metric metric) noexcept;

// Convenience for non-hot-path code.
[[nodiscard]] inline float distance(Metric metric, std::span<const float> a,
                                    std::span<const float> b) noexcept {
  return distance_function(metric)(a, b);
}

}  // namespace strata
