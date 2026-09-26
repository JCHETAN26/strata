#include <cassert>
#include <cmath>
#include <cstddef>

#include "strata/distance.hpp"

namespace strata {

namespace scalar {

float l2_squared(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  float sum = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

namespace {
float dot(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  float sum = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}
}  // namespace

float inner_product(std::span<const float> a, std::span<const float> b) noexcept {
  return -dot(a, b);
}

float cosine_distance(std::span<const float> a, std::span<const float> b) noexcept {
  assert(a.size() == b.size());
  float ab = 0.0F;
  float aa = 0.0F;
  float bb = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  if (aa == 0.0F || bb == 0.0F) {
    return 1.0F;
  }
  return 1.0F - ab / std::sqrt(aa * bb);
}

}  // namespace scalar

std::string_view to_string(Metric metric) noexcept {
  switch (metric) {
    case Metric::kL2:
      return "l2";
    case Metric::kInnerProduct:
      return "ip";
    case Metric::kCosine:
      return "cosine";
  }
  return "unknown";
}

std::optional<Metric> parse_metric(std::string_view name) noexcept {
  if (name == "l2") {
    return Metric::kL2;
  }
  if (name == "ip") {
    return Metric::kInnerProduct;
  }
  // ann-benchmarks calls cosine "angular".
  if (name == "cosine" || name == "angular") {
    return Metric::kCosine;
  }
  return std::nullopt;
}

DistanceFn distance_function(Metric metric) noexcept {
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

}  // namespace strata
