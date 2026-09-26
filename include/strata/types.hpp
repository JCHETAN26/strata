#pragma once

#include <compare>
#include <cstdint>

namespace strata {

// Dense id assigned in insertion order, starting at 0.
using VectorId = std::uint32_t;

// A search result. Lower distance means more similar, for every metric.
// Ordering is by distance, then id, so results are deterministic when distances tie.
struct Neighbor {
  VectorId id;
  float distance;

  friend bool operator==(const Neighbor&, const Neighbor&) = default;
  friend auto operator<=>(const Neighbor& a, const Neighbor& b) {
    if (auto cmp = a.distance <=> b.distance; cmp != 0) {
      return cmp;
    }
    return static_cast<std::partial_ordering>(a.id <=> b.id);
  }
};

}  // namespace strata
