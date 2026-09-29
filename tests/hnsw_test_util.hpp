#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "strata/hnsw.hpp"

namespace strata::test {

// Whether two HNSW indexes hold the same graph: everything that determines future behavior
// (vectors, tombstones, levels, every neighbor list, entry point, parameters). The level
// generator's state is not observable; tests check it by adding more vectors to both indexes.
inline ::testing::AssertionResult same_hnsw_graph(const HnswIndex& a, const HnswIndex& b) {
  const auto differ = [](const std::string& what) {
    return ::testing::AssertionFailure() << "graphs differ: " << what;
  };
  if (a.size() != b.size() || a.live_size() != b.live_size()) {
    return differ("size " + std::to_string(a.size()) + " vs " + std::to_string(b.size()) +
                  ", live " + std::to_string(a.live_size()) + " vs " +
                  std::to_string(b.live_size()));
  }
  if (a.dim() != b.dim() || a.metric() != b.metric()) {
    return differ("dimension or metric");
  }
  if (a.entry_point() != b.entry_point() || a.max_level() != b.max_level()) {
    return differ("entry point or max level");
  }
  const auto& pa = a.params();
  const auto& pb = b.params();
  if (pa.M != pb.M || pa.ef_construction != pb.ef_construction || pa.seed != pb.seed ||
      pa.selection != pb.selection) {
    return differ("parameters");
  }
  for (VectorId id = 0; id < a.size(); ++id) {
    if (a.level(id) != b.level(id) || a.is_deleted(id) != b.is_deleted(id) ||
        !std::ranges::equal(a.vector(id), b.vector(id))) {
      return differ("node " + std::to_string(id));
    }
    for (int layer = 0; layer <= a.level(id); ++layer) {
      if (!std::ranges::equal(a.neighbors(id, layer), b.neighbors(id, layer))) {
        return differ("node " + std::to_string(id) + " layer " + std::to_string(layer));
      }
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace strata::test
