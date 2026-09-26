#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/matrix.hpp"
#include "strata/types.hpp"

namespace strata {

// Construction parameters, named as in Malkov & Yashunin (2018) and hnswlib.
struct HnswParams {
  // Max neighbors per node on layers >= 1. Layer 0 allows 2 * M (M_max0 in the paper).
  std::size_t M = 16;
  // Candidate list size while inserting. Larger = better graph, slower build.
  std::size_t ef_construction = 200;
  // Seed for level assignment. Same seed + same insertion order = same graph.
  std::uint64_t seed = 42;
};

// Hierarchical Navigable Small World graph index (approximate k-NN).
//
// Spec that tests/hnsw_test.cpp checks:
//   - Levels are drawn as floor(-ln(U) * mL) with mL = 1 / ln(M), U uniform in (0, 1], so a node
//     reaches layer >= 1 with probability 1/M.
//   - Degree bounds: at most 2 * M neighbors on layer 0, at most M on layers >= 1.
//   - Neighbor lists hold no self-loops, no duplicates, and only nodes present on that layer.
//   - The entry point is a node on the top layer.
//   - search() returns min(k, size()) results, sorted by (distance, id), with exact distances
//     and distinct ids. The effective beam width is max(ef_search, k).
//
// Thread safety: concurrent calls to const methods (search and the accessors) are safe.
// add/add_batch require exclusive access: no other call may run concurrently with them.
class HnswIndex {
 public:
  // Fails if dim == 0, M < 2, or ef_construction == 0.
  [[nodiscard]] static Expected<HnswIndex> create(std::size_t dim, Metric metric,
                                                  HnswParams params = {});

  // Inserts a vector and returns its id (dense, in insertion order). Fails on dimension mismatch.
  Expected<VectorId> add(std::span<const float> vector);
  // Inserts every row in order. Fails (adding nothing) on dimension mismatch.
  Expected<void> add_batch(MatrixView<const float> vectors);

  // Approximate k nearest neighbors. Fails on dimension mismatch. k == 0 or an empty index gives
  // an empty result.
  [[nodiscard]] Expected<std::vector<Neighbor>> search(std::span<const float> query, std::size_t k,
                                                       std::size_t ef_search) const;

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t dim() const noexcept;
  [[nodiscard]] Metric metric() const noexcept;
  [[nodiscard]] const HnswParams& params() const noexcept;
  [[nodiscard]] std::span<const float> vector(VectorId id) const noexcept;

  // Introspection, for tests, debugging, and graph statistics.
  // nullopt when the index is empty.
  [[nodiscard]] std::optional<VectorId> entry_point() const noexcept;
  // Top layer of the graph; -1 when the index is empty.
  [[nodiscard]] int max_level() const noexcept;
  // Top layer of a node. Precondition: id < size().
  [[nodiscard]] int level(VectorId id) const noexcept;
  // Out-neighbors of a node on a layer. Precondition: id < size() and 0 <= layer <= level(id).
  [[nodiscard]] std::span<const VectorId> neighbors(VectorId id, int layer) const noexcept;

 private:
  // Your design: data layout, level generator, and the insertion / neighbor-selection / search
  // routines. Implement everything declared above in src/index/hnsw.cpp.
};

}  // namespace strata
