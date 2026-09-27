#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
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
// Thread safety: concurrent calls to const methods (search and the accessors) are safe; each
// thread uses its own visited-set scratch buffer. add/add_batch are single-writer: they require
// exclusive access, so no other call (including search) may run concurrently with them.
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
  HnswIndex(std::size_t dim, Metric metric, HnswParams params);

  // Insertion (paper Algorithm 1). Precondition: values.size() == dim(), values does not alias
  // this index's storage, and the id space has room.
  VectorId insert(std::span<const float> values);
  // floor(-ln(U) * mL), U uniform in (0, 1].
  int random_level();
  void reserve(std::size_t n);
  [[nodiscard]] bool aliases_storage(std::span<const float> values) const noexcept;

  [[nodiscard]] float dist(std::span<const float> query, VectorId id) const noexcept {
    return distance_(query, vector(id));
  }
  // Greedy walk on one layer: move to the closest neighbor until none is closer (ef = 1).
  [[nodiscard]] Neighbor greedy_search(std::span<const float> query, Neighbor start,
                                       int layer) const;
  // Beam search on one layer (paper Algorithm 2). Returns up to ef nodes, sorted ascending.
  [[nodiscard]] std::vector<Neighbor> search_layer(std::span<const float> query,
                                                   const std::vector<Neighbor>& entry_points,
                                                   std::size_t ef, int layer) const;
  // Shrinks sorted candidates to at most m neighbors (paper Algorithm 3).
  static void select_neighbors(std::vector<Neighbor>& candidates, std::size_t m);
  // Replaces id's out-links on a layer. Precondition: links.size() <= capacity(layer).
  void set_links(VectorId id, int layer, std::span<const Neighbor> links);
  // Adds the edge from -> to.id, re-selecting from's neighbors if the list overflows.
  // to.distance must be the distance between the two nodes.
  void add_link(VectorId from, Neighbor to, int layer);
  [[nodiscard]] std::size_t capacity(int layer) const noexcept {
    return layer == 0 ? max_links0_ : params_.M;
  }
  // A node's link list on a layer: slot 0 holds the count, slots 1..capacity(layer) the ids.
  [[nodiscard]] VectorId* link_list(VectorId id, int layer) noexcept;
  [[nodiscard]] const VectorId* link_list(VectorId id, int layer) const noexcept;

  std::size_t dim_;
  Metric metric_;
  HnswParams params_;
  DistanceFn distance_;     // best SIMD kernel for the metric, looked up once
  std::size_t max_links0_;  // M_max0 = 2 * M
  double level_mult_;       // mL = 1 / ln(M)
  std::mt19937_64 rng_;     // level generator, seeded from params.seed

  // Node id is the index into every array below.
  std::vector<float> data_;           // size() * dim floats, row-major
  std::vector<std::uint8_t> levels_;  // top layer per node (never above 53; see random_level)
  // Layer 0: fixed stride of (max_links0_ + 1) ids per node, so a node's list is at
  // links0_[id * (max_links0_ + 1)] with no pointer chase.
  std::vector<VectorId> links0_;
  // Layers >= 1: level(id) blocks of (M + 1) ids; block (layer - 1) is that layer's list. Empty
  // for the ~(1 - 1/M) of nodes that live only on layer 0.
  std::vector<std::vector<VectorId>> upper_links_;
  std::optional<VectorId> entry_point_;
  int max_level_ = -1;
};

}  // namespace strata
