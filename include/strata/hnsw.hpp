#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/matrix.hpp"
#include "strata/snapshot.hpp"
#include "strata/types.hpp"

namespace strata {

// How a node's neighbors are chosen from its candidates (on insert, and when a full neighbor list
// is re-selected).
enum class NeighborSelection {
  // The M closest candidates (paper Algorithm 3). Kept as a baseline for comparison.
  kSimple,
  // Paper Algorithm 4: take candidates nearest first, skipping any that is closer to an
  // already-selected neighbor than to the node itself. Keeps links pointing in diverse
  // directions, so clusters stay connected.
  kHeuristic,
};

// Construction parameters, named as in Malkov & Yashunin (2018) and hnswlib.
struct HnswParams {
  // Max neighbors per node on layers >= 1. Layer 0 allows 2 * M (M_max0 in the paper).
  std::size_t M = 16;
  // Candidate list size while inserting. Larger = better graph, slower build.
  std::size_t ef_construction = 200;
  // Seed for level assignment. Same seed + same insertion order = same graph.
  std::uint64_t seed = 42;
  NeighborSelection selection = NeighborSelection::kHeuristic;
};

// Hierarchical Navigable Small World graph index (approximate k-NN).
//
// Spec that tests/hnsw_test.cpp checks:
//   - Levels are drawn as floor(-ln(U) * mL) with mL = 1 / ln(M), U uniform in (0, 1], so a node
//     reaches layer >= 1 with probability 1/M.
//   - Degree bounds: at most 2 * M neighbors on layer 0, at most M on layers >= 1.
//   - Neighbor lists hold no self-loops, no duplicates, and only nodes present on that layer.
//   - The entry point is a node on the top layer.
//   - search() returns min(k, live nodes reachable from the entry point) results, never a deleted
//     id, sorted by (distance, id), with exact distances and distinct ids. The effective beam width
//     is max(ef_search, k). With no deletes, every node is live.
//
// Deletes are tombstones: a deleted node stays in the graph. Searches still walk through it (it
// keeps the graph connected) and inserts may still link to it, but it never appears in results.
// Trade-offs: its memory is never reclaimed, and searches pay to traverse deleted nodes. Under
// heavy deletion a search keeps expanding until it has found ef_search live nodes (or run out of
// reachable ones), so it still returns k results but grows slower, up to a scan of the whole graph
// when almost everything is deleted. Rebuilding (re-adding the live vectors) is the remedy.
//
// Snapshot index section (IndexKind::kHnsw), little-endian, after the vectors and tombstones that
// the snapshot layer stores (see include/strata/snapshot.hpp):
//   u32 section version (1), u32 selection (0 simple, 1 heuristic)
//   u64 M, u64 ef_construction, u64 seed
//   i32 max_level (-1 if empty), u32 entry point (0xFFFFFFFF if empty)
//   u32 n, then n bytes: the level generator's state as text (std::mt19937_64's stream format,
//       which the C++ standard fixes, so it restores identically on every standard library)
//   count x u8: each node's level
//   count x (2M + 1) u32: layer-0 lists (count, then 2M slots)
//   for each node with level > 0, in id order: level x (M + 1) u32 upper-layer lists
// Everything needed to keep building is saved, so load-then-add builds the same graph as never
// saving. Loading checks the structure (ids in range, neighbors present on their layer, counts
// within capacity, entry point on the top layer), so a damaged file fails to load instead of
// crashing a later search.
//
// Thread safety: concurrent calls to const methods (search, save, and the accessors) are safe;
// each thread uses its own visited-set scratch buffer. add, add_batch, and remove are
// single-writer: they require exclusive access, so no other call (including search) may run
// concurrently with them.
class HnswIndex {
 public:
  // Fails if dim == 0, M < 2, or ef_construction == 0.
  [[nodiscard]] static Expected<HnswIndex> create(std::size_t dim, Metric metric,
                                                  HnswParams params = {});

  // Inserts a vector and returns its id (dense, in insertion order). Fails on dimension mismatch.
  Expected<VectorId> add(std::span<const float> vector);
  // Inserts every row in order. Fails (adding nothing) on dimension mismatch.
  Expected<void> add_batch(MatrixView<const float> vectors);

  // Marks id deleted (a tombstone; see above). Fails with kNotFound if id is out of range or
  // already deleted.
  Expected<void> remove(VectorId id);

  // Approximate k nearest live neighbors. Fails on dimension mismatch. k == 0 or an index with no
  // live nodes gives an empty result.
  [[nodiscard]] Expected<std::vector<Neighbor>> search(std::span<const float> query, std::size_t k,
                                                       std::size_t ef_search) const;

  // Persistence (format above). to_snapshot/from_snapshot are what a collection uses to combine
  // the index with its write-ahead log; save/load write and read a snapshot file directly.
  [[nodiscard]] Snapshot to_snapshot(std::uint64_t last_lsn = 0) const;
  // Fails with kInvalidArgument if the snapshot is not an HNSW snapshot, and with kCorruptData if
  // its index section is malformed.
  [[nodiscard]] static Expected<HnswIndex> from_snapshot(const Snapshot& snapshot);
  [[nodiscard]] Expected<void> save(const std::filesystem::path& path) const;
  [[nodiscard]] static Expected<HnswIndex> load(const std::filesystem::path& path);

  [[nodiscard]] std::size_t size() const noexcept;  // ids assigned, including deleted
  [[nodiscard]] std::size_t live_size() const noexcept;
  // Precondition: id < size().
  [[nodiscard]] bool is_deleted(VectorId id) const noexcept;
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
  // Beam search on one layer (paper Algorithm 2). Returns up to ef nodes, sorted ascending. With
  // kSkipDeleted, deleted nodes are expanded but not returned (see search_layer in the .cpp).
  template <bool kSkipDeleted>
  [[nodiscard]] std::vector<Neighbor> search_layer(std::span<const float> query,
                                                   const std::vector<Neighbor>& entry_points,
                                                   std::size_t ef, int layer) const;
  // Shrinks candidates (sorted by distance to a base node, which they must not contain) to at
  // most m neighbors of that node, per params_.selection.
  void select_neighbors(std::vector<Neighbor>& candidates, std::size_t m) const;
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
  std::vector<std::uint8_t> deleted_;  // 1 = tombstone
  std::size_t num_deleted_ = 0;
  std::optional<VectorId> entry_point_;
  int max_level_ = -1;
};

}  // namespace strata
