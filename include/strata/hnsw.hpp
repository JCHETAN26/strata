#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "strata/bitset.hpp"
#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/matrix.hpp"
#include "strata/snapshot.hpp"
#include "strata/types.hpp"

namespace strata {

class CompiledFilter;
class ThreadPool;

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

// Filtered search (HnswIndex::search_filtered): how to find the nearest vectors that match.
enum class FilterStrategy {
  // Pick per query: the pre-filter when the filter's estimated selectivity is below
  // FilteredSearchOptions::prefilter_below, else the graph, falling back to the pre-filter if the
  // graph search exceeds its budget.
  kAuto,
  // Search the graph, expanding every node but returning only matching live ones (the tombstone
  // mechanism, with the widened stopping rule). Approximate, like search(). Never falls back.
  kGraph,
  // Evaluate the filter on every id and score each matching live vector. Exact; cost grows with
  // the index size plus the number of matches.
  kPreFilter,
};

// Default for FilteredSearchOptions::prefilter_below: 1.3%, the crossover measured on 200k SIFT
// vectors (1.03-1.28% across random and cluster-correlated filters at recall 0.95-0.99; the
// largest, rounded: results/hnsw_filter/filter_sift1m-200k-q1000.md, docs/explainers/hnsw.md
// section 11). It depends on index size: the pre-filter's cost grows with n, the graph's with ef /
// selectivity. Re-measured at 1M and 10M in the AWS session.
inline constexpr double kDefaultPrefilterBelow = 0.013;

struct FilteredSearchOptions {
  std::size_t ef_search = 64;
  FilterStrategy strategy = FilterStrategy::kAuto;
  // kAuto: use the pre-filter when the estimated selectivity is below this.
  double prefilter_below = kDefaultPrefilterBelow;
  // kAuto: a graph search that computes more than (fallback_budget + estimated selectivity) *
  // size() distances gives up, and the pre-filter answers instead. That is about the pre-filter's
  // own cost in distance computations: one filter test per id (about a tenth of a distance, hence
  // the default) plus one distance per match. 0 disables the fallback.
  double fallback_budget = 0.1;
  // kAuto: the filter's selectivity, if the caller already knows it (for example, counted once for
  // a batch of queries sharing a filter). Unset: estimated per call, from a cached sample for a
  // CompiledFilter or a popcount for a Bitset.
  std::optional<double> selectivity;
};

// What a filtered search did, for measurement and debugging.
struct FilteredSearchStats {
  FilterStrategy used = FilterStrategy::kGraph;  // kGraph or kPreFilter
  bool fell_back = false;                        // kAuto chose the graph, which hit its budget
  double estimated_selectivity = -1;             // kAuto only; -1 otherwise
  bool resampled = false;                        // kAuto needed the larger selectivity sample
  std::size_t graph_distances = 0;               // distances the graph search computed on layer 0
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
// when almost everything is deleted. Rebuilding (re-adding the live vectors) is the remedy; the
// measured guideline (docs/explainers/hnsw.md, section 9) is to rebuild once a quarter to a half
// of the index is deleted, if search speed matters.
//
// Snapshot index section (IndexKind::kHnsw), little-endian, after the vectors and tombstones that
// the snapshot layer stores (see include/strata/snapshot.hpp):
//   u32 section version (2), u32 selection (0 simple, 1 heuristic)
//   u64 M, u64 ef_construction, u64 seed
//   i32 max_level (-1 if empty), u32 entry point (0xFFFFFFFF if empty)
//   u64 number of level-generator draws since seeding. Loading re-seeds std::mt19937_64 with the
//       seed and calls discard(draws), which every standard library implements identically. (The
//       generator's stream text is not portable in practice: libc++ and libstdc++ write different
//       formats. Section version 1, in snapshot format version 2, stored that text and loads only
//       on the standard library that wrote it.)
//   count x u8: each node's level
//   count x (2M + 1) u32: layer-0 lists (count, then 2M slots)
//   for each node with level > 0, in id order: level x (M + 1) u32 upper-layer lists
// Everything needed to keep building is saved, so load-then-add builds the same graph as never
// saving. Loading checks the structure (ids in range, neighbors present on their layer, counts
// within capacity, entry point on the top layer), so a damaged file fails to load instead of
// crashing a later search.
//
// Parallel build: add_batch(vectors, pool) links the batch on the pool's threads. Its levels are
// drawn in id order before any thread starts, so ids, levels, and the level generator's position
// are exactly those of the sequential add_batch; the neighbor lists depend on thread timing, so the
// graph is not deterministic (the sequential add_batch stays the deterministic default). During
// the build, a node's neighbor lists are read and written only under its lock (a striped table of
// mutexes), and the entry point and top level under one more; see docs/explainers/hnsw.md.
//
// Thread safety: concurrent calls to const methods (search, save, and the accessors) are safe;
// each thread uses its own visited-set scratch buffer. add, add_batch (both forms), and remove are
// single-writer: they require exclusive access, so no other call (including search) may run
// concurrently with them. The parallel add_batch's concurrency is internal to that one call.
class HnswIndex {
 public:
  // Fails if dim == 0, M < 2, or ef_construction == 0.
  [[nodiscard]] static Expected<HnswIndex> create(std::size_t dim, Metric metric,
                                                  HnswParams params = {});

  // Inserts a vector and returns its id (dense, in insertion order). Fails on dimension mismatch.
  Expected<VectorId> add(std::span<const float> vector);
  // Inserts every row in order. Fails (adding nothing) on dimension mismatch.
  Expected<void> add_batch(MatrixView<const float> vectors);
  // Inserts every row using `pool`'s threads (see "Parallel build" above). Ids are assigned in row
  // order, as above. With a pool of one thread the graph is exactly the sequential one.
  Expected<void> add_batch(MatrixView<const float> vectors, ThreadPool& pool);

  // Marks id deleted (a tombstone; see above). Fails with kNotFound if id is out of range or
  // already deleted.
  Expected<void> remove(VectorId id);

  // Approximate k nearest live neighbors. Fails on dimension mismatch. k == 0 or an index with no
  // live nodes gives an empty result.
  [[nodiscard]] Expected<std::vector<Neighbor>> search(std::span<const float> query, std::size_t k,
                                                       std::size_t ef_search) const;

  // k nearest live vectors among those `filter` matches (strategies above). `filter` must cover
  // exactly size() ids (kInvalidArgument otherwise). Results are sorted by (distance, id) with
  // exact distances; the pre-filter returns min(k, matching live) exact results, the graph an
  // approximation of them. `stats`, if given, reports what was done.
  [[nodiscard]] Expected<std::vector<Neighbor>> search_filtered(
      std::span<const float> query, std::size_t k, const CompiledFilter& filter,
      const FilteredSearchOptions& options = {}, FilteredSearchStats* stats = nullptr) const;
  // The same, with the matching ids given as a bitset of size() bits (e.g. a filter evaluated
  // once and reused across queries). Its selectivity is counted exactly.
  [[nodiscard]] Expected<std::vector<Neighbor>> search_filtered(
      std::span<const float> query, std::size_t k, const Bitset& allowed,
      const FilteredSearchOptions& options = {}, FilteredSearchStats* stats = nullptr) const;

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

  // search_filtered for either filter form. Source provides rows(), operator()(id) (matches),
  // matching() (a Bitset of every match), and selectivity(threshold) (see the .cpp).
  template <typename Source>
  [[nodiscard]] Expected<std::vector<Neighbor>> search_filtered_impl(
      std::span<const float> query, std::size_t k, const Source& source,
      const FilteredSearchOptions& options, FilteredSearchStats* stats) const;
  // The pre-filter strategy: the exact k nearest live vectors among the bits set in `allowed`.
  [[nodiscard]] std::vector<Neighbor> prefilter_scan(std::span<const float> query, std::size_t k,
                                                     const Bitset& allowed) const;
  // Checks a batch before anything is added: dimension and id space.
  [[nodiscard]] Expected<void> check_batch(MatrixView<const float> vectors) const;
  // Insertion: append_node then link_node. Precondition: values.size() == dim(), values does not
  // alias this index's storage, and the id space has room.
  VectorId insert(std::span<const float> values);
  // Stores a node: its vector, a freshly drawn level, and empty neighbor lists. Not yet reachable.
  VectorId append_node(std::span<const float> values);
  // Links an appended node into the graph (paper Algorithm 1). With kConcurrent, other threads are
  // linking other nodes at the same time (parallel add_batch): neighbor lists are then read and
  // written only under their node's lock, and the entry point under the top lock.
  template <bool kConcurrent>
  void link_node(VectorId id);
  // floor(-ln(U) * mL), U uniform in (0, 1].
  int random_level();
  // One call of the level generator, counted in rng_draws_.
  std::uint64_t draw();
  void reserve(std::size_t n);
  [[nodiscard]] bool aliases_storage(std::span<const float> values) const noexcept;

  [[nodiscard]] float dist(std::span<const float> query, VectorId id) const noexcept {
    return distance_(query, vector(id));
  }
  // Greedy walk on one layer: move to the closest neighbor until none is closer (ef = 1).
  template <bool kConcurrent>
  [[nodiscard]] Neighbor greedy_search(std::span<const float> query, Neighbor start,
                                       int layer) const;
  // Distance computations allowed to one filtered graph search before it gives up (the auto
  // strategy then switches to the pre-filter). Filled in by search_layer.
  struct TraversalBudget {
    std::size_t max_distances = 0;
    std::size_t distances = 0;
    bool exceeded = false;
  };
  // Beam search on one layer (paper Algorithm 2). Returns up to ef nodes the `keep` policy accepts,
  // sorted ascending; rejected nodes are still expanded (see search_layer and the policies in the
  // .cpp). kConcurrent: during a parallel build (see link_node). A budgeted policy needs `budget`
  // and returns nothing once it is exceeded.
  template <typename Keep, bool kConcurrent = false>
  [[nodiscard]] std::vector<Neighbor> search_layer(std::span<const float> query,
                                                   const std::vector<Neighbor>& entry_points,
                                                   std::size_t ef, int layer, const Keep& keep,
                                                   TraversalBudget* budget = nullptr) const;
  // Shrinks candidates (sorted by distance to a base node, which they must not contain) to at
  // most m neighbors of that node, per params_.selection.
  void select_neighbors(std::vector<Neighbor>& candidates, std::size_t m) const;
  // Parallel build: writes a new node's own links, keeping any back-links other threads added to
  // its (initially empty) list first. Call with id's lock held. `selected` is sorted and consumed.
  void merge_links(VectorId id, int layer, std::vector<Neighbor>& selected);
  // Replaces id's out-links on a layer. Precondition: links.size() <= capacity(layer).
  void set_links(VectorId id, int layer, std::span<const Neighbor> links);
  // Adds the edge from -> to.id, re-selecting from's neighbors if the list overflows.
  // to.distance must be the distance between the two nodes. kConcurrent: under from's lock.
  template <bool kConcurrent>
  void add_link(VectorId from, Neighbor to, int layer);
  // A node's neighbor list on a layer, to traverse. During a parallel build (kConcurrent) another
  // thread may be writing it, so it is copied into `buffer` under the node's lock.
  template <bool kConcurrent>
  [[nodiscard]] std::span<const VectorId> read_links(VectorId id, int layer,
                                                     std::vector<VectorId>& buffer) const;
  [[nodiscard]] std::mutex& link_lock(VectorId id) const noexcept;
  [[nodiscard]] std::size_t capacity(int layer) const noexcept {
    return layer == 0 ? max_links0_ : params_.M;
  }
  // A node's link list on a layer: slot 0 holds the count, slots 1..capacity(layer) the ids.
  [[nodiscard]] VectorId* link_list(VectorId id, int layer) noexcept;
  [[nodiscard]] const VectorId* link_list(VectorId id, int layer) const noexcept;

  std::size_t dim_;
  Metric metric_;
  HnswParams params_;
  DistanceFn distance_;          // best SIMD kernel for the metric, looked up once
  std::size_t max_links0_;       // M_max0 = 2 * M
  double level_mult_;            // mL = 1 / ln(M)
  std::mt19937_64 rng_;          // level generator, seeded from params.seed
  std::uint64_t rng_draws_ = 0;  // calls of rng_ since seeding (what a snapshot saves)

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

  // Locks for the parallel add_batch, created on its first use. Node id i uses stripe
  // i % kLockStripes, so two nodes occasionally share a lock; no thread ever holds two node locks
  // at once, so sharing cannot deadlock. Copying an index copies no locks (none is held between
  // calls, and a copy creates its own when it needs them).
  static constexpr std::size_t kLockStripes = std::size_t{1} << 16;
  struct BuildLocks {
    std::unique_ptr<std::mutex[]> stripes;  // one mutex per stripe of node ids
    std::unique_ptr<std::mutex> top;        // entry point and max level
    BuildLocks() = default;
    BuildLocks(const BuildLocks& /*other*/) {}
    BuildLocks& operator=(const BuildLocks& /*other*/) { return *this; }
    BuildLocks(BuildLocks&&) noexcept = default;
    BuildLocks& operator=(BuildLocks&&) noexcept = default;
    ~BuildLocks() = default;
  };
  BuildLocks locks_;
};

}  // namespace strata
