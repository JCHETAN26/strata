// HNSW (Malkov & Yashunin, 2018). Algorithm numbers below refer to the paper; the long-form
// walkthrough of every function is docs/explainers/hnsw.md.

#include "strata/hnsw.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <locale>
#include <memory>
#include <mutex>
#include <queue>
#include <ranges>
#include <sstream>
#include <string>

#include "strata/filter.hpp"
#include "strata/thread_pool.hpp"
#include "util/bytes.hpp"

namespace strata {

namespace {

tl::unexpected<Error> dimension_error(std::size_t expected, std::size_t got) {
  return make_error(
      ErrorCode::kDimensionMismatch,
      "expected dimension " + std::to_string(expected) + ", got " + std::to_string(got));
}

// Both heaps order by (distance, id), Neighbor's operator<=>. Breaking ties by id makes every
// traversal deterministic, and it is what lets 100 identical vectors still have a well-defined
// "closest 10".
using MaxHeap = std::priority_queue<Neighbor>;  // top = furthest
using MinHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<>>;  // nearest

// Visited marks for one traversal. A node is visited iff marks_[id] == epoch_, so starting a new
// traversal is one increment instead of clearing size() entries. Only a wrap of the 32-bit epoch
// (every ~4 billion traversals) pays for a full clear.
class VisitedSet {
 public:
  void reset(std::size_t n) {
    if (marks_.size() < n) {
      marks_.resize(n, 0);  // 0 never equals a live epoch, so new slots read as unvisited
    }
    if (++epoch_ == 0) {
      std::ranges::fill(marks_, 0);
      epoch_ = 1;
    }
  }
  // Marks id visited; returns true if it was not visited before.
  bool insert(VectorId id) noexcept {
    if (marks_[id] == epoch_) {
      return false;
    }
    marks_[id] = epoch_;
    return true;
  }

 private:
  std::vector<std::uint32_t> marks_;
  std::uint32_t epoch_ = 0;
};

// Per-thread scratch for search_layer: concurrent const searches never share it, so they need no
// lock. The visited marks grow to the largest index the thread has searched (4 bytes per node);
// `unvisited` and `links` each hold one neighbor list's worth of ids (`links` is the copy a
// parallel build takes under a node's lock). All are reused across searches and indexes, so a
// search allocates nothing here after the first.
struct SearchScratch {
  VisitedSet visited;
  std::vector<VectorId> unvisited;
  std::vector<VectorId> links;
};

SearchScratch& thread_scratch() {
  thread_local SearchScratch scratch;
  return scratch;
}

// Asks the CPU to start loading [p, p + bytes) into cache without waiting for it. A hint only:
// it cannot change any result, only when memory arrives. One hint per 64 bytes covers every cache
// line on x86 (64-byte lines); on Apple silicon (128-byte lines) every other hint is redundant
// but cheap.
inline void prefetch(const void* p, std::size_t bytes) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  const auto* bytes_ptr = static_cast<const char*>(p);
  for (std::size_t offset = 0; offset < bytes; offset += 64) {
    __builtin_prefetch(bytes_ptr + offset, /*rw=*/0, /*locality=*/3);
  }
#else
  (void)p;
  (void)bytes;
#endif
}

// Which nodes search_layer may return (keep in W). Every node reached is still expanded, since a
// rejected node can lead to kept ones. kFilters = true switches on the widened stopping rule (see
// search_layer); kBudgeted = true counts distance computations against a TraversalBudget. The
// policies are compile-time types, so an unfiltered search compiles to exactly the plain loop.
struct KeepAll {
  static constexpr bool kFilters = false;
  static constexpr bool kBudgeted = false;
  bool operator()(VectorId /*id*/) const noexcept { return true; }
};

// Tombstones: live nodes only.
struct KeepLive {
  static constexpr bool kFilters = true;
  static constexpr bool kBudgeted = false;
  const std::vector<std::uint8_t>* deleted;
  bool operator()(VectorId id) const noexcept { return (*deleted)[id] == 0; }
};

// Filtered search: live nodes that the filter allows. `allowed` is any bool(VectorId) callable
// (a compiled filter's matches, or a bitset's test); it runs only on nodes the search reaches.
template <typename Allowed>
struct KeepLiveAllowed {
  static constexpr bool kFilters = true;
  static constexpr bool kBudgeted = true;
  const std::vector<std::uint8_t>* deleted;
  const Allowed* allowed;
  bool operator()(VectorId id) const noexcept { return (*deleted)[id] == 0 && (*allowed)(id); }
};

// Filter sources for search_filtered_impl. Each gives the graph strategy a per-id predicate (run
// only on nodes the search reaches), the pre-filter a Bitset of every match, and the auto strategy
// a selectivity estimate.
struct CompiledFilterSource {
  // A 1000-id sample holds about one match at 0.1% selectivity, too few to place a filter near
  // the threshold. So: sample 1000; if that lands within 3 standard errors of the threshold (the
  // error of a 1000-sample estimate at the threshold), sample 20000 more precisely (exact when the
  // index is no larger). Fixed seeds keep the choice deterministic for a given filter.
  static constexpr std::size_t kFirstSample = 1000;
  static constexpr std::size_t kSecondSample = 20000;

  const CompiledFilter* filter;
  [[nodiscard]] std::size_t rows() const noexcept { return filter->rows(); }
  bool operator()(VectorId id) const noexcept { return filter->matches(id); }
  [[nodiscard]] Bitset matching() const { return filter->evaluate(); }
  [[nodiscard]] double selectivity(double threshold, bool& resampled) const {
    const double first = filter->estimate_selectivity(kFirstSample, 1);
    const double error = std::sqrt(threshold * (1 - threshold) / kFirstSample);
    resampled = std::abs(first - threshold) < 3 * error;
    return resampled ? filter->estimate_selectivity(kSecondSample, 2) : first;
  }
};

struct BitsetSource {
  const Bitset* bits;
  [[nodiscard]] std::size_t rows() const noexcept { return bits->size(); }
  bool operator()(VectorId id) const noexcept { return bits->test(id); }
  [[nodiscard]] const Bitset& matching() const noexcept { return *bits; }
  // Counting set bits costs one popcount per 64 ids: exact and cheap.
  [[nodiscard]] double selectivity(double /*threshold*/, bool& resampled) const {
    resampled = false;
    return bits->size() == 0
               ? 0.0
               : static_cast<double>(bits->count()) / static_cast<double>(bits->size());
  }
};

}  // namespace

HnswIndex::HnswIndex(std::size_t dim, Metric metric, HnswParams params)
    : dim_(dim),
      metric_(metric),
      params_(params),
      distance_(distance_function(metric)),
      max_links0_(2 * params.M),
      level_mult_(1.0 / std::log(static_cast<double>(params.M))),
      rng_(params.seed) {}

Expected<HnswIndex> HnswIndex::create(std::size_t dim, Metric metric, HnswParams params) {
  if (dim == 0) {
    return make_error(ErrorCode::kInvalidArgument, "dimension must be positive");
  }
  // mL = 1/ln(M) needs M >= 2 (ln 1 = 0). The upper bound keeps the link count, stored in a
  // VectorId slot, from overflowing.
  if (params.M < 2 || params.M > std::numeric_limits<VectorId>::max() / 4) {
    return make_error(ErrorCode::kInvalidArgument, "M must be at least 2");
  }
  if (params.ef_construction == 0) {
    return make_error(ErrorCode::kInvalidArgument, "ef_construction must be positive");
  }
  return HnswIndex(dim, metric, params);
}

// --- Public API ----------------------------------------------------------------------------------

Expected<VectorId> HnswIndex::add(std::span<const float> vector) {
  if (vector.size() != dim_) {
    return dimension_error(dim_, vector.size());
  }
  if (size() >= std::numeric_limits<VectorId>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "index is full");
  }
  if (aliases_storage(vector)) {
    // e.g. add(index.vector(3)): insert() appends to data_, which may reallocate under the span.
    const std::vector<float> copy(vector.begin(), vector.end());
    return insert(copy);
  }
  return insert(vector);
}

Expected<void> HnswIndex::check_batch(MatrixView<const float> vectors) const {
  if (vectors.cols() != dim_) {
    return dimension_error(dim_, vectors.cols());
  }
  if (size() + vectors.rows() > std::numeric_limits<VectorId>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "batch would overflow the id space");
  }
  return {};
}

Expected<void> HnswIndex::add_batch(MatrixView<const float> vectors) {
  if (vectors.empty()) {
    return {};
  }
  if (auto ok = check_batch(vectors); !ok) {
    return ok;
  }
  // A batch taken from this index's own storage is copied first: insert() appends to data_,
  // which may reallocate under the view.
  Matrix<float> copy;
  if (aliases_storage(vectors.data())) {
    const auto values = vectors.data();
    copy = Matrix<float>(vectors.rows(), vectors.cols(),
                         std::vector<float>(values.begin(), values.end()));
    vectors = copy;
  }
  reserve(size() + vectors.rows());
  for (std::size_t i = 0; i < vectors.rows(); ++i) {
    insert(vectors.row(i));
  }
  return {};
}

Expected<void> HnswIndex::add_batch(MatrixView<const float> vectors, ThreadPool& pool) {
  if (vectors.empty()) {
    return {};
  }
  if (auto ok = check_batch(vectors); !ok) {
    return ok;
  }
  // Appending below copies every row into data_, which may reallocate under a view of it.
  Matrix<float> copy;
  if (aliases_storage(vectors.data())) {
    const auto values = vectors.data();
    copy = Matrix<float>(vectors.rows(), vectors.cols(),
                         std::vector<float>(values.begin(), values.end()));
    vectors = copy;
  }
  // 1. Store every node first: vectors, levels (drawn in id order, exactly as the sequential
  //    add_batch draws them), and empty lists of their final size. From here on no array is
  //    resized, so every node's vector and list stay at a fixed address while threads run.
  const std::size_t first = size();
  reserve(first + vectors.rows());
  for (std::size_t i = 0; i < vectors.rows(); ++i) {
    append_node(vectors.row(i));
  }
  if (!locks_.stripes) {
    locks_.stripes = std::make_unique<std::mutex[]>(kLockStripes);
    locks_.top = std::make_unique<std::mutex>();
  }
  // 2. An empty index gets its first node linked alone: it becomes the entry point.
  std::size_t next = first;
  if (!entry_point_) {
    link_node<false>(static_cast<VectorId>(next++));
  }
  // 3. Link the rest in parallel. Chunks of one id: threads take ids nearly in order, so the
  //    graph grows much as it would sequentially (the order hnswlib's parallel add uses too).
  pool.parallel_for(
      first + vectors.rows() - next,
      [&](std::size_t i) { link_node<true>(static_cast<VectorId>(next + i)); }, 1);
  return {};
}

// Paper Algorithm 5 (K-NN-SEARCH).
Expected<std::vector<Neighbor>> HnswIndex::search(std::span<const float> query, std::size_t k,
                                                  std::size_t ef_search) const {
  if (query.size() != dim_) {
    return dimension_error(dim_, query.size());
  }
  // With no live nodes there is nothing to return; without this check the search below would
  // walk the whole (all-deleted) graph looking for one.
  if (k == 0 || live_size() == 0) {
    return std::vector<Neighbor>{};
  }
  // Upper layers: one greedy walk per layer; the closest node found seeds the layer below. Deleted
  // nodes are fine to walk through: this phase only navigates.
  Neighbor entry{.id = *entry_point_, .distance = dist(query, *entry_point_)};
  for (int layer = max_level_; layer > 0; --layer) {
    entry = greedy_search<false>(query, entry, layer);
  }
  // Layer 0: beam search. The beam can never be narrower than k, or we could not return k. The
  // tombstone-aware version is used only when there are tombstones, so an index without deletes
  // pays nothing for them.
  const std::size_t ef = std::max(ef_search, k);
  auto results = num_deleted_ == 0
                     ? search_layer(query, {entry}, ef, 0, KeepAll{})
                     : search_layer(query, {entry}, ef, 0, KeepLive{.deleted = &deleted_});
  if (results.size() > k) {
    results.resize(k);
  }
  return results;
}

Expected<std::vector<Neighbor>> HnswIndex::search_filtered(std::span<const float> query,
                                                           std::size_t k,
                                                           const CompiledFilter& filter,
                                                           const FilteredSearchOptions& options,
                                                           FilteredSearchStats* stats) const {
  return search_filtered_impl(query, k, CompiledFilterSource{.filter = &filter}, options, stats);
}

Expected<std::vector<Neighbor>> HnswIndex::search_filtered(std::span<const float> query,
                                                           std::size_t k, const Bitset& allowed,
                                                           const FilteredSearchOptions& options,
                                                           FilteredSearchStats* stats) const {
  return search_filtered_impl(query, k, BitsetSource{.bits = &allowed}, options, stats);
}

template <typename Source>
Expected<std::vector<Neighbor>> HnswIndex::search_filtered_impl(
    std::span<const float> query, std::size_t k, const Source& source,
    const FilteredSearchOptions& options, FilteredSearchStats* stats) const {
  if (query.size() != dim_) {
    return dimension_error(dim_, query.size());
  }
  if (source.rows() != size()) {
    return make_error(ErrorCode::kInvalidArgument, "filter covers " +
                                                       std::to_string(source.rows()) +
                                                       " ids, index has " + std::to_string(size()));
  }
  FilteredSearchStats local;
  FilteredSearchStats& st = stats != nullptr ? *stats : local;
  st = {};
  if (k == 0 || live_size() == 0) {
    return std::vector<Neighbor>{};
  }

  FilterStrategy strategy = options.strategy;
  if (strategy == FilterStrategy::kAuto) {
    st.estimated_selectivity = source.selectivity(options.prefilter_below, st.resampled);
    strategy = st.estimated_selectivity < options.prefilter_below ? FilterStrategy::kPreFilter
                                                                  : FilterStrategy::kGraph;
  }

  if (strategy == FilterStrategy::kGraph) {
    // Navigate the upper layers unfiltered (they only find a starting point), then run the
    // layer-0 beam search keeping only matching live nodes: non-matching ones are expanded like
    // tombstones, and the stopping rule waits until W holds ef matches.
    Neighbor entry{.id = *entry_point_, .distance = dist(query, *entry_point_)};
    for (int layer = max_level_; layer > 0; --layer) {
      entry = greedy_search<false>(query, entry, layer);
    }
    const bool may_fall_back =
        options.strategy == FilterStrategy::kAuto && options.fallback_budget > 0;
    // The budget is about what the pre-filter would cost, in distance computations: it tests the
    // filter on every id (fallback_budget, ~0.1 of a distance each) and computes one distance per
    // match (the estimated selectivity).
    const double budget_fraction = options.fallback_budget + st.estimated_selectivity;
    TraversalBudget budget{
        .max_distances =
            may_fall_back ? static_cast<std::size_t>(budget_fraction * static_cast<double>(size()))
                          : std::numeric_limits<std::size_t>::max()};
    const KeepLiveAllowed<Source> keep{.deleted = &deleted_, .allowed = &source};
    auto results = search_layer(query, {entry}, std::max(options.ef_search, k), 0, keep, &budget);
    st.graph_distances = budget.distances;
    if (!budget.exceeded) {
      st.used = FilterStrategy::kGraph;
      if (results.size() > k) {
        results.resize(k);
      }
      return results;
    }
    // Past the budget, the graph has already spent about what the pre-filter costs; a filter this
    // hard (very selective, or matching a region the search has to travel far to reach) is
    // answered exactly instead.
    st.fell_back = true;
  }
  st.used = FilterStrategy::kPreFilter;
  const auto& allowed = source.matching();
  return prefilter_scan(query, k, allowed);
}

std::vector<Neighbor> HnswIndex::prefilter_scan(std::span<const float> query, std::size_t k,
                                                const Bitset& allowed) const {
  MaxHeap best;  // the k nearest so far, furthest on top
  allowed.for_each_set([&](std::size_t i) {
    const auto id = static_cast<VectorId>(i);
    if (deleted_[id] != 0) {
      return;
    }
    const Neighbor n{.id = id, .distance = dist(query, id)};
    if (best.size() < k) {
      best.push(n);
    } else if (n < best.top()) {
      best.pop();
      best.push(n);
    }
  });
  std::vector<Neighbor> sorted(best.size());
  for (Neighbor& slot : std::views::reverse(sorted)) {
    slot = best.top();
    best.pop();
  }
  return sorted;
}

Expected<void> HnswIndex::remove(VectorId id) {
  if (id >= size()) {
    return make_error(ErrorCode::kNotFound, "no vector with id " + std::to_string(id));
  }
  if (deleted_[id] != 0) {
    return make_error(ErrorCode::kNotFound, "vector " + std::to_string(id) + " already deleted");
  }
  // Only the tombstone: the node keeps its links and stays reachable for navigation.
  deleted_[id] = 1;
  ++num_deleted_;
  return {};
}

std::size_t HnswIndex::size() const noexcept { return levels_.size(); }
std::size_t HnswIndex::live_size() const noexcept { return size() - num_deleted_; }

bool HnswIndex::is_deleted(VectorId id) const noexcept {
  assert(id < size());
  return deleted_[id] != 0;
}
std::size_t HnswIndex::dim() const noexcept { return dim_; }
Metric HnswIndex::metric() const noexcept { return metric_; }
const HnswParams& HnswIndex::params() const noexcept { return params_; }

std::span<const float> HnswIndex::vector(VectorId id) const noexcept {
  assert(id < size());
  return {data_.data() + (static_cast<std::size_t>(id) * dim_), dim_};
}

std::optional<VectorId> HnswIndex::entry_point() const noexcept { return entry_point_; }
int HnswIndex::max_level() const noexcept { return max_level_; }

int HnswIndex::level(VectorId id) const noexcept {
  assert(id < size());
  return levels_[id];
}

std::span<const VectorId> HnswIndex::neighbors(VectorId id, int layer) const noexcept {
  const VectorId* list = link_list(id, layer);
  return {list + 1, list[0]};
}

// --- Construction --------------------------------------------------------------------------------

VectorId HnswIndex::insert(std::span<const float> values) {
  const VectorId id = append_node(values);
  link_node<false>(id);
  return id;
}

VectorId HnswIndex::append_node(std::span<const float> values) {
  const auto id = static_cast<VectorId>(size());
  const int level = random_level();
  data_.insert(data_.end(), values.begin(), values.end());
  levels_.push_back(static_cast<std::uint8_t>(level));
  deleted_.push_back(0);
  links0_.resize(links0_.size() + (max_links0_ + 1), 0);
  upper_links_.emplace_back(static_cast<std::size_t>(level) * (params_.M + 1));
  return id;
}

// Paper Algorithm 1 (INSERT), for a node append_node has stored.
template <bool kConcurrent>
void HnswIndex::link_node(VectorId id) {
  const int level = levels_[id];
  const auto query = vector(id);

  // Where to start, and the top layer. In a parallel build these are read under the top lock. An
  // insert that will raise the top level keeps holding it until it is done, so promotions happen
  // one at a time and no insert starts from an entry point whose lists are still being written;
  // every other insert releases it at once. Promotions are rare (probability 1/M per level).
  std::unique_lock<std::mutex> top;
  if constexpr (kConcurrent) {
    top = std::unique_lock(*locks_.top);
  }
  const std::optional<VectorId> entry_point = entry_point_;
  const int max_level = max_level_;
  if constexpr (kConcurrent) {
    if (level <= max_level) {
      top.unlock();
    }
  }
  if (!entry_point) {
    entry_point_ = id;
    max_level_ = level;
    return;
  }

  // Phase 1: above the new node's top layer, only find a good starting point (greedy, ef = 1).
  Neighbor entry{.id = *entry_point, .distance = dist(query, *entry_point)};
  for (int layer = max_level; layer > level; --layer) {
    entry = greedy_search<kConcurrent>(query, entry, layer);
  }

  // Phase 2: on every layer the node lives on, beam-search ef_construction candidates, select up
  // to M of them (select_neighbors), link to them, and link them back. The whole candidate set W
  // seeds the next layer down, as in the paper (hnswlib passes only the closest one).
  std::vector<Neighbor> entry_points{entry};
  for (int layer = std::min(level, max_level); layer >= 0; --layer) {
    // Deleted nodes are still candidates: they stay in the graph for navigation, and keeping
    // them makes the graph independent of which deletes happened.
    auto candidates = search_layer<KeepAll, kConcurrent>(query, entry_points,
                                                         params_.ef_construction, layer, KeepAll{});
    if constexpr (kConcurrent) {
      // In a parallel build this node can already be reachable on this layer (another thread
      // linked to it after finding it on a layer above), so the search may return the node itself.
      // It must not become its own neighbor. (Sequentially a node is unreachable until linked.)
      std::erase_if(candidates, [id](const Neighbor& n) { return n.id == id; });
    }
    auto selected = candidates;
    // M new links on every layer, including layer 0; layer 0's larger capacity (2M) leaves
    // room for the back-links later nodes add.
    select_neighbors(selected, params_.M);
    if constexpr (kConcurrent) {
      // Other threads may already have linked back to this node on this layer (it becomes
      // reachable on the layers above first), so its list may hold their back-links. Overwriting
      // them could leave those nodes with no way in; merge them instead, exactly as if they had
      // arrived after this write (see merge_links). Sequentially the list is always empty here.
      const std::lock_guard lock(link_lock(id));
      merge_links(id, layer, selected);
    } else {
      set_links(id, layer, selected);
    }
    for (const Neighbor& nbr : selected) {
      // Every metric here is symmetric, so d(new, nbr) is also d(nbr, new).
      add_link<kConcurrent>(nbr.id, Neighbor{.id = id, .distance = nbr.distance}, layer);
    }
    entry_points = std::move(candidates);
  }

  // A parallel insert reaching here with a higher level still holds the top lock.
  if (level > max_level) {
    entry_point_ = id;
    max_level_ = level;
  }
}

int HnswIndex::random_level() {
  // U = (r + 1) / 2^53 for the top 53 bits r of one engine draw: uniform on (0, 1], never 0, so
  // ln(U) is finite. Written out instead of std::uniform_real_distribution because the engine's
  // output sequence is fixed by the standard but the distributions are not: libc++ (macOS) and
  // libstdc++ (Linux) would otherwise build different graphs from the same seed.
  // Largest possible level: -ln(2^-53) / ln(2) = 53 (at M = 2), so it fits levels_'s uint8_t.
  const double u = static_cast<double>((draw() >> 11) + 1) * 0x1.0p-53;
  return static_cast<int>(std::floor(-std::log(u) * level_mult_));
}

std::uint64_t HnswIndex::draw() {
  // Every use of the level generator goes through here, so rng_draws_ is exactly the number of
  // engine calls: a snapshot restores the generator by re-seeding and discarding that many.
  ++rng_draws_;
  return rng_();
}

void HnswIndex::reserve(std::size_t n) {
  data_.reserve(n * dim_);
  levels_.reserve(n);
  deleted_.reserve(n);
  links0_.reserve(n * (max_links0_ + 1));
  upper_links_.reserve(n);
}

bool HnswIndex::aliases_storage(std::span<const float> values) const noexcept {
  // std::less gives a total order even for pointers into different objects.
  const std::less<> less;
  const float* begin = data_.data();
  const float* end = begin + data_.size();
  return !values.empty() && less(values.data(), end) && less(begin, values.data() + values.size());
}

// --- Graph traversal -----------------------------------------------------------------------------

template <bool kConcurrent>
Neighbor HnswIndex::greedy_search(std::span<const float> query, Neighbor start, int layer) const {
  // Equivalent to SEARCH-LAYER with ef = 1, minus the heaps and the visited set: the current
  // node strictly improves in (distance, id) order each step, so the walk cannot revisit a node
  // and must terminate.
  auto& buffer = thread_scratch().links;
  Neighbor current = start;
  while (true) {
    Neighbor best = current;
    for (VectorId nbr : read_links<kConcurrent>(current.id, layer, buffer)) {
      const Neighbor candidate{.id = nbr, .distance = dist(query, nbr)};
      if (candidate < best) {
        best = candidate;
      }
    }
    if (best.id == current.id) {
      return current;
    }
    current = best;
  }
}

// Paper Algorithm 2 (SEARCH-LAYER). Nodes the keep policy rejects (deleted, or not matching a
// filter) are expanded, since they still lead to kept ones, but never enter W, so only kept nodes
// are returned. With a budgeted policy, the search stops and reports it once it has computed
// budget->max_distances distances.
template <typename Keep, bool kConcurrent>
std::vector<Neighbor> HnswIndex::search_layer(std::span<const float> query,
                                              const std::vector<Neighbor>& entry_points,
                                              std::size_t ef, int layer, const Keep& keep_node,
                                              TraversalBudget* budget) const {
  SearchScratch& scratch = thread_scratch();
  VisitedSet& visited = scratch.visited;
  visited.reset(size());
  MinHeap candidates;  // C: frontier still to expand, nearest first
  MaxHeap results;     // W: best ef found so far, furthest on top
  const auto keep = [&](const Neighbor& n) {
    if constexpr (Keep::kFilters) {
      if (!keep_node(n.id)) {
        return;
      }
    }
    results.push(n);
    if (results.size() > ef) {
      results.pop();
    }
  };
  for (const Neighbor& entry : entry_points) {
    visited.insert(entry.id);
    candidates.push(entry);
    keep(entry);
  }

  while (!candidates.empty()) {
    const Neighbor nearest = candidates.top();
    // The nearest unexpanded node is further than everything we keep, and expanding only moves
    // outward from it, so nothing left in C can improve W.
    if constexpr (Keep::kFilters) {
      // With rejected nodes (tombstones, filtered-out ids), those sit in C but not in W, so W can
      // be short of ef while C holds nodes beyond W's furthest. Stopping then would return fewer
      // kept results than exist nearby, so the rule applies only once W is full: until then the
      // search keeps widening. (Without rejection the two rules agree: while W is not full, every
      // node in C is also in W, so C's nearest can never lie beyond W's furthest.)
      if (results.size() >= ef && results.top() < nearest) {
        break;
      }
    } else {
      if (results.top() < nearest) {
        break;
      }
    }
    candidates.pop();
    // The next node expanded is probably the new top of C: start loading its neighbor list now,
    // so that load overlaps this expansion. If a closer node is pushed below, the hint is wasted,
    // never wrong. (A prefetch is not a read, so it needs no lock even during a parallel build.)
    if (!candidates.empty()) {
      prefetch(link_list(candidates.top().id, layer), (capacity(layer) + 1) * sizeof(VectorId));
    }
    // Pass 1: mark the unvisited neighbors and ask for all their vectors at once, so the memory
    // loads (one scattered 4*dim-byte vector each, the dominant cost) are in flight together
    // instead of one after another. Pass 2 then computes distances in the same order a single
    // pass would; a neighbor list has no duplicates, so marking them all first changes nothing.
    auto& unvisited = scratch.unvisited;
    unvisited.clear();
    for (VectorId nbr : read_links<kConcurrent>(nearest.id, layer, scratch.links)) {
      if (visited.insert(nbr)) {
        unvisited.push_back(nbr);
        prefetch(vector(nbr).data(), dim_ * sizeof(float));
      }
    }
    if constexpr (Keep::kBudgeted) {
      budget->distances += unvisited.size();
      if (budget->distances > budget->max_distances) {
        budget->exceeded = true;
        return {};
      }
    }
    for (VectorId nbr : unvisited) {
      const Neighbor next{.id = nbr, .distance = dist(query, nbr)};
      // Only nodes that would enter W are worth expanding later. (A rejected node that qualifies
      // is expanded but not kept.)
      if (results.size() < ef || next < results.top()) {
        candidates.push(next);
        keep(next);
      }
    }
  }

  // Drain the max-heap back to front to get ascending order.
  std::vector<Neighbor> sorted(results.size());
  for (Neighbor& slot : std::views::reverse(sorted)) {
    slot = results.top();
    results.pop();
  }
  return sorted;
}

// --- Persistence ---------------------------------------------------------------------------------

namespace {

// Index section versions. 2 (snapshot format version 3) stores the generator as a draw count.
// 1 (snapshot format version 2) stored the generator's stream text, which is not portable (see
// from_snapshot); it is still read, on the standard library that wrote it.
constexpr std::uint32_t kGraphVersion = 2;
constexpr std::uint32_t kGraphVersionTextRng = 1;
constexpr std::uint32_t kNoEntryPoint = 0xFFFFFFFF;

template <typename T>
void put(std::vector<std::byte>& out, const T& value) {
  const auto* p = reinterpret_cast<const std::byte*>(&value);
  out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
void put_all(std::vector<std::byte>& out, std::span<const T> values) {
  const auto bytes = std::as_bytes(values);
  out.insert(out.end(), bytes.begin(), bytes.end());
}

// Reads the index section front to back. Every read is bounds-checked against what is left, so a
// truncated or lying section fails cleanly instead of reading past the buffer.
class SectionReader {
 public:
  explicit SectionReader(std::span<const std::byte> bytes) : rest_(bytes) {}

  template <typename T>
  bool read(T& value) {
    if (rest_.size() < sizeof(T)) {
      return false;
    }
    util::copy_bytes(&value, rest_.data(), sizeof(T));
    rest_ = rest_.subspan(sizeof(T));
    return true;
  }
  // Fills `out` (already sized) from the section.
  template <typename T>
  bool read_all(std::span<T> out) {
    if (rest_.size() / sizeof(T) < out.size()) {
      return false;
    }
    // An empty span (an empty index, a level-0 node's upper lists) may have a null data();
    // copy_bytes makes that zero-length copy safe.
    util::copy_bytes(out.data(), rest_.data(), out.size_bytes());
    rest_ = rest_.subspan(out.size_bytes());
    return true;
  }
  [[nodiscard]] bool done() const noexcept { return rest_.empty(); }

 private:
  std::span<const std::byte> rest_;
};

tl::unexpected<Error> bad_graph(const std::string& what) {
  return make_error(ErrorCode::kCorruptData, "HNSW snapshot: " + what);
}

// Index section version 1 (snapshot format version 2) saved the generator as its stream text. The
// standard fixes that format, but the libraries disagree in practice: libc++ writes the 312 state
// words, libstdc++ writes 313 numbers (its state array plus an internal index), and each rejects
// the other's. So such a file loads only on the standard library that wrote it. Once it is parsed,
// the draw count is recovered by re-seeding and checking that `count` draws (one per node, the only
// way that format was ever written) reproduce the saved state, so the next save is portable.
// Returns the draw count and sets `rng` to the saved state.
Expected<std::uint64_t> read_text_rng(SectionReader& in, std::uint64_t seed, std::size_t count,
                                      std::mt19937_64& rng) {
  std::uint32_t bytes = 0;
  if (!in.read(bytes)) {
    return bad_graph("truncated generator state");
  }
  std::string text(bytes, '\0');
  if (!in.read_all(std::span(text))) {
    return bad_graph("truncated generator state");
  }
  std::istringstream rng_text(text);
  rng_text.imbue(std::locale::classic());
  std::mt19937_64 saved;
  rng_text >> saved;
  if (rng_text.fail() || !(rng_text >> std::ws).eof()) {
    return bad_graph(
        "snapshot format version 2 stores the level generator in a form that depends on the C++ "
        "standard library, and this file was written by a different one (libc++ on macOS vs. "
        "libstdc++ on Linux). Load it on the kind of machine that wrote it and save it again to "
        "upgrade it to the portable format (version 3)");
  }
  std::mt19937_64 probe(seed);
  probe.discard(count);
  if (probe != saved) {
    return bad_graph("generator state does not match the saved seed and node count");
  }
  rng = saved;
  return static_cast<std::uint64_t>(count);
}

}  // namespace

Snapshot HnswIndex::to_snapshot(std::uint64_t last_lsn) const {
  Snapshot snapshot;
  snapshot.metric = metric_;
  snapshot.index = IndexKind::kHnsw;
  snapshot.last_lsn = last_lsn;
  snapshot.vectors = Matrix<float>(size(), dim_, data_);
  snapshot.deleted = deleted_;

  auto& out = snapshot.index_data;
  put(out, kGraphVersion);
  put(out, params_.selection == NeighborSelection::kSimple ? std::uint32_t{0} : std::uint32_t{1});
  put(out, static_cast<std::uint64_t>(params_.M));
  put(out, static_cast<std::uint64_t>(params_.ef_construction));
  put(out, params_.seed);
  put(out, static_cast<std::int32_t>(max_level_));
  put(out, entry_point_.value_or(kNoEntryPoint));
  // The level generator's position, as the number of engine calls since seeding: loading
  // re-seeds and discards that many, which the standard defines identically for every library.
  // (Its stream text is not portable: libc++ and libstdc++ write different formats.)
  put(out, rng_draws_);
  put_all(out, std::span(levels_));
  put_all(out, std::span(links0_));
  for (std::size_t id = 0; id < size(); ++id) {
    put_all(out, std::span(upper_links_[id]));
  }
  return snapshot;
}

Expected<HnswIndex> HnswIndex::from_snapshot(const Snapshot& snapshot) {
  if (snapshot.index != IndexKind::kHnsw) {
    return make_error(ErrorCode::kInvalidArgument, "not an HNSW snapshot");
  }
  const std::size_t count = snapshot.vectors.rows();
  if (snapshot.deleted.size() != count || count > std::numeric_limits<VectorId>::max()) {
    return bad_graph("bad node count");
  }
  SectionReader in(snapshot.index_data);
  std::uint32_t version = 0;
  std::uint32_t selection = 0;
  std::uint64_t m = 0;
  std::uint64_t ef_construction = 0;
  std::uint64_t seed = 0;
  std::int32_t max_level = 0;
  std::uint32_t entry = 0;
  if (!in.read(version) || !in.read(selection) || !in.read(m) || !in.read(ef_construction) ||
      !in.read(seed) || !in.read(max_level) || !in.read(entry)) {
    return bad_graph("truncated header");
  }
  if (version != kGraphVersion && version != kGraphVersionTextRng) {
    return bad_graph("unsupported graph version " + std::to_string(version));
  }
  if (selection > 1) {
    return bad_graph("bad neighbor selection");
  }
  // create() applies the same parameter checks as for a new index.
  auto index = create(
      snapshot.vectors.cols(), snapshot.metric,
      {.M = m,
       .ef_construction = ef_construction,
       .seed = seed,
       .selection = selection == 0 ? NeighborSelection::kSimple : NeighborSelection::kHeuristic});
  if (!index) {
    return bad_graph("bad parameters: " + index.error().message);
  }
  HnswIndex& h = *index;

  // create() seeded the generator from `seed`; move it to where the saved index left it.
  if (version == kGraphVersion) {
    std::uint64_t draws = 0;
    if (!in.read(draws)) {
      return bad_graph("truncated generator state");
    }
    h.rng_.discard(draws);
    h.rng_draws_ = draws;
  } else {
    auto draws = read_text_rng(in, seed, count, h.rng_);
    if (!draws) {
      return tl::unexpected(draws.error());
    }
    h.rng_draws_ = *draws;
  }

  h.levels_.resize(count);
  if (!in.read_all(std::span(h.levels_))) {
    return bad_graph("truncated levels");
  }
  // (2M + 1) is at most ~2^31 (create() capped M), so this only needs a guard against a huge count.
  const std::size_t stride = h.max_links0_ + 1;
  if (count != 0 && stride > std::numeric_limits<std::size_t>::max() / count) {
    return bad_graph("layer-0 size overflows");
  }
  h.links0_.resize(count * stride);
  if (!in.read_all(std::span(h.links0_))) {
    return bad_graph("truncated layer-0 lists");
  }
  h.upper_links_.resize(count);
  for (std::size_t id = 0; id < count; ++id) {
    h.upper_links_[id].resize(static_cast<std::size_t>(h.levels_[id]) * (h.params_.M + 1));
    if (!in.read_all(std::span(h.upper_links_[id]))) {
      return bad_graph("truncated upper-layer lists");
    }
  }
  if (!in.done()) {
    return bad_graph("trailing bytes");
  }

  // Structure: enough to guarantee that search and insert only touch valid memory.
  const int top = count == 0 ? -1 : *std::ranges::max_element(h.levels_);
  if (max_level != top) {
    return bad_graph("max level does not match node levels");
  }
  if (count == 0 ? entry != kNoEntryPoint
                 : entry >= count || static_cast<int>(h.levels_[entry]) != max_level) {
    return bad_graph("entry point is not a node on the top layer");
  }
  for (std::size_t id = 0; id < count; ++id) {
    for (int layer = 0; layer <= h.levels_[id]; ++layer) {
      const VectorId* list = h.link_list(static_cast<VectorId>(id), layer);
      if (list[0] > h.capacity(layer)) {
        return bad_graph("neighbor count over capacity at node " + std::to_string(id));
      }
      for (VectorId i = 1; i <= list[0]; ++i) {
        if (list[i] >= count || h.levels_[list[i]] < layer) {
          return bad_graph("neighbor not on its layer at node " + std::to_string(id));
        }
      }
    }
  }

  const auto values = snapshot.vectors.data();
  h.data_.assign(values.begin(), values.end());
  h.deleted_ = snapshot.deleted;
  h.num_deleted_ = static_cast<std::size_t>(std::ranges::count(h.deleted_, std::uint8_t{1}));
  h.max_level_ = max_level;
  if (count != 0) {
    h.entry_point_ = entry;
  }
  return index;
}

Expected<void> HnswIndex::save(const std::filesystem::path& path) const {
  return write_snapshot(path, to_snapshot());
}

Expected<HnswIndex> HnswIndex::load(const std::filesystem::path& path) {
  auto snapshot = read_snapshot(path);
  if (!snapshot) {
    return tl::unexpected(snapshot.error());
  }
  return from_snapshot(*snapshot);
}

// --- Links ---------------------------------------------------------------------------------------

void HnswIndex::select_neighbors(std::vector<Neighbor>& candidates, std::size_t m) const {
  assert(std::ranges::is_sorted(candidates));
  if (params_.selection == NeighborSelection::kSimple) {
    // SELECT-NEIGHBORS-SIMPLE (Algorithm 3): the m closest. Candidates are sorted: a prefix.
    if (candidates.size() > m) {
      candidates.resize(m);
    }
    return;
  }

  // SELECT-NEIGHBORS-HEURISTIC (Algorithm 4), with extendCandidates and keepPrunedConnections
  // off. Each candidate's distance is to the base node b. Walking nearest first, candidate e is
  // kept only if no already-kept r is strictly closer to e than b is: d(e, r) < d(e, b) means
  // the edge b -> r already leads toward e, so b -> e would be redundant. Ties keep e, so exact
  // duplicates (all distances 0) still link to each other instead of collapsing to one link.
  // Applied even when there are <= m candidates, as in the paper (hnswlib skips that case).
  // Kept candidates are compacted to the front in place (kept <= i, so nothing is overwritten
  // before it is read).
  std::size_t kept = 0;
  for (std::size_t i = 0; i < candidates.size() && kept < m; ++i) {
    const Neighbor candidate = candidates[i];
    const auto values = vector(candidate.id);
    bool diverse = true;
    for (std::size_t j = 0; j < kept; ++j) {
      if (dist(values, candidates[j].id) < candidate.distance) {
        diverse = false;
        break;
      }
    }
    if (diverse) {
      candidates[kept++] = candidate;
    }
  }
  candidates.resize(kept);
}

void HnswIndex::set_links(VectorId id, int layer, std::span<const Neighbor> links) {
  assert(links.size() <= capacity(layer));
  VectorId* list = link_list(id, layer);
  list[0] = static_cast<VectorId>(links.size());
  for (std::size_t i = 0; i < links.size(); ++i) {
    list[1 + i] = links[i].id;
  }
}

void HnswIndex::merge_links(VectorId id, int layer, std::vector<Neighbor>& selected) {
  VectorId* list = link_list(id, layer);
  const std::size_t count = list[0];
  if (count == 0) {
    set_links(id, layer, selected);
    return;
  }
  // Back-links other threads added before this node wrote its own list. They are distinct from
  // each other; drop any that are also in `selected`, then keep everything if it fits (as appending
  // them afterwards would), else re-select from this node's point of view (as an overflowing
  // append would).
  const auto base = vector(id);
  for (std::size_t i = 0; i < count; ++i) {
    const VectorId other = list[1 + i];
    if (std::ranges::none_of(selected, [&](const Neighbor& n) { return n.id == other; })) {
      selected.push_back({.id = other, .distance = dist(base, other)});
    }
  }
  std::ranges::sort(selected);
  if (selected.size() > capacity(layer)) {
    select_neighbors(selected, capacity(layer));
  }
  set_links(id, layer, selected);
}

template <bool kConcurrent>
void HnswIndex::add_link(VectorId from, Neighbor to, int layer) {
  // In a parallel build, from's lists change only under from's lock. The re-selection below also
  // runs under it: it reads vectors, which never change during a build, so it needs no other lock.
  std::unique_lock<std::mutex> lock;
  if constexpr (kConcurrent) {
    lock = std::unique_lock(link_lock(from));
  }
  VectorId* list = link_list(from, layer);
  const std::size_t count = list[0];
  if constexpr (kConcurrent) {
    // Two nodes inserted at the same time can select each other: `from` may already have linked to
    // `to` itself, so the back-link would be a duplicate. (Sequentially this cannot happen: an
    // earlier node never selects a later one.)
    if (std::find(list + 1, list + 1 + count, to.id) != list + 1 + count) {
      return;
    }
  }
  if (count < capacity(layer)) {
    list[1 + count] = to.id;
    list[0] = static_cast<VectorId>(count + 1);
    return;
  }
  // Overflow (paper Algorithm 1, lines 13-16): re-select from's neighbors, from its own point of
  // view, among its current links plus the new one. The new link may itself be dropped.
  const auto base = vector(from);
  std::vector<Neighbor> candidates;
  candidates.reserve(count + 1);
  for (std::size_t i = 0; i < count; ++i) {
    candidates.push_back({.id = list[1 + i], .distance = dist(base, list[1 + i])});
  }
  candidates.push_back(to);
  std::ranges::sort(candidates);
  select_neighbors(candidates, capacity(layer));
  set_links(from, layer, candidates);
}

template <bool kConcurrent>
std::span<const VectorId> HnswIndex::read_links(VectorId id, int layer,
                                                std::vector<VectorId>& buffer) const {
  if constexpr (kConcurrent) {
    const std::lock_guard lock(link_lock(id));
    const auto list = neighbors(id, layer);
    buffer.assign(list.begin(), list.end());
    return buffer;
  } else {
    (void)buffer;
    return neighbors(id, layer);
  }
}

std::mutex& HnswIndex::link_lock(VectorId id) const noexcept {
  return locks_.stripes[id % kLockStripes];
}

VectorId* HnswIndex::link_list(VectorId id, int layer) noexcept {
  assert(id < size() && layer >= 0 && layer <= level(id));
  if (layer == 0) {
    return links0_.data() + (static_cast<std::size_t>(id) * (max_links0_ + 1));
  }
  return upper_links_[id].data() + (static_cast<std::size_t>(layer - 1) * (params_.M + 1));
}

const VectorId* HnswIndex::link_list(VectorId id, int layer) const noexcept {
  assert(id < size() && layer >= 0 && layer <= level(id));
  if (layer == 0) {
    return links0_.data() + (static_cast<std::size_t>(id) * (max_links0_ + 1));
  }
  return upper_links_[id].data() + (static_cast<std::size_t>(layer - 1) * (params_.M + 1));
}

}  // namespace strata
