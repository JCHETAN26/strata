// HNSW (Malkov & Yashunin, 2018). Algorithm numbers below refer to the paper; the long-form
// walkthrough of every function is docs/explainers/hnsw.md.

#include "strata/hnsw.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <string>

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

// One buffer per thread: concurrent const searches never share scratch state, so they need no
// lock. The buffer grows to the largest index the thread has searched (4 bytes per node) and is
// reused across searches and across indexes.
VisitedSet& thread_visited() {
  thread_local VisitedSet visited;
  return visited;
}

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

Expected<void> HnswIndex::add_batch(MatrixView<const float> vectors) {
  if (vectors.empty()) {
    return {};
  }
  if (vectors.cols() != dim_) {
    return dimension_error(dim_, vectors.cols());
  }
  if (size() + vectors.rows() > std::numeric_limits<VectorId>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "batch would overflow the id space");
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

// Paper Algorithm 5 (K-NN-SEARCH).
Expected<std::vector<Neighbor>> HnswIndex::search(std::span<const float> query, std::size_t k,
                                                  std::size_t ef_search) const {
  if (query.size() != dim_) {
    return dimension_error(dim_, query.size());
  }
  if (k == 0 || !entry_point_) {
    return std::vector<Neighbor>{};
  }
  // Upper layers: one greedy walk per layer; the closest node found seeds the layer below.
  Neighbor entry{.id = *entry_point_, .distance = dist(query, *entry_point_)};
  for (int layer = max_level_; layer > 0; --layer) {
    entry = greedy_search(query, entry, layer);
  }
  // Layer 0: beam search. The beam can never be narrower than k, or we could not return k.
  auto results = search_layer(query, {entry}, std::max(ef_search, k), 0);
  if (results.size() > k) {
    results.resize(k);
  }
  return results;
}

std::size_t HnswIndex::size() const noexcept { return levels_.size(); }
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

// Paper Algorithm 1 (INSERT).
VectorId HnswIndex::insert(std::span<const float> values) {
  const auto id = static_cast<VectorId>(size());
  const int level = random_level();

  data_.insert(data_.end(), values.begin(), values.end());
  levels_.push_back(static_cast<std::uint8_t>(level));
  links0_.resize(links0_.size() + (max_links0_ + 1), 0);
  upper_links_.emplace_back(static_cast<std::size_t>(level) * (params_.M + 1));
  const auto query = vector(id);

  if (!entry_point_) {
    entry_point_ = id;
    max_level_ = level;
    return id;
  }

  // Phase 1: above the new node's top layer, only find a good starting point (greedy, ef = 1).
  Neighbor entry{.id = *entry_point_, .distance = dist(query, *entry_point_)};
  for (int layer = max_level_; layer > level; --layer) {
    entry = greedy_search(query, entry, layer);
  }

  // Phase 2: on every layer the node lives on, beam-search ef_construction candidates, link to
  // the best M of them, and link them back. The whole candidate set W seeds the next layer down,
  // as in the paper (hnswlib passes only the closest one).
  std::vector<Neighbor> entry_points{entry};
  for (int layer = std::min(level, max_level_); layer >= 0; --layer) {
    auto candidates = search_layer(query, entry_points, params_.ef_construction, layer);
    auto selected = candidates;
    // M new links on every layer, including layer 0; layer 0's larger capacity (2M) leaves
    // room for the back-links later nodes add.
    select_neighbors(selected, params_.M);
    set_links(id, layer, selected);
    for (const Neighbor& nbr : selected) {
      // Every metric here is symmetric, so d(new, nbr) is also d(nbr, new).
      add_link(nbr.id, Neighbor{.id = id, .distance = nbr.distance}, layer);
    }
    entry_points = std::move(candidates);
  }

  if (level > max_level_) {
    entry_point_ = id;
    max_level_ = level;
  }
  return id;
}

int HnswIndex::random_level() {
  // U = (r + 1) / 2^53 for the top 53 bits r of one engine draw: uniform on (0, 1], never 0, so
  // ln(U) is finite. Written out instead of std::uniform_real_distribution because the engine's
  // output sequence is fixed by the standard but the distributions are not: libc++ (macOS) and
  // libstdc++ (Linux) would otherwise build different graphs from the same seed.
  // Largest possible level: -ln(2^-53) / ln(2) = 53 (at M = 2), so it fits levels_'s uint8_t.
  const double u = static_cast<double>((rng_() >> 11) + 1) * 0x1.0p-53;
  return static_cast<int>(std::floor(-std::log(u) * level_mult_));
}

void HnswIndex::reserve(std::size_t n) {
  data_.reserve(n * dim_);
  levels_.reserve(n);
  links0_.reserve(n * (max_links0_ + 1));
  upper_links_.reserve(n);
}

bool HnswIndex::aliases_storage(std::span<const float> values) const noexcept {
  // std::less gives a total order even for pointers into different objects.
  const std::less<const float*> less;
  const float* begin = data_.data();
  const float* end = begin + data_.size();
  return !values.empty() && less(values.data(), end) && less(begin, values.data() + values.size());
}

// --- Graph traversal -----------------------------------------------------------------------------

Neighbor HnswIndex::greedy_search(std::span<const float> query, Neighbor start, int layer) const {
  // Equivalent to SEARCH-LAYER with ef = 1, minus the heaps and the visited set: the current
  // node strictly improves in (distance, id) order each step, so the walk cannot revisit a node
  // and must terminate.
  Neighbor current = start;
  while (true) {
    Neighbor best = current;
    for (VectorId nbr : neighbors(current.id, layer)) {
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

// Paper Algorithm 2 (SEARCH-LAYER).
std::vector<Neighbor> HnswIndex::search_layer(std::span<const float> query,
                                              const std::vector<Neighbor>& entry_points,
                                              std::size_t ef, int layer) const {
  VisitedSet& visited = thread_visited();
  visited.reset(size());
  MinHeap candidates;  // C: frontier still to expand, nearest first
  MaxHeap results;     // W: best ef found so far, furthest on top
  for (const Neighbor& entry : entry_points) {
    visited.insert(entry.id);
    candidates.push(entry);
    results.push(entry);
    if (results.size() > ef) {
      results.pop();
    }
  }

  while (!candidates.empty()) {
    const Neighbor nearest = candidates.top();
    // The nearest unexpanded node is further than everything we keep, and expanding only moves
    // outward from it, so nothing left in C can improve W.
    if (results.top() < nearest) {
      break;
    }
    candidates.pop();
    for (VectorId nbr : neighbors(nearest.id, layer)) {
      if (!visited.insert(nbr)) {
        continue;
      }
      const Neighbor next{.id = nbr, .distance = dist(query, nbr)};
      // Only nodes that would enter W are worth expanding later.
      if (results.size() < ef || next < results.top()) {
        candidates.push(next);
        results.push(next);
        if (results.size() > ef) {
          results.pop();
        }
      }
    }
  }

  // Drain the max-heap back to front to get ascending order.
  std::vector<Neighbor> sorted(results.size());
  for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
    *it = results.top();
    results.pop();
  }
  return sorted;
}

// --- Links ---------------------------------------------------------------------------------------

void HnswIndex::select_neighbors(std::vector<Neighbor>& candidates, std::size_t m) {
  // SELECT-NEIGHBORS-SIMPLE: the m closest. Candidates arrive sorted, so that is a prefix.
  assert(std::ranges::is_sorted(candidates));
  if (candidates.size() > m) {
    candidates.resize(m);
  }
}

void HnswIndex::set_links(VectorId id, int layer, std::span<const Neighbor> links) {
  assert(links.size() <= capacity(layer));
  VectorId* list = link_list(id, layer);
  list[0] = static_cast<VectorId>(links.size());
  for (std::size_t i = 0; i < links.size(); ++i) {
    list[1 + i] = links[i].id;
  }
}

void HnswIndex::add_link(VectorId from, Neighbor to, int layer) {
  VectorId* list = link_list(from, layer);
  const std::size_t count = list[0];
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
