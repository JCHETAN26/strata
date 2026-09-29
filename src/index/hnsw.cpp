// HNSW (Malkov & Yashunin, 2018). Algorithm numbers below refer to the paper; the long-form
// walkthrough of every function is docs/explainers/hnsw.md.

#include "strata/hnsw.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <locale>
#include <queue>
#include <ranges>
#include <sstream>
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

// Per-thread scratch for search_layer: concurrent const searches never share it, so they need no
// lock. The visited marks grow to the largest index the thread has searched (4 bytes per node);
// `unvisited` holds one neighbor list's worth of ids. Both are reused across searches and
// indexes, so a search allocates nothing here after the first.
struct SearchScratch {
  VisitedSet visited;
  std::vector<VectorId> unvisited;
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
  // With no live nodes there is nothing to return; without this check the search below would
  // walk the whole (all-deleted) graph looking for one.
  if (k == 0 || live_size() == 0) {
    return std::vector<Neighbor>{};
  }
  // Upper layers: one greedy walk per layer; the closest node found seeds the layer below. Deleted
  // nodes are fine to walk through: this phase only navigates.
  Neighbor entry{.id = *entry_point_, .distance = dist(query, *entry_point_)};
  for (int layer = max_level_; layer > 0; --layer) {
    entry = greedy_search(query, entry, layer);
  }
  // Layer 0: beam search. The beam can never be narrower than k, or we could not return k. The
  // tombstone-aware version is used only when there are tombstones, so an index without deletes
  // pays nothing for them.
  const std::size_t ef = std::max(ef_search, k);
  auto results = num_deleted_ == 0 ? search_layer<false>(query, {entry}, ef, 0)
                                   : search_layer<true>(query, {entry}, ef, 0);
  if (results.size() > k) {
    results.resize(k);
  }
  return results;
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

// Paper Algorithm 1 (INSERT).
VectorId HnswIndex::insert(std::span<const float> values) {
  const auto id = static_cast<VectorId>(size());
  const int level = random_level();

  data_.insert(data_.end(), values.begin(), values.end());
  levels_.push_back(static_cast<std::uint8_t>(level));
  deleted_.push_back(0);
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

  // Phase 2: on every layer the node lives on, beam-search ef_construction candidates, select up
  // to M of them (select_neighbors), link to them, and link them back. The whole candidate set W
  // seeds the next layer down, as in the paper (hnswlib passes only the closest one).
  std::vector<Neighbor> entry_points{entry};
  for (int layer = std::min(level, max_level_); layer >= 0; --layer) {
    // Deleted nodes are still candidates: they stay in the graph for navigation, and keeping
    // them makes the graph independent of which deletes happened.
    auto candidates = search_layer<false>(query, entry_points, params_.ef_construction, layer);
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

// Paper Algorithm 2 (SEARCH-LAYER). With kSkipDeleted, deleted nodes are expanded (they still
// lead to live ones) but never enter W, so only live nodes are returned.
template <bool kSkipDeleted>
std::vector<Neighbor> HnswIndex::search_layer(std::span<const float> query,
                                              const std::vector<Neighbor>& entry_points,
                                              std::size_t ef, int layer) const {
  SearchScratch& scratch = thread_scratch();
  VisitedSet& visited = scratch.visited;
  visited.reset(size());
  MinHeap candidates;  // C: frontier still to expand, nearest first
  MaxHeap results;     // W: best ef found so far, furthest on top
  const auto keep = [&](const Neighbor& n) {
    if constexpr (kSkipDeleted) {
      if (deleted_[n.id] != 0) {
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
    if constexpr (kSkipDeleted) {
      // With tombstones, deleted nodes sit in C but not in W, so W can be short of ef while C
      // holds nodes beyond W's furthest. Stopping then would return fewer live results than
      // exist nearby, so the rule applies only once W is full: until then the search keeps
      // widening. (Without tombstones the two rules agree: while W is not full, every node in C is
      // also in W, so C's nearest can never lie beyond W's furthest.)
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
    // never wrong.
    if (!candidates.empty()) {
      prefetch(link_list(candidates.top().id, layer), (capacity(layer) + 1) * sizeof(VectorId));
    }
    // Pass 1: mark the unvisited neighbors and ask for all their vectors at once, so the memory
    // loads (one scattered 4*dim-byte vector each, the dominant cost) are in flight together
    // instead of one after another. Pass 2 then computes distances in the same order a single
    // pass would; a neighbor list has no duplicates, so marking them all first changes nothing.
    auto& unvisited = scratch.unvisited;
    unvisited.clear();
    for (VectorId nbr : neighbors(nearest.id, layer)) {
      if (visited.insert(nbr)) {
        unvisited.push_back(nbr);
        prefetch(vector(nbr).data(), dim_ * sizeof(float));
      }
    }
    for (VectorId nbr : unvisited) {
      const Neighbor next{.id = nbr, .distance = dist(query, nbr)};
      // Only nodes that would enter W are worth expanding later. (A deleted node that qualifies
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

constexpr std::uint32_t kGraphVersion = 1;
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
    std::memcpy(&value, rest_.data(), sizeof(T));
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
    // memcpy with a null pointer is undefined even for a zero byte count, so skip the copy.
    // glibc declares memcpy nonnull, so UBSan reports this on Linux; macOS's libc does not.
    if (out.empty()) {
      return true;
    }
    std::memcpy(out.data(), rest_.data(), out.size_bytes());
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
  // The level generator's exact position, so the levels drawn after a load continue the same
  // sequence. The text form is specified by the standard; the classic locale keeps it free of
  // digit grouping whatever the process locale is.
  std::ostringstream rng_text;
  rng_text.imbue(std::locale::classic());
  rng_text << rng_;
  const std::string text = rng_text.str();
  put(out, static_cast<std::uint32_t>(text.size()));
  put_all(out, std::span(text));
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
  std::uint32_t rng_bytes = 0;
  if (!in.read(version) || !in.read(selection) || !in.read(m) || !in.read(ef_construction) ||
      !in.read(seed) || !in.read(max_level) || !in.read(entry) || !in.read(rng_bytes)) {
    return bad_graph("truncated header");
  }
  if (version != kGraphVersion) {
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

  std::string text(rng_bytes, '\0');
  if (!in.read_all(std::span(text))) {
    return bad_graph("truncated generator state");
  }
  std::istringstream rng_text(text);
  rng_text.imbue(std::locale::classic());
  rng_text >> h.rng_;
  if (rng_text.fail() || !(rng_text >> std::ws).eof()) {
    return bad_graph("bad generator state");
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
