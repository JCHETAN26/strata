# HNSW in Strata: how it works and why every line is there

This explains `src/index/hnsw.cpp` and `include/strata/hnsw.hpp` well enough to defend them
line by line. It follows Malkov & Yashunin, *Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs* (TPAMI 2018). "Algorithm N" always means
the paper's pseudocode.

Status: **stage (a)**: levels, layer search, and insertion with the *simple* neighbor selection
(closest M). The paper's heuristic (stage b) and the measured recall-vs-QPS curves (stage c) are
marked TODO below and will be filled in with numbers from `bench/`.

---

## 1. The intuition

**The problem.** Given N vectors and a query, find the k closest. Brute force computes N
distances. We want roughly log N, and we will accept missing a few true neighbors (that is what
recall measures).

**A proximity graph.** Connect each vector to some of its near neighbors. To search, start
anywhere and walk greedily: move to whichever neighbor is closer to the query, stop when none is.
If the graph links every node to its true nearest neighbors, the walk tends to reach the query's
neighborhood. Two problems:

1. **Long trips.** Starting far away, a walk over short edges takes many steps (about N^(1/d)
   on a grid-like graph).
2. **Local minima.** The walk stops at a node none of whose neighbors is closer, even if a
   better node exists elsewhere.

**Navigable small world (NSW).** Mix short edges with some long ones, and greedy search takes a
few long hops to get close, then short hops to refine. In NSW the long edges appear for free: the
nodes inserted early link to each other while the graph is still sparse, so their edges span the
whole space.

**Hierarchy (the H in HNSW).** Make the long-vs-short split explicit, like a skip list:

- Every node lives on layer 0. A node lives on layer 1 with probability 1/M, on layer 2 with
  probability 1/M^2, and so on.
- Each layer is an NSW graph over only the nodes on it. Upper layers are sparse, so their edges
  are long; layer 0 is dense, so its edges are short.
- Search starts at the single top-layer entry point, greedily walks the sparse top layer to the
  closest node there, drops down a layer from that node, and repeats. By layer 0 it starts right
  next to the answer and only needs a local, careful search.

Each layer shrinks the node count by a factor of M, so there are about log_M(N) layers, and a
greedy walk on each takes a roughly constant number of hops. That is where the logarithmic search
cost comes from.

---

## 2. Data layout

Everything is indexed by the dense node id (insertion order), so each field is a flat array:

| Member | Contents | Why |
|---|---|---|
| `data_` | `size() * dim` floats, row-major | One contiguous block: `vector(id)` is a pointer offset, and the SIMD kernels read it linearly. |
| `levels_` | one `uint8_t` per node | The top layer of every node. The maximum possible level is 53 (see `random_level`), so a byte is enough. |
| `links0_` | `(2M + 1)` ids per node | Layer-0 neighbor lists at a **fixed stride**: node `id`'s list is at `id * (2M+1)`. No pointer chase and no allocation per node. Slot 0 is the count. |
| `upper_links_` | per node, `level * (M + 1)` ids | Layers >= 1. Only about 1/(M-1) of the total link storage lives here, so a separate small vector per node is fine. Block `layer - 1` is that layer's list, count first. |
| `entry_point_`, `max_level_` | the top-layer node and its level | `nullopt` / `-1` when empty. |
| `rng_` | `std::mt19937_64` seeded from `params.seed` | Deterministic level assignment. |
| `distance_` | a function pointer | `distance_function(metric)` looks up the best compiled kernel (NEON on the M2, AVX2 on the Ryzen) once, in the constructor. |

Layer 0 holds every node and is where search spends most of its time, so it gets the
cache-friendly fixed-stride layout. This is the same layout hnswlib uses.

**Memory per node:** `4·dim` (vector) + `4·(2M+1)` (layer 0) + about `4·(M+1)/(M-1)` (upper
layers, in expectation) + 24 (the `std::vector` header in `upper_links_`) + 1 (level).
For SIFT (dim 128, M 16): 512 + 132 + 4 + 24 + 1 ≈ 673 bytes, so ≈ 0.67 GB for SIFT1M. The
24-byte vector header is the one wasteful item; a single flat array with offsets would save about
16 bytes per node. That is not worth the complexity yet.

---

## 3. Walkthrough, function by function

### `create(dim, metric, params)`

Validates, then constructs. It rejects:
- `dim == 0`;
- `M < 2`, because mL = 1/ln(M) divides by zero at M = 1. The upper bound (`VectorId` max / 4)
  keeps `2M` counts from overflowing their 32-bit slot;
- `ef_construction == 0`, because a zero-width beam finds nothing.

The constructor precomputes `max_links0_ = 2M` (the paper's M_max0) and `level_mult_ = 1/ln(M)`.

### `random_level()`: the level distribution

```cpp
const double u = static_cast<double>((rng_() >> 11) + 1) * 0x1.0p-53;
return static_cast<int>(std::floor(-std::log(u) * level_mult_));
```

- **Why `floor(-ln(U) · mL)`?** If U is uniform on (0, 1], then −ln(U) is exponential with rate
  1, so P(level >= l) = P(−ln U >= l/mL) = e^(−l/mL). With mL = 1/ln(M) that is
  e^(−l·ln M) = M^(−l). So a node reaches layer 1 with probability 1/M, layer 2 with 1/M^2, and
  so on. This is a geometric distribution, like the levels of a skip list.
- **Why mL = 1/ln(M)?** The paper shows this choice makes layers overlap as little as possible:
  each layer has about 1/M as many nodes as the one below, which matches the M links each node
  keeps per layer. Search then does about the same work on every layer. Larger mL gives more
  layers and more overlap (wasted work); mL → 0 gives a single layer, which is plain NSW.
- **Why build U by hand?** `(r >> 11)` keeps the top 53 bits of a 64-bit draw, which is exactly
  a double's precision. Adding 1 and scaling by 2^−53 gives values in (0, 1]; zero is impossible,
  so `log` never returns −∞. `std::uniform_real_distribution` would be shorter, but the C++
  standard fixes the *engine's* output sequence and not the *distribution's* algorithm: libc++
  (macOS) and libstdc++ (Linux) would turn the same seed into different levels and therefore
  different graphs. Building U by hand keeps Mac and Ryzen results comparable. (`std::log` is
  not required to be correctly rounded, so the platforms could in principle disagree when
  −ln(U)·mL lands within an ulp of an integer. That is vanishingly rare.)
- **Largest level:** the smallest U is 2^−53, so the maximum is 53·ln 2 / ln M, which is at most
  53 (at M = 2). That is why a `uint8_t` per node is enough.

The `LevelDistributionMatchesMl` test checks that about 1/M of 5000 nodes land on layer >= 1.

### `greedy_search(query, start, layer)`: the upper-layer walk

Look at every neighbor of the current node and move to the best one if it beats the current
node; stop when none does. This is `SEARCH-LAYER` with ef = 1 (the paper's descent phase uses
exactly that), written without heaps or a visited set:

- **No visited set needed:** each move strictly decreases the current node in (distance, id)
  order, so a node can never be revisited, and the walk ends because the graph is finite.
- **Ties broken by id:** `Neighbor`'s ordering compares distance, then id. So "closer" is a
  strict total order even among equal distances, and the walk is deterministic.

### `search_layer(query, entry_points, ef, layer)`: the beam search (Algorithm 2)

The heart of HNSW. Two heaps:

- **C** (`candidates`, a min-heap) holds discovered nodes whose neighbors we have not looked at
  yet, nearest first.
- **W** (`results`, a max-heap capped at `ef`) holds the best `ef` nodes found so far, with the
  furthest on top so it is cheap to evict.

```
seed C and W with the entry points, mark them visited
loop:
  c ← nearest in C
  if c is further than the furthest in W: stop            (1)
  pop c; for each neighbor e of c on this layer:
    if e unvisited: mark it
      if W has room or e beats W's furthest:               (2)
        push e into C and W; if |W| > ef, evict W's furthest
return W sorted ascending
```

- **(1) Stopping rule.** Every unexpanded node is at least as far as `c`, and `c` is already
  worse than all `ef` results we hold. Expanding moves outward through the graph, so no remaining
  candidate is likely to produce something better. This is where "approximate" comes from: the
  rule assumes the graph is navigable. Larger `ef` delays the stop and explores more.
- **(2) Only promising nodes enter C.** A node that could not enter W is never expanded. That
  keeps C small and focused.
- **Visited set.** Each node's distance is computed at most once per call. The set is a
  per-thread array of 32-bit "epochs": node `id` counts as visited iff `marks[id] == epoch`.
  Starting a new search is a single `++epoch`, O(1), instead of clearing N flags. On the rare
  32-bit wraparound we clear everything once.
- **Why `thread_local`?** It is scratch state, and giving each thread its own copy is what makes
  concurrent `const` searches race-free without locks. The cost is 4 bytes × (largest index
  searched) per thread, kept for the thread's lifetime. The `ConcurrentSearchesMatchSerial` test
  runs 4 threads under ThreadSanitizer.
- **Entry points are a set.** Search passes one node; insertion passes all of W from the layer
  above (see `insert`).
- **Result order.** Draining a max-heap yields descending order, so we fill the output vector back
  to front and it comes out ascending by (distance, id), as the API promises.

### `search(query, k, ef_search)`: K-NN search (Algorithm 5)

1. Validate the dimension. Return empty for `k == 0` or an empty index.
2. Start at the entry point and run `greedy_search` from the top layer down to layer 1, each
   layer starting from the previous layer's result. This is the "zoom in" phase.
3. On layer 0, run `search_layer` with beam width `max(ef_search, k)`. The `max` exists because a
   beam narrower than k could not return k results (`EfSmallerThanKStillReturnsK`).
4. Truncate to k.

The returned distances are the exact distances computed during the walk, so they match brute
force bit for bit (`ResultsAreSortedDistinctAndExact`).

### `insert(values)`: INSERT (Algorithm 1)

1. **Allocate.** Draw the level `l`, append the vector, the level, a zeroed layer-0 list, and `l`
   zeroed upper-layer lists. `query` is a span into `data_`; nothing below reallocates `data_`, so
   it stays valid.
2. **First node.** It becomes the entry point, and we are done.
3. **Phase 1: descend.** From the top layer L down to `l + 1`, the new node has no links, so we
   only need a good starting point: `greedy_search`, as in search.
4. **Phase 2: link.** For each layer `lc` from `min(l, L)` down to 0:
   - `search_layer` with beam `ef_construction` finds candidates W. This is the same search a
     query does, just wider: a better candidate list gives better neighbors, so
     `ef_construction` trades build time for graph quality.
   - `select_neighbors` picks M of them, and `set_links` writes them as the new node's out-links.
     The new node gets **M** links on every layer, including layer 0 (as in the paper and
     hnswlib), while layer 0's *capacity* is **2M**. The spare room absorbs back-links from
     later insertions without immediately forcing a prune.
   - `add_link(neighbor → new)` for each selected neighbor, so edges go both ways. Without
     back-links, old nodes would never point at new ones, and new nodes would be unreachable.
   - **All of W**, not just its best element, seeds the next layer down (`ep ← W` in the
     paper). hnswlib passes only the closest one; the paper's version gives the lower layer's
     search a wider start.
5. **New top.** If `l > L`, the new node becomes the entry point. Only nodes on the top layer can
   be the entry point, because search starts there (`EntryPointIsOnTopLayer`).

Distances are only computed between the new node and existing ones. Every metric Strata supports
(L2², negative inner product, cosine distance) is symmetric, so the distance `nbr.distance` found
from the new node's side is reused as the back-link's distance.

### `add_link(from, to, layer)`: back-links and pruning

If `from`'s list has room, append. Otherwise we **re-select** (Algorithm 1, lines 13–16):
compute the distance from `from` to each current neighbor, add the new candidate, sort, and run
`select_neighbors` with the layer's capacity (2M on layer 0, M above). The new link can lose, in
which case the edge exists only in the new → old direction.

This keeps degrees bounded (`DegreeBounds`), and therefore memory and per-hop search cost too.
The price: edges are not guaranteed to be bidirectional, and a node can in principle lose every
in-link and become unreachable. `Layer0IsReachableFromEntryPoint` measures that. How pruning
chooses which links to keep is exactly where the heuristic (section 4) matters most.

### `select_neighbors(candidates, m)`: stage (a), SELECT-NEIGHBORS-SIMPLE (Algorithm 3)

Keep the m closest. Candidates always arrive sorted, so this is a truncation.

### `add`, `add_batch`: API boundary

- Both validate the dimension and the 32-bit id space **before** changing anything, so a failed
  batch adds nothing (`DimensionMismatchIsAnError`).
- **Aliasing.** `add(index.vector(3))` passes a span into `data_`, and appending to `data_` may
  reallocate it, which would leave the span dangling mid-insert. `aliases_storage` detects that
  (using `std::less`, which is defined for any two pointers), and we copy first. Normal calls pay
  one pointer comparison.
- `add_batch` reserves every array once, then inserts rows in order. Building a batch in parallel
  is Phase 3 work (see Thread safety).

### Accessors

`neighbors(id, layer)` returns a span over a list's filled slots (`list + 1`, `count`). `level`,
`entry_point`, and `max_level` exist so tests can check graph invariants directly instead of only
recall.

---

## 4. Why the neighbor-selection heuristic matters

TODO (stage b): Algorithm 4, the clustered-data failure of "closest M", and the measured
before/after recall from `bench/`.

---

## 5. Complexity

Let N be the number of nodes, d the dimension, and M, ef as usual. One distance costs O(d).

**Layers.** P(level >= l) = M^(−l), so the expected top layer is about log_M(N), and the
expected number of layers per node is 1 + 1/(M−1).

**Search.**
- Upper layers: about log_M(N) layers, each a greedy walk of O(1) expected hops (the paper's
  argument: a layer holds about M times fewer nodes than the one below, so starting from the
  previous layer's closest node the walk is short) with up to M distances per hop:
  O(M · log N · d).
- Layer 0: the beam search expands O(ef) nodes (roughly: the ones that entered W), each examining
  up to 2M neighbors: O(ef · M · d) distances, plus O(log ef) heap operations each.
- Total: **O((M · log N + ef · M) · d)**. In practice ef dominates, so doubling ef roughly
  doubles query time.

**Insertion.**
- The same descent, plus a beam search with `ef_construction` on each of the new node's
  1 + 1/(M−1) expected layers: O(ef_construction · M · d).
- Plus linking: M back-links per layer. An overflow re-selection costs O(M_max · d) distances
  plus a sort; the heuristic in stage (b) costs O(M_max² · d) in the worst case.
- Total per insert: **O((M · log N + ef_construction · M) · d)**, so a build is
  **O(N · log N)** in the paper's empirical scaling.

**Caveat.** These are *expected* costs on data with low intrinsic dimension, which is where
navigable graphs work. There is no worst-case guarantee: on adversarial or very high intrinsic
dimensional data, search can degrade toward linear and recall can drop. That is why Strata
measures recall against brute force instead of assuming it.

**Memory.** O(N · (d + M)); see section 2 for the exact bytes.

---

## 6. The parameters

| Parameter | Controls | Raising it | Typical |
|---|---|---|---|
| `M` | Links per node (2M on layer 0), and the layer ratio via mL | Better recall at fixed ef, especially on high-dimensional or clustered data; more memory (4·2M bytes/node on layer 0); slower inserts and hops | 12–48; 16 is the default |
| `ef_construction` | Beam width while inserting | Better neighbor candidates, so a better graph; build time grows roughly linearly. Gains flatten once it is well above M | 100–500 |
| `ef_search` | Beam width at query time (at least k) | Higher recall, lower QPS. **The only knob you can change after building**, and the one the recall-vs-QPS curves sweep | 10–500 |

`seed` does not affect quality on average; it makes builds reproducible.

TODO (stage c): measured recall-vs-QPS curves on SIFT10K and SIFT1M against hnswlib and FAISS
(Mac development results: recall valid, QPS indicative only).

---

## 7. Thread safety

- **Concurrent searches are safe.** `search` and every accessor are `const` and only read the
  index. Their only scratch state, the visited set, is `thread_local`.
- **Inserts are single-writer.** `add`/`add_batch` need exclusive access: no other call,
  including searches, may run at the same time. A concurrent insert rewrites neighbor lists that a
  search may be reading, and it can reallocate `data_`, `links0_`, and `upper_links_`. The Python
  bindings enforce this with a `shared_mutex`: searches take it shared, `add` takes it exclusive.
- **Why not concurrent inserts now?** It needs a lock per node (hnswlib's approach), a
  preallocated capacity so arrays never move, and care with the entry point. That is Phase 3
  (parallel build), after the single-threaded version is measured.

---

## 8. Deviations from the paper, and from hnswlib

| Point | Paper | hnswlib | Strata | Why |
|---|---|---|---|---|
| Next-layer entry points during insert | all of W | closest one | all of W | Faithful to the paper |
| Upper-layer descent | SEARCH-LAYER, ef = 1 | greedy loop | greedy loop | Same result, no heaps or visited set |
| Ties | unspecified | by heap order | by (distance, id) | Deterministic; duplicates stay well-defined |
| Level RNG | unspecified | `std::uniform_real_distribution` | hand-built U from mt19937_64 | Same graph on libc++ and libstdc++ |
| Visited set | a set | pool of epoch arrays | `thread_local` epoch array | Lock-free concurrent search |
