# HNSW in Strata: how it works and why every line is there

This explains `src/index/hnsw.cpp` and `include/strata/hnsw.hpp` well enough to defend them
line by line. It follows Malkov & Yashunin, *Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs* (TPAMI 2018). "Algorithm N" always means
the paper's pseudocode.

Status: levels, layer search, insertion, and the paper's neighbor-selection heuristic (the
default), with closest-M selection kept as a switchable baseline; tombstone deletes and snapshot
save/load (section 9). Bound in Python and measured against hnswlib and FAISS on SIFT10K, SIFT1M,
and a 200k subset (section 6). The measurements are Mac development results: recall is final,
speed is indicative until the Phase 9 runs on dedicated hardware.

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
- **Prefetching: two passes per expansion.** Most of search time is spent waiting for memory.
  Each neighbor's vector (512 bytes for SIFT) sits at a scattered address, so on a large index
  almost every distance starts with a cache miss. A one-pass loop pays those misses one after
  another. `search_layer` instead expands a node in two passes:
  1. Walk the neighbor list, mark each unvisited neighbor, remember it, and issue
     `__builtin_prefetch` for its whole vector (one hint per 64 bytes, which covers every line
     on both x86's 64-byte and Apple's 128-byte cache lines). The loads now overlap instead of
     queueing.
  2. Compute distances and update C and W for those neighbors, in list order.

  Before the passes, it also prefetches the neighbor list of the node most likely to be
  expanded next (the new top of C).

  **Why results cannot change.** A prefetch is only a hint about timing. And because a neighbor
  list holds no duplicates, marking all of them visited before computing any distance selects
  the same neighbors, in the same order, as the one-pass loop. The C++ reference outputs
  (`strata_reference`: both selection modes, ef 16 and 64 on SIFT10K) were bit-identical before
  and after, and the A/B runs below found identical recall and identical graphs.

  **Measured** (`bench/run_ab_search.py`: 5 interleaved before/after pairs, 5 timed passes per
  ef per run; M2, so QPS is indicative):
  - `results/ab/prefetch-sift1m-200k.md` (200k SIFT vectors, ~100 MB, far larger than cache):
    about **2× QPS** at every ef_search (mean 1.80–2.12×; the worst single pair was 1.34×,
    against 1–6% run-to-run noise), and builds 0.63× the time, since insertion runs the same
    search.
  - `results/ab/prefetch-siftsmall.md` (SIFT10K, 5 MB, which fits in cache): **no measurable
    difference** (mean 0.96–1.00×, every range straddles 1). When the data is already cached,
    the hints are nearly free.

  Apple's hardware prefetchers are aggressive, so the gain on x86 may differ. It is re-measured
  on the Ryzen in Phase 9.
- **Tombstones (`kSkipDeleted`).** `search_layer` is a template with two instantiations. The
  plain one is used by inserts, and by searches when nothing is deleted. The tombstone-aware one
  expands deleted nodes but never lets them into W, and applies the stopping rule only once W is
  full (section 9 explains why). An index with no deletes pays nothing for the check.
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
   - `select_neighbors` picks up to M of them (the heuristic may keep fewer), and `set_links`
     writes them as the new node's out-links.
     The new node gets up to **M** links on every layer, including layer 0 (as in the paper and
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

### `select_neighbors(candidates, m)`: Algorithm 3 or Algorithm 4

Input: candidates sorted by distance to a *base node* b (the new node in `insert`, or the node
whose list overflowed in `add_link`), never containing b itself. Output: at most m of them,
chosen according to `params.selection`:

- `kSimple` (Algorithm 3): keep the m closest, which is a prefix because the input is sorted.
- `kHeuristic` (Algorithm 4, the default): see section 4. It walks the candidates nearest first
  and keeps each one only if no already-kept neighbor is strictly closer to it than b is. The kept
  entries are compacted to the front of the same vector (the write index `kept` never passes the
  read index `i`), so it allocates nothing.

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

### The failure of "closest M"

Picture clustered data: many tight groups, far apart. Under closest-M selection, the M nearest
candidates of almost every node are in its own cluster, so almost every layer-0 edge stays inside
a cluster. Edges *between* clusters appear only by accident, for example when a cluster's first
node is inserted before its neighbors exist. On layer 0 the clusters become islands, joined weakly
or not at all.

Search suffers in two ways:

1. **Unreachable nodes.** A node with no path from the entry point can never be returned, whatever
   `ef_search` is.
2. **Trapped beams.** If the upper-layer descent ends in the wrong cluster, the layer-0 beam search
   cannot leave it. Raising `ef` only explores the wrong island more thoroughly, so recall levels
   off below 1 instead of approaching it.

### The rule (Algorithm 4)

Walk the candidates from nearest to furthest. Keep a candidate e only if, for every neighbor r
already kept,

    d(e, b) <= d(e, r)          (e is at least as close to the base b as to r)

Otherwise discard e: some kept r is strictly closer to e than b is, so b → r already leads toward
e. A search can reach e through r, and a separate b → e edge would be redundant.

Geometrically, each kept neighbor r "claims" the region of space closer to r than to b. Later
candidates are kept only if they lie in a direction no kept neighbor covers. The links therefore
spread out in all directions from b, including toward other clusters, instead of piling up on the
nearest clump. This is the relative neighborhood graph idea, which is also used by other graph
indexes (e.g. NSG, Vamana/DiskANN's α-pruning).

### Worked example (the `HeuristicPrefersDiverseNeighbors` test)

M = 2. A tight cluster to the right of the new node q, and a lone point to the left:

    B(-3,0)            q(0,0)  A0(1,0) A1(1.1,0) A2(1.2,0)

Squared L2 distances to q: A0 = 1, A1 = 1.21, A2 = 1.44, B = 9.

- **Closest M:** A0 and A1. Both edges point into the same cluster, and q has no edge toward B.
- **Heuristic:** keep A0 (nothing kept yet). A1: d(A1, A0) = 0.01 < d(A1, q) = 1.21, so A0 covers
  it; discard. A2: 0.04 < 1.44; discard. B: d(B, A0) = 16 > d(B, q) = 9, so no kept neighbor covers
  it; keep. Result: {A0, B}, one edge each way.

### Measured difference

Source: `results/selection/selection_comparison.md` and `results/plots/hnsw_selection.png`, both
generated by `bench/run_selection_comparison.py`. M = 16, ef_construction = 200, seed 42. The
recall and graph numbers below are deterministic for a given seed, so they reproduce exactly. For
QPS see the generated table (Apple M2, fanless: indicative only).

**Clustered data** (`clustered-c100-n1000-d16-s0.05-seed0`: 100 Gaussian clusters × 1000 points,
16 dimensions):

| | closest M | heuristic |
|---|---:|---:|
| recall@10, ef 10 | 0.745 | 0.908 |
| recall@10, ef 40 | 0.806 | 0.998 |
| recall@10, ef 320 | 0.853 | 1.000 |
| Layer-0 nodes reachable from the entry point | 93.1% | 100% |
| Mean layer-0 out-degree | 24.2 | 16.5 |

Closest M levels off at 0.85: raising ef from 40 to 320 buys only +0.05, the "trapped beam"
failure above. About 7% of nodes cannot be reached at all. The heuristic reaches 1.000 by ef 80
**with a third fewer edges**, so each hop is also cheaper.

**SIFT10K** (real descriptors, dim 128, not strongly clustered): the effect is small. The
heuristic gains +0.022 recall at ef 10 and +0.024 at ef 20, then the two are equal. At ef 80 it
is slightly *behind*, 0.998 vs 1.000 (2 of 1000 results). Both graphs are fully reachable. This
matches the paper: the heuristic matters most for clustered and low-dimensional data, and barely
matters for data that is already well spread out.

**Cost.** Checking a candidate costs up to (number kept so far) extra distance computations, so
selection is O(|C| · m · d) instead of O(1) past the sort. Build time rose about 8% on both
datasets (indicative, same caveat as QPS). Because it keeps fewer edges, the heuristic graph is
also cheaper to search per hop.

### Details and choices

- **Ties keep the candidate** (`d(e, r) < d(e, b)` rejects; equal does not). With exact
  duplicates every distance is 0. A strict "must be closer to b than to every r" rule would reject
  every duplicate after the first, collapsing each node to a single link. With ties kept,
  duplicates link to each other normally (`DuplicateVectors`). hnswlib makes the same choice.
- **Applied even when there are at most m candidates**, as in the paper. hnswlib skips the
  heuristic in that case and keeps them all.
- **`extendCandidates` off.** The paper's option to add the candidates' own neighbors to the
  pool before selecting helps only on extremely clustered data, and costs many more distance
  computations. It is not implemented.
- **`keepPrunedConnections` off.** The paper's option to top the list up to m with discarded
  candidates would give back part of the sparsity that makes the heuristic work. It is not
  implemented, and hnswlib doesn't have it either.
- **Pruning uses it too.** When a neighbor's list overflows (`add_link`), the same rule
  re-selects from that neighbor's point of view. That is the place it matters most for keeping
  bridges between clusters: a pure distance cut would drop the long bridge edges first, since
  they are the longest.

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
  plus a sort; with the heuristic it is O(M_max² · d) in the worst case.
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

### Measured: ef_search sweep against hnswlib and FAISS

Source: `results/hnsw/hnsw_vs_reference.md` and `results/plots/hnsw_vs_reference_{siftsmall,sift1m}.png`,
generated by `bench/run_hnsw_curves.py` at commit `72c148e`. Setup: M = 16, ef_construction = 200,
single thread for build and search, 3 runs per point, all 10,000 SIFT1M queries. **These are Mac
development results.** The M2 is fanless and throttles, so recall is final (it does not depend on
the machine) but QPS and build times are indicative only. Final speed comparisons run on
dedicated hardware in Phase 9.

**Recall@10 on SIFT1M by ef_search** (deterministic for a fixed seed):

| ef_search | 10 | 40 | 80 | 160 | 320 |
|---|---:|---:|---:|---:|---:|
| Strata | 0.711 | 0.928 | 0.975 | 0.994 | 0.999 |
| hnswlib | 0.712 | 0.929 | 0.976 | 0.994 | 0.999 |
| FAISS | 0.716 | 0.934 | 0.978 | 0.994 | 0.999 |

What the curves show:

- **Same algorithm, same recall.** Strata matches hnswlib's recall at every ef to within 0.001.
  That is the strongest correctness check available: two independent implementations of the same
  paper, built with the same M and ef_construction, find the same neighbors. FAISS gets slightly
  higher recall at low ef (+0.005 at ef 10), converging by ef 160.
- **ef_search is the recall/speed dial.** Each doubling of ef roughly halves QPS, and the recall
  gained per doubling shrinks quickly: on SIFT1M, ef 40 → 80 adds 0.047 recall, ef 160 → 320 adds
  0.005. Past recall ≈ 0.99 you pay a lot of speed for very little recall.
- **Against brute force.** At recall 0.994 (ef 160), HNSW answers SIFT1M queries roughly 40× faster
  than SIMD brute force on the same machine. The gap grows with N, because HNSW's cost grows
  about logarithmically and brute force's linearly.
- **Speed (indicative, M2).** Strata runs between the two references: faster than hnswlib (about
  1.6–1.8× at matched ef) and slower than FAISS (about 0.7× at ef 10, closing to about 0.97×
  at ef 320).
  - **The hnswlib comparison favors Strata on ARM.** hnswlib has no NEON code path (its SIMD is
    SSE/AVX only), so on the M2 it computes distances in scalar code while Strata and FAISS use
    NEON. The 1.6–1.8× says little about the two graph implementations. Fair speed comparisons
    against both libraries come from the x86 runs in Phase 9, where all three use AVX2.
  - **FAISS is faster than Strata at low ef.** The first guess was the per-call allocation of
    `search_layer`'s heaps and output vector. **Profiling refuted it:** allocator time is about
    1.5% of search time (`results/profiles/sift1m-200k-q1000-ef40-9a5abdd.sample.txt`,
    `bench/profile_hnsw_search.sh`), so removing it could not close a 10–30% gap. Search time
    splits roughly evenly between the distance kernel (~45%) and `search_layer`'s own loop
    (~48%: heap updates, visited checks, loading neighbor lists), which points at memory access
    rather than allocation. Prefetching neighbor vectors, as FAISS and hnswlib do, then
    doubled QPS on a 200k subset (see `search_layer` above). The SIFT1M curves in this section
    predate it and have not been re-run; that happens with the Phase 9 runs. On the 200k subset
    after prefetching, Strata leads FAISS at every ef (about 1.4-1.9x, from a noisy 3-run
    comparison; `results/hnsw/hnsw_vs_reference.md`), while FAISS keeps slightly higher recall
    at low ef. A new profile puts ~60% of search time in `search_layer`'s own loop and ~33% in
    the distance kernel. The devlog (2026-09-28) records the next steps.
- **Build (indicative).** On SIFT1M, single-threaded, Strata took about 5 minutes, FAISS about
  7, and hnswlib about 10. Strata's build is not parallel yet (Phase 3).

**What is not measured yet:** only one build configuration was run (the reduced Mac protocol),
so the M and ef_construction rows in the table above describe the paper's and the libraries'
documented behavior, not Strata measurements. An M × ef_construction sweep belongs on the Ryzen
with the final runs.

---

## 7. Thread safety

- **Concurrent searches are safe.** `search` and every accessor are `const` and only read the
  index. Their only scratch state, the visited set, is `thread_local`.
- **Writes are single-writer.** `add`/`add_batch`/`remove` need exclusive access: no other call,
  including searches, may run at the same time. A concurrent insert rewrites neighbor lists that a
  search may be reading, and it can reallocate `data_`, `links0_`, and `upper_links_`. The Python
  bindings enforce this with a `shared_mutex`: searches and `save` take it shared; `add` and
  `remove` take it exclusive. `save` only reads, so it may run alongside searches.
- **Parallel build:** `add_batch(vectors, pool)` inserts one batch on several threads. It is still
  a write, so it still needs exclusive access: the concurrency is internal to that one call. See
  section 10.

---

## 8. Deviations from the paper, and from hnswlib

| Point | Paper | hnswlib | Strata | Why |
|---|---|---|---|---|
| Next-layer entry points during insert | all of W | closest one | all of W | Faithful to the paper |
| Upper-layer descent | SEARCH-LAYER, ef = 1 | greedy loop | greedy loop | Same result, no heaps or visited set |
| Ties | unspecified | by heap order | by (distance, id) | Deterministic; duplicates stay well-defined |
| Level RNG | unspecified | `std::uniform_real_distribution` | hand-built U from mt19937_64 | Same graph on libc++ and libstdc++ |
| Visited set | a set | pool of epoch arrays | `thread_local` epoch array | Lock-free concurrent search |
| Neighbor vectors | loaded on demand | prefetch next neighbor | prefetch all unvisited, then compute | Overlaps memory loads; results unchanged |
| Heuristic with <= m candidates | applied | skipped (keep all) | applied | Faithful to the paper |
| Heuristic ties d(e,r) = d(e,b) | unspecified | keep | keep | Duplicates stay connected |
| extendCandidates / keepPrunedConnections | optional flags | not implemented | not implemented | See section 4 |
| Deletes | not covered | tombstones (`markDelete`) | tombstones | Same trade-off; section 9 |
| Search under many deletes | not covered | W holds live nodes only | W holds live nodes; stop rule waits until W is full | Still returns k live results |

---

## 9. Deletes and persistence

### Deletes are tombstones

`remove(id)` sets one byte per node (`deleted_`) and bumps a counter. Nothing else changes: the
node keeps its vector and all its links, other nodes keep their links to it, and later inserts
may still pick it as a neighbor.

- **Why keep it in the graph?** Removing a node properly means repairing every list that points
  at it. Without that repair, the neighbors that relied on it could be cut off from the rest of the
  graph (the connectivity problem of section 4). A tombstone keeps the graph as navigable as it
  was. hnswlib's `markDelete` works the same way.
- **Why let inserts link to deleted nodes?** Then the graph depends only on what was inserted,
  never on which deletes happened (`DeletesDoNotChangeTheGraph`). That keeps save/load and
  write-ahead-log replay simple: replaying the inserts rebuilds exactly the same graph.
- **The trade-offs.** Memory is never reclaimed: a deleted node costs as much as a live one. And
  a search pays to walk through deleted nodes. Recall is *not* what suffers: it rises with
  deletes, because the search widens through tombstones until it holds ef_search live nodes.
  What suffers is speed.

### Measured: when to rebuild

Source: `results/hnsw_deletes/deletes_sift1m-200k-q1000.md` and
`results/plots/hnsw_deletes_sift1m-200k-q1000.png` (`bench/run_hnsw_delete_bench.py`, at
`0f25f11`). 200k SIFT vectors, M = 16, ef_construction = 200, 0 / 25 / 50 / 90% of ids deleted by
a hash, ground truth recomputed over the live vectors. It compares the tombstoned index with a
fresh index over only the live vectors, at **matched recall** (matching ef_search would flatter
the tombstoned index, which gets higher recall per ef). These are Mac development results, so the
ratios are indicative.

| deleted | tombstoned vs. rebuilt, recall 0.95 | recall 0.99 | tombstoned vs. before deletes, recall 0.99 | rebuild time |
|---:|---:|---:|---:|---:|
| 25% | 0.83x | 0.79x | 0.97x | 17.3 s |
| 50% | 0.69x | 0.64x | 0.91x | 10.1 s |
| 90% | ≥ 0.23x | 0.24x | 0.59x | 1.5 s |

**Guideline:** rebuild once roughly a quarter to a half of the index is deleted, if search speed
matters. At 25% deleted you are leaving ~20% QPS on the table compared with a rebuilt index; at
50%, ~35%; at 90%, ~75%. A rebuild costs about as much as building the live vectors (~17 s for
150k vectors, single thread, on the M2), and it also returns the dead nodes' memory. Relative to
the index before any deletes, tombstones are cheap up to 50% (0.91x at recall 0.99): the loss is
mostly that a smaller, rebuilt index would be faster still.

A rebuild assigns new, dense ids. Neither `HnswIndex` nor `Collection` offers a rebuild operation
yet, or a mapping from old ids to new ones; compaction is future work.

### Search with tombstones: why the stopping rule changes

In `search_layer<true>`, a deleted node that qualifies is pushed into C (so the search can walk
through it) but not into W (so it is never returned). The paper's stopping rule, "stop when C's
nearest is further than W's furthest", assumes every node in C was also offered to W. With
tombstones that no longer holds: W can hold only 3 live nodes while C holds deleted nodes beyond
them, and stopping there would return 3 results when k = 10 live results exist a few hops away.
So with tombstones the rule applies only once **W is full** (ef live nodes). Until then the
search keeps widening.

Without tombstones the two rules agree. While W is not full, every node ever pushed into C is
also in W, so C's nearest can never be beyond W's furthest. The plain instantiation keeps the
paper's rule exactly, and the C++ reference outputs stayed bit-identical.

Results: a search returns min(k, live nodes reachable from the entry point). Deleted nodes are
still traversed, so reachability does not shrink as nodes are deleted. Tests: with 95% deleted and
ef_search = k = 10, every query still returns 10 live results (recall at least 0.9 against brute
force over the live set). With 7 live nodes, all 7 come back in exact brute-force order. With
everything deleted, a search returns immediately with nothing (`live_size() == 0`), instead of
walking the whole dead graph.

### Persistence: what is saved and why

`save` writes a snapshot, format version 3 (`include/strata/snapshot.hpp`): header, vectors,
tombstone bitmap, and an index section that holds the graph (format in `include/strata/hnsw.hpp`).
It is one file with one CRC32C, written atomically (temp file, sync, rename, directory sync), so a
crash during `save` leaves either the old file or the new one, never a mix.

The index section holds **everything needed to keep building**, not only to search:

| Saved | Why search needs it | Why further inserts need it |
|---|---|---|
| Parameters (M, ef_construction, seed, selection) | M sets list capacities | Same insert behavior |
| Levels, layer-0 and upper-layer lists | The graph itself | Neighbors for new links, pruning |
| Entry point, max level | Where search starts | Where inserts start |
| **Level generator position** | Not needed | **The next node's level** |

The last row is the subtle one. Levels come from `std::mt19937_64`. Seeded afresh after a load,
node 1001 would get the level that node 1 got, the graph would diverge from never saving, and
`SaveLoadThenAddEqualsNeverSaving` would fail. That test does fail when the position is dropped
(deliberate mutations during development broke every case that adds after a load).

**How the position is saved: a draw count, not the engine's state.** Every call of the engine
goes through one function, `draw()`, which counts calls in `rng_draws_`. The snapshot stores that
count (a u64). Loading re-seeds with the saved seed and calls `discard(count)`, which advances the
engine exactly that many steps. `discard` and the engine's output sequence are defined by the
standard and computed identically everywhere, so the position restores identically on libc++
(macOS) and libstdc++ (Linux). Discarding costs one engine step per saved draw, linear in the
number of nodes like the rest of loading (not yet measured separately). The count is of generator calls,
not of inserts: today they are equal (one draw per insert), but anything that ever draws more
often stays correct.

**What went wrong first (snapshot format version 2).** Version 2 saved the engine's stream text
(`operator<<`), which the standard specifies as the 312 state words. libc++ writes exactly that.
libstdc++ writes 313 numbers (its state array plus an internal index) and requires the extra
number when reading. So a Mac-written snapshot failed to load on Linux, and the reverse fails
too. The first Linux run found it, the golden test failed with "bad generator state", while every
other byte of the index was identical. Version 2 files still load on the standard library that
wrote them. The draw count is then recovered by re-seeding and checking that one draw per node
reproduces the saved state, so saving again writes version 3. Under the other library they fail
with an error that names the mismatch and says to re-save on the machine that wrote the file.

### Loading defensively

A checksum catches accidental damage, but not a file written by a buggy version. And a bad
neighbor id is not a wrong answer, it is an out-of-bounds read the next time a search follows it.
`from_snapshot` therefore checks, before returning an index:
- every read is bounds-checked (a truncated or padded section fails; sizes from the file are
  guarded against overflow);
- the parameters pass the same checks as `create()`;
- `max_level` equals the largest node level, and the entry point is a node on that level;
- every list's count is within its layer's capacity, and every neighbor id is a node that exists
  on that layer.

That is enough for search and insert to touch only valid memory. It does not re-verify every
invariant (for example, duplicate links are harmless, since the visited set skips them), which
keeps loading linear in the graph size.

### Byte order and cross-machine files

The format is little-endian with IEEE 754 floats. Both are `static_assert`ed at compile time
(every supported host qualifies), and the header carries a byte-order mark, `0x01020304`,
checked on load. A file from a big-endian writer reads as `0x04030201` and is rejected with that
message instead of being misread.

`tests/golden/hnsw_v3.snap` was written on the Mac and is committed. On every machine that runs the
suite, `HnswGolden.LoadsBitIdenticallyOnThisMachine` loads it and must reproduce the stored results
bit for bit. It also rebuilds the same graph from the same vectors and seed, which must equal the
file, so it checks that level assignment and neighbor selection are platform-independent. It then
adds more vectors to both, which must stay identical, so it checks the restored generator position
too. The vectors are small integers, so NEON, AVX2, and scalar kernels all compute the same exact
distances. The first Linux run (Oracle, GCC 13) showed everything except the old generator text is
portable. Version 3 is written to close that gap and is next confirmed on Linux.

`tests/golden/hnsw_v2_libcxx.snap` is the same index in format version 2, written on the Mac. It
must load under libc++ (and upgrade to version 3 on save) and be rejected with the explanation
under libstdc++. A second test rewrites its generator text into the other library's shape, so the
rejection is exercised on every platform.

### Measured: loading instead of rebuilding

Source: `results/storage/hnsw_persist_sift1m-200k-q1000.md` (`bench/run_hnsw_persist_bench.py`, at
`397bd38`, snapshot format version 3; 3 runs; M2, indicative). For 200k × 128-dimensional vectors,
M = 16: build 25.6 s (single thread), save 0.53 s (durable and atomic, including fsyncs), load 0.39 s
(read, checksum, validate, re-seed and discard the level generator). The snapshot is 124 MiB, and
loading is about **65x faster** than rebuilding. Every run checked that the loaded index answers
all 1000 queries identically. One of the three runs was 15–25% slower in every step, build
included, so the spread reflects the machine, not the format. Format version 2 measured the same
within noise (build 23.5 s, save 0.51 s, load 0.36 s, at `0f25f11`).

### Known limit

Loading reads the whole file, then copies the vectors into the index, so peak memory is about
twice the index size. That is fine up to SIFT1M on the Mac. The AWS machine for the 10M run has
enough memory, so streaming loads are deferred.

---

## 10. Parallel build

`add_batch(vectors, pool)` links a batch of vectors on a `ThreadPool`. The sequential `add` and
`add_batch` stay the default, because they are deterministic. Python: `HnswIndex.add(vectors,
threads=...)`, default 1.

### Why the graph is not deterministic, but the levels are

Each insert searches the graph as it stands at that moment. With several threads, what a node
finds depends on which neighbors other threads have linked so far, so the neighbor lists depend on
timing. The **levels** do not: the parallel `add_batch` draws every node's level in id order before
any thread starts, exactly the draws a sequential build would make. So ids, levels, max level, and
the level generator's position match the sequential build. That keeps snapshots consistent: a
parallel-built index can be saved, loaded, and extended sequentially, deterministically from then
on. `Collection` never uses the parallel build, because its crash recovery relies on WAL replay
reproducing the graph exactly.

### Three steps

1. **Store everything first.** Every vector is copied in, every level drawn, and every node's
   neighbor lists allocated at their final size (empty). After this no array is resized, so
   every vector and list stays at a fixed address while threads run. Vectors and levels are never
   written again during the build, so they need no locks.
2. **The first node alone.** If the index is empty, the first node is linked on its own: it becomes
   the entry point.
3. **Link the rest in parallel**, with `parallel_for` handing out one id at a time. Threads then take
   ids nearly in order, so the graph grows much as it would sequentially. hnswlib's parallel add
   uses the same order.

### Locking

| Shared state | Protected by | Rule |
|---|---|---|
| A node's neighbor lists (all layers) | its lock: stripe `id % 65536` of a table of `std::mutex` | Read: lock, copy the list (at most 2M + 1 ids) into per-thread scratch, unlock, then traverse the copy. Write: only while holding it. |
| Entry point, max level | one `top` mutex | Every insert reads them under it. An insert whose level is above the current max holds it until done, then promotes itself; every other insert releases it at once. |
| Vectors, levels | nothing | Written in step 1, read-only after. |
| Visited set, scratch lists | nothing | Per thread (`thread_local`). |

- **No deadlock:** a thread never holds two node locks at once. It locks a node, copies or writes,
  and unlocks before touching another. The top lock is taken first, before any node lock. So
  striping, where two nodes can share a lock, cannot deadlock either.
- **Why striped:** one mutex per node costs 64 bytes each on macOS (640 MB at 10M nodes). A fixed
  table of 65,536 costs 4 MiB whatever the size, and two nodes sharing a lock only causes
  occasional contention. The table is created on the first parallel build; copying an index does
  not copy it.
- **Why the top lock is held through a promotion:** a node raising the top level becomes the new
  entry point. Holding the lock until it is fully linked means no other insert can start from an
  entry point whose lists are still empty. Promotions are rare (probability 1/M per level), so
  serializing them costs nothing measurable. hnswlib does the same.
- **Prefetching needs no lock:** a prefetch hint is not a memory access in the C++ model.
- **Sequential cost is zero:** every locking step is behind `if constexpr (kConcurrent)`, and the
  sequential instantiations contain no locks. The C++ reference outputs stayed bit-identical, and a
  pool of one thread builds exactly the sequential graph (`OneThreadEqualsSequentialExactly`).

### Three races the tests found

A sequential build never sees a node before it is linked. A parallel one does: a node becomes
reachable on its upper layers, through its own back-links, while it is still being linked on
lower ones. The first version missed three consequences. The tests caught each one.

1. **Overwritten back-links (reachability).** Another thread could add a back-link to node X on
   layer L before X wrote its own list on L. X's write then replaced it, and if that link was the
   other node's only way in, the other node became unreachable. Measured on random data, layer-0
   reachability fell from 1.0 (sequential) to 0.997 at 4 threads and 0.994 at 8. **Fix:** X
   *merges* its list (`merge_links`). Its lists start empty, so anything present must be back-links
   from other threads. They are kept, and re-selected only if they overflow, exactly as if they
   had arrived after X's write. After the fix: 0.9993-1.0 over repeated runs.
2. **Duplicate links.** Two nodes inserted at the same time can select each other. X linked to Y
   itself, then Y's back-link added Y to X's list again. **Fix:** in concurrent mode a back-link
   that already exists is skipped.
3. **Self-loops.** Once another thread has linked to X on layer L, X's own search on L can reach X.
   X then selected itself as a neighbor. **Fix:** in concurrent mode a node is removed from its own
   candidates.

Fixes 2 and 3 are checked only in concurrent mode: sequentially those cases cannot occur, so the
sequential path is unchanged.

### How it is tested

A parallel graph cannot be compared for equality, so the tests compare **quality** with the
sequential build of the same data (`tests/hnsw_parallel_test.cpp`, 4 threads, random 6000 × 32
and SIFT10K):
- well-formed (degree bounds, no self-loops or duplicates, neighbors on their layer, entry point
  on the top layer);
- identical levels and max level;
- layer-0 reachability at least 0.999;
- mean layer-0 degree within 5%;
- recall@10 against brute force within 0.01 at ef 10, 40, and 160.

Plus: one thread equals sequential exactly; save, load, then sequential add is deterministic; a
parallel batch on top of tombstones; and a **high-contention stress test** (M = 4, near-duplicate
clusters, 8 threads, repeated) run under ThreadSanitizer. TSan reported no races. The tests also
ran repeatedly to catch timing-dependent failures.

### Measured (Mac development results, indicative)

Source: `results/hnsw_build/build_scaling_sift1m-200k-q1000.md` (`bench/run_hnsw_build_scaling.py`,
at `97f4dda`). 200k SIFT vectors, M = 16, ef_construction = 200, 3 runs per thread count,
interleaved, with cool-downs and no thermal warnings.

| build threads | build time | speedup | recall@10 at ef 10 / 40 / 160 | layer-0 reachable | mean degree |
|---:|---:|---:|---|---:|---:|
| 1 (sequential) | 24.6 s | 1.00x | 0.766 / 0.956 / 0.998 | 1.000 | 20.3 |
| 2 | 13.6 s | 1.81x | 0.766 / 0.956 / 0.998 | 1.000 | 20.3 |
| 4 | 7.7 s | 3.20x | 0.765 / 0.956 / 0.998 | 1.000 | 20.3 |

Graph quality is unchanged: recall moves by at most 0.0004, reachability and degree match. The
speedup is 0.9x per thread at 2 threads and 0.8x at 4, where the M2's 4 performance cores are all
busy (and the fanless chassis runs warm). The full 1-16 thread curve, on a machine that does not
throttle, comes from the AWS session.

**What the stress test does not check:** reachability. Its data (5 tight clusters, M = 4) is
pathological for HNSW: 20 sequential builds in shuffled insertion orders gave layer-0
reachability anywhere from 0.2 to 1.0, and parallel builds fall inside that range. On such data
reachability measures insertion order, not correctness.

---

## 11. Filtered search

`search_filtered(query, k, filter, options)` returns the k nearest live vectors among those a
filter matches. The filter is a `CompiledFilter` (metadata predicates), or a `Bitset` of allowed
ids when the caller has already evaluated one and reuses it across queries. There are two
strategies and an automatic choice between them.

### The graph strategy: filtered-out nodes are tombstones

This reuses the deletion mechanism (section 9). `search_layer` takes a compile-time **keep
policy**: `KeepAll` (plain search, and inserts), `KeepLive` (tombstones), or `KeepLiveAllowed`
(tombstones plus the filter). Rejected nodes are still expanded, because they lead to accepted
ones, but never enter W. The stopping rule waits until W holds ef accepted nodes: the same
widening that makes a search under heavy deletion still return k results. So:
- the upper layers are walked unfiltered (they only choose a starting point);
- the filter runs only on nodes the search reaches, never on the whole index;
- reachability is unchanged by the filter, since non-matching nodes still carry the search;
- a filter matching everything gives exactly the unfiltered results, bit for bit
  (`FilterMatchingEverything`).

Policies are types, so `search()` and inserts compile to the same code as before. The C++
reference outputs stayed bit-identical.

**Cost:** to fill W with ef matches, the search expands roughly ef / selectivity nodes. At 50%
selectivity that is about twice the unfiltered work. At 0.1% it is most of the index.

### The pre-filter strategy: exact

Evaluate the filter on every id (for a `CompiledFilter`), then compute the distance to every
matching live vector and keep the k nearest. The result is exact. Its cost is one filter test per
id plus one distance per match: nearly constant as selectivity falls, and cheap when few vectors
match.

### Auto: estimate, choose, and fall back

1. **Estimate selectivity.** A `Bitset` is counted exactly (one popcount per 64 ids). A
   `CompiledFilter` is sampled: 1,000 random ids first. At 0.1% selectivity that sample holds
   about one match, too noisy to place a filter near the threshold. So if the first estimate lands
   within 3 standard errors of the threshold (the error of a 1,000-sample estimate at the
   threshold), it samples 20,000 more (exact when the index is no larger). Fixed seeds make the
   choice deterministic for a given filter.
2. **Choose.** Pre-filter if the estimate is below `prefilter_below` (default: the measured
   crossover, next subsection), otherwise the graph.
3. **Fall back.** A graph search that computes more than
   `(fallback_budget + estimated selectivity) × n` distances gives up, and the pre-filter answers
   that query. That budget is about the pre-filter's own cost in distance units: one filter test
   per id (about a tenth of a distance, hence the default of 0.1) plus one distance per match. So a
   query the graph handles badly (a hard filter the estimate misjudged, or matches far from the
   query) costs at most about twice the pre-filter's price, and is answered exactly. A forced
   `kGraph` never falls back.

`FilteredSearchStats` reports, per query, the strategy used, whether it fell back, the estimate,
whether it resampled, and the graph's distance count. That is what the measurements count.

### A filter-compiler fix found on the way

`Filter::in` on an int column compiled to an OR of one-point ranges, so testing an id cost one
comparison per listed value. A correlated filter listing 500 clusters then made the pre-filter
evaluate 500 comparisons per id, which would have skewed the crossover toward the graph. Now an int
`in` is a sorted set (`kIntIn`), and both int and category sets use a linear scan up to 16 values
and binary search above. Pre-filter QPS on a 50% correlated filter (SIFT10K smoke test) went from
142 to 3,346.

### Measured (Mac development results, indicative)

Source: `results/hnsw_filter/filter_sift1m-200k-q1000.md` and
`results/plots/hnsw_filter_sift1m-200k-q1000.png` (`bench/run_hnsw_filter_bench.py`, at
`08eb3cb`). 200k SIFT vectors, one index (M = 16, ef_construction = 200) loaded by every process,
500 queries, 3 runs per point. Filters are evaluated per query inside the timed region. Random
filters use a hash of the id; correlated filters take whole k-means clusters (1,000 clusters).

**The crossover is about 1%.** For each filter kind and target recall, it is the selectivity below
which the pre-filter's QPS beats the graph's QPS at that recall:

| filters | recall 0.95 | recall 0.99 |
|---|---:|---:|
| random | 1.03% | 1.04% |
| correlated | 1.25% | 1.28% |

`kDefaultPrefilterBelow` is the largest, rounded: **1.3%**. Auto cannot tell at query time whether a
filter is correlated, and the pre-filter is exact, so erring toward it is the safe side. Either
side of the crossover the gap is large: at 0.1% random the pre-filter runs at 2.3k QPS against the
graph's 28-317 QPS (ef 320 down to 10), and at 50% the graph runs at 2.7k-43k QPS against the
pre-filter's 204.

**Auto made the right choice everywhere measured.** It chose the pre-filter for 100% of queries at
0.1%, 0.2%, 1.0% and 1.1% selectivity (recall 1.0), and the graph for 100% at 10% and 50%.

**The fallback never fired in this protocol (0% everywhere),** because every measured selectivity
is far from the threshold: the estimate chose correctly, and no graph search came near its budget.
The fallback is tested (`FallbackToPreFilterWhenTheGraphSearchIsTooLong`) but not yet measured.
Measuring it needs selectivities close to the threshold (1-3%), or filters whose sampled estimate
misleads.

**No recall ceiling on correlated filters.** Raising ef lifted the graph's recall to 1.0 at every
point, correlated included (1.0 by ef 40-320). The split by the query's own cluster shows one
pattern: at low ef, queries whose own cluster matches get lower recall (at 10% selectivity,
ef 10: 0.89 vs 0.95), and the difference is gone by ef 80. The expected failure, a query far from
every matching cluster that the graph cannot reach, did not appear at these sizes. If it appears at
larger scale, predicate-aware traversal (for example ACORN, which adds edges so filtered subgraphs
stay connected) is the known remedy. That is future work.

**A cost of auto: per-query sampling.** At high selectivity auto runs slower than a forced graph
search at low ef: 32k vs 43k QPS (random 50%, ef 10), and 14k vs 41k (correlated 50%, ef 10,
noisy). The 1,000-id selectivity sample costs about as much as a fast graph search itself. Two
cheaper options: pass a `Bitset` (counted exactly with popcount), or, a future improvement,
estimate once per filter rather than once per query.

**Depends on n.** The pre-filter's cost grows with the index size, the graph's with
ef / selectivity (and slowly with n), so the crossover moves with n. It is re-measured at 1M and
10M in the AWS session before being quoted as general.

