"""Tests for the Python bindings (strata._core).

Two groups:
- Behavior: conversions, padding, errors, locking, filters, PQ, HNSW (incl. deletes, save/load).
  No dataset needed.
- Agreement with C++ on SIFT10K: the C++ tool tests/strata_reference writes its results; the
  bindings must return the same ids and codes. Distances must be bit-identical when both builds
  compute floating point the same way (same kernel, compiler, FP flags, target; see
  strata.build_info()). Otherwise they must agree within DISTANCE_RTOL (relative).

Needs: `pip install -e .`, SIFT10K (`python scripts/prepare_datasets.py siftsmall`), and a C++
build of strata_reference (`cmake --build --preset release`). Missing pieces skip, not fail.
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

strata = pytest.importorskip("strata", reason="bindings not built: run `pip install -e .`")

from prepare_datasets import read_bin  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SIFT = REPO_ROOT / "data" / "siftsmall"
K = 10
# Used only when the two builds differ in kernel/compiler/FP flags/target. SIMD kernels are within
# 1e-5 * sum|terms| of scalar (strata::kSimdTolerance); for L2 that sum is the distance itself.
DISTANCE_RTOL = 1e-5
FP_KEYS = ("kernel", "compiler", "fp_flags", "target")


def rng_matrix(n: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d), dtype=np.float32)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Same hash as tests/strata_reference.cpp (uint64 arithmetic wraps, as in C++)."""
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


# --- Behavior ----------


def test_build_info_describes_the_build() -> None:
    info = strata.build_info()
    assert set(FP_KEYS) <= info.keys()
    assert info["kernel"] in {"neon", "avx2", "scalar"}
    assert "-ffp-contract=off" in info["fp_flags"]
    assert info["has_hnsw"] == strata.has_hnsw


def test_brute_force_matches_numpy() -> None:
    base, queries = rng_matrix(500, 24, 1), rng_matrix(20, 24, 2)
    index = strata.BruteForceIndex(24)
    np.testing.assert_array_equal(index.add(base), np.arange(500))
    ids, dists = index.search_batch(queries, K)
    assert ids.shape == dists.shape == (20, K)
    assert ids.dtype == np.int64 and dists.dtype == np.float32
    expected = ((queries[:, None, :].astype(np.float64) - base[None]) ** 2).sum(-1)
    np.testing.assert_array_equal(ids, np.argsort(expected, axis=1, kind="stable")[:, :K])
    np.testing.assert_allclose(dists, np.sort(expected, axis=1)[:, :K], rtol=1e-5)


def test_single_query_returns_1d_and_matches_batch() -> None:
    base, queries = rng_matrix(300, 8, 3), rng_matrix(5, 8, 4)
    index = strata.BruteForceIndex(8)
    index.add(base)
    batch_ids, batch_d = index.search_batch(queries, 3)
    for i, q in enumerate(queries):
        ids, d = index.search(q, 3)
        assert ids.shape == (3,)
        np.testing.assert_array_equal(ids, batch_ids[i])
        np.testing.assert_array_equal(d, batch_d[i])


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda x: x.astype(np.float64), id="float64"),
        pytest.param(lambda x: np.asfortranarray(x), id="fortran"),
        pytest.param(lambda x: np.repeat(x, 2, axis=0)[::2], id="strided"),
    ],
)
def test_non_float32_inputs_are_converted(transform) -> None:  # type: ignore[no-untyped-def]
    base, queries = rng_matrix(200, 16, 5), rng_matrix(7, 16, 6)
    reference = strata.BruteForceIndex(16)
    reference.add(base)
    converted = strata.BruteForceIndex(16)
    converted.add(transform(base))
    np.testing.assert_array_equal(
        converted.search_batch(transform(queries), 5)[0], reference.search_batch(queries, 5)[0]
    )


def test_python_lists_are_rejected_with_a_type_error() -> None:
    # Only array-like buffers are accepted; wrap lists with np.asarray.
    with pytest.raises(TypeError):
        strata.BruteForceIndex(2).add([[0.0, 1.0]])


def test_results_are_padded_faiss_style() -> None:
    index = strata.BruteForceIndex(2)
    index.add(np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32))
    ids, dists = index.search(np.zeros(2, np.float32), 5)
    np.testing.assert_array_equal(ids, [0, 1, 2, -1, -1])
    assert np.isinf(dists[3:]).all() and np.isfinite(dists[:3]).all()
    empty_ids, _ = strata.BruteForceIndex(2).search_batch(np.zeros((2, 2), np.float32), 3)
    assert (empty_ids == -1).all()


def test_results_outlive_the_index() -> None:
    index = strata.BruteForceIndex(4)
    index.add(rng_matrix(50, 4))
    ids, dists = index.search_batch(rng_matrix(3, 4, 1), 5)
    del index
    gc.collect()
    assert ids.shape == (3, 5) and np.isfinite(dists).all()  # buffers are owned by the arrays
    ids[0, 0] = 7  # and writable


def test_errors_become_python_exceptions() -> None:
    index = strata.BruteForceIndex(4)
    index.add(rng_matrix(10, 4))
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.zeros(3, np.float32), 1)
    with pytest.raises(ValueError, match="dimension"):
        index.add(rng_matrix(2, 5))
    with pytest.raises(ValueError):
        index.search(np.zeros((1, 1, 4), np.float32), 1)
    with pytest.raises(ValueError, match="threads"):
        index.search_batch(rng_matrix(2, 4), 1, threads=0)
    with pytest.raises(KeyError):
        index.remove(99)
    with pytest.raises(ValueError, match="metric"):
        strata.BruteForceIndex(4, metric="hamming")
    with pytest.raises(ValueError):
        strata.BruteForceIndex(0)
    assert len(index) == 10  # failed add left the index unchanged


def test_remove_excludes_vector() -> None:
    base = rng_matrix(20, 4)
    index = strata.BruteForceIndex(4)
    index.add(base)
    index.remove(3)
    assert index.is_deleted(3) and index.live_size == 19
    ids, _ = index.search(base[3], 20)
    assert 3 not in ids and ids[-1] == -1


def test_distance_functions() -> None:
    a, b = rng_matrix(1, 64, 1)[0], rng_matrix(1, 64, 2)[0]
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    assert strata.l2_squared(a, b) == pytest.approx(((a64 - b64) ** 2).sum(), rel=1e-5)
    assert strata.inner_product(a, b) == pytest.approx(-(a64 @ b64), rel=1e-5)
    cos = a64 @ b64 / np.linalg.norm(a64) / np.linalg.norm(b64)
    assert strata.cosine_distance(a, b) == pytest.approx(1 - cos, rel=1e-5)
    assert strata.l2_squared(a, b, kernel="scalar") == pytest.approx(
        strata.l2_squared(a, b), rel=1e-5
    )
    with pytest.raises(ValueError):
        strata.l2_squared(a, b[:10])
    queries, base = rng_matrix(3, 16, 3), rng_matrix(40, 16, 4)
    matrix = strata.distances(queries, base, metric="l2")
    assert matrix.shape == (3, 40)
    for i in range(3):
        for j in (0, 17, 39):
            assert matrix[i, j] == strata.l2_squared(queries[i], base[j])


# --- Filters ----------


def make_table() -> strata.AttributeTable:
    table = strata.AttributeTable([("year", "int"), ("source", "category")])
    table.extend([[2019, "arxiv"], [2020, "blog"], [2021, "arxiv"], [2022, "news"], [2023, "blog"]])
    return table


def test_filter_expressions() -> None:
    table = make_table()
    Filter = strata.Filter  # noqa: N806

    def ids(f: strata.Filter) -> list[int]:
        return list(np.flatnonzero(f.compile(table).evaluate()))

    assert ids(Filter.equals("source", "arxiv")) == [0, 2]
    assert ids(Filter.range("year", 2020, 2022)) == [1, 2, 3]
    assert ids(Filter.in_("source", ["news", "blog"])) == [1, 3, 4]
    assert ids(Filter.equals("source", "arxiv") & Filter.range("year", 2021, 9999)) == [2]
    assert ids(Filter.equals("source", "arxiv") | Filter.equals("year", 2022)) == [0, 2, 3]
    assert ids(~Filter.equals("source", "arxiv")) == [1, 3, 4]
    assert ids(Filter.equals("source", "podcast")) == []
    compiled = Filter.range("year", 0, 2020).compile(table)
    assert compiled.matches(1) and not compiled.matches(2)
    assert compiled.estimate_selectivity(10_000) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        Filter.equals("nope", 1).compile(table)
    with pytest.raises(ValueError):
        Filter.range("source", 0, 1).compile(table)


def test_compiled_filter_detects_table_changes() -> None:
    table = make_table()
    compiled = strata.Filter.range("year", 2020, 2030).compile(table)
    table.append([2024, "news"])
    with pytest.raises(RuntimeError, match="compile it again"):
        compiled.evaluate()


def test_filtered_search_forms_agree() -> None:
    base, queries = rng_matrix(1000, 8, 7), rng_matrix(10, 8, 8)
    index = strata.BruteForceIndex(8)
    index.add(base)
    table = strata.AttributeTable([("bucket", "int")])
    buckets = np.arange(1000) % 10
    table.extend([[int(b)] for b in buckets])
    compiled = strata.Filter.equals("bucket", 3).compile(table)
    mask = buckets == 3
    by_filter = index.search_filtered(queries, 5, compiled)
    by_mask = index.search_filtered(queries, 5, mask)
    by_ids = index.search_filtered(queries, 5, np.flatnonzero(mask))
    by_ids32 = index.search_filtered(queries, 5, np.flatnonzero(mask).astype(np.int32))
    for other in (by_mask, by_ids, by_ids32):
        np.testing.assert_array_equal(by_filter[0], other[0])
    assert np.isin(by_filter[0], np.flatnonzero(mask)).all()
    with pytest.raises(ValueError, match="covers"):
        index.search_filtered(queries, 5, mask[:-1])
    with pytest.raises(IndexError):
        index.search_filtered(queries, 5, np.array([5000]))


# --- Product quantization ----------


def test_pq_roundtrip_and_errors() -> None:
    data = rng_matrix(600, 16, 9)
    pq = strata.ProductQuantizer.train(data, m=4, kmeans_iterations=5)
    assert (pq.dim, pq.m, pq.sub_dim, pq.code_size) == (16, 4, 4, 4)
    codes = pq.encode(data[:10])
    assert codes.shape == (10, 4) and codes.dtype == np.uint8
    assert pq.encode(data[0]).shape == (4,)
    assert pq.decode(codes).shape == (10, 16)
    assert pq.compute_table(data[0]).shape == (4, 256)
    assert pq.codebooks.shape == (4, 256, 4)
    index = strata.PqIndex(pq, keep_originals=False)
    index.add(data)
    assert index.code_bytes == 600 * 4
    with pytest.raises(ValueError, match="rerank"):
        index.search(data[0], 5, rerank=10)
    with pytest.raises(ValueError, match="divisible"):
        strata.ProductQuantizer.train(data, m=5)


# --- HNSW ----------


def id_recall(got: np.ndarray, truth: np.ndarray) -> float:
    k = truth.shape[1]
    return float(np.mean([len(set(g) & set(t)) / k for g, t in zip(got, truth, strict=True)]))


def test_hnsw_is_built() -> None:
    # src/index/hnsw.cpp exists, so a module without HNSW is a stale or broken build, not a skip.
    assert strata.has_hnsw
    assert strata.build_info()["has_hnsw"]


@pytest.mark.parametrize("selection", ["heuristic", "simple"])
@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_hnsw_recall_vs_brute_force(metric: str, selection: str) -> None:
    base, queries = rng_matrix(3000, 32, 20), rng_matrix(100, 32, 21)
    index = strata.HnswIndex(32, metric=metric, selection=selection)
    index.add(base)
    exact = strata.BruteForceIndex(32, metric=metric)
    exact.add(base)
    truth = exact.search_batch(queries, K)[0]
    got, dists = index.search(queries, K, ef_search=128, threads=None)
    assert id_recall(got, truth) >= 0.95
    # Distances are exact: the same kernel, evaluated on the returned ids.
    all_dists = strata.distances(queries, base, metric=metric)
    np.testing.assert_array_equal(dists, np.take_along_axis(all_dists, got, axis=1))
    assert (np.diff(dists, axis=1) >= 0).all()  # sorted ascending


def test_hnsw_recall_improves_with_ef() -> None:
    base, queries = rng_matrix(3000, 32, 22), rng_matrix(100, 32, 23)
    index = strata.HnswIndex(32, M=8, ef_construction=64)
    index.add(base)
    exact = strata.BruteForceIndex(32)
    exact.add(base)
    truth = exact.search_batch(queries, K)[0]
    low = id_recall(index.search(queries, K, ef_search=10)[0], truth)
    high = id_recall(index.search(queries, K, ef_search=256)[0], truth)
    assert high > low and high >= 0.95


def test_hnsw_parameters_and_graph_introspection() -> None:
    index = strata.HnswIndex(8, M=6, ef_construction=40, selection="simple")
    assert (index.dim, index.M, index.ef_construction, index.selection) == (8, 6, 40, "simple")
    assert strata.HnswIndex(8).selection == "heuristic"
    assert len(index) == 0 and index.entry_point is None and index.max_level == -1
    np.testing.assert_array_equal(index.add(rng_matrix(500, 8, 24)), np.arange(500))
    assert len(index) == 500
    assert index.level(index.entry_point) == index.max_level
    for node in range(500):
        for layer in range(index.level(node) + 1):
            links = index.neighbors(node, layer)
            assert links.dtype == np.int64
            assert len(links) <= (12 if layer == 0 else 6)
            assert node not in links and len(set(links)) == len(links)


def test_hnsw_selection_worked_example() -> None:
    # Same example as the C++ test HeuristicPrefersDiverseNeighbors and docs/explainers/hnsw.md.
    points = np.array([[1, 0], [1.1, 0], [1.2, 0], [-3, 0], [0, 0]], dtype=np.float32)
    links = {}
    for selection in ("simple", "heuristic"):
        index = strata.HnswIndex(2, M=2, ef_construction=100, selection=selection)
        index.add(points)
        links[selection] = set(index.neighbors(4, 0).tolist())
    assert links == {"simple": {0, 1}, "heuristic": {0, 3}}


def test_hnsw_edge_cases_and_padding() -> None:
    empty = strata.HnswIndex(2)
    ids, dists = empty.search(np.zeros((2, 2), np.float32), 3)
    assert (ids == -1).all() and np.isinf(dists).all()
    index = strata.HnswIndex(2)
    index.add(np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32))
    ids, dists = index.search(np.zeros(2, np.float32), 5, ef_search=1)  # 1-D in, 1-D out
    np.testing.assert_array_equal(ids, [0, 1, 2, -1, -1])
    np.testing.assert_array_equal(dists[:3], [0, 2, 8])
    assert np.isinf(dists[3:]).all()
    dup = strata.HnswIndex(4, M=8, ef_construction=50)
    dup.add(np.tile(np.array([1, 2, 3, 4], np.float32), (100, 1)))
    ids, dists = dup.search(np.array([1, 2, 3, 4], np.float32), K, ef_search=50)
    assert len(set(ids.tolist())) == K and (dists == 0).all()


def test_hnsw_errors_become_python_exceptions() -> None:
    index = strata.HnswIndex(4)
    index.add(rng_matrix(10, 4))
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.zeros(3, np.float32), 1)
    with pytest.raises(ValueError, match="dimension"):
        index.add(rng_matrix(2, 5))
    with pytest.raises(ValueError, match="M must be"):
        strata.HnswIndex(4, M=1)
    with pytest.raises(ValueError, match="ef_construction"):
        strata.HnswIndex(4, ef_construction=0)
    with pytest.raises(ValueError, match="selection"):
        strata.HnswIndex(4, selection="closest")
    with pytest.raises(ValueError, match="metric"):
        strata.HnswIndex(4, metric="hamming")
    with pytest.raises(IndexError):
        index.level(10)
    with pytest.raises(IndexError):
        index.neighbors(0, index.level(0) + 1)
    assert len(index) == 10  # failed add left the index unchanged


def test_hnsw_same_seed_same_results_and_threads_agree() -> None:
    base, queries = rng_matrix(2000, 16, 25), rng_matrix(200, 16, 26)
    a, b = strata.HnswIndex(16, seed=7), strata.HnswIndex(16, seed=7)
    a.add(base)
    b.add(base)
    serial = a.search(queries, K, ef_search=40, threads=1)
    for other in (
        b.search(queries, K, ef_search=40),
        a.search(queries, K, ef_search=40, threads=None),
    ):
        np.testing.assert_array_equal(other[0], serial[0])
        np.testing.assert_array_equal(other[1], serial[1])


def test_hnsw_concurrent_adds_and_searches() -> None:
    index = strata.HnswIndex(8, M=8, ef_construction=40)
    index.add(rng_matrix(100, 8))
    batches = [rng_matrix(50, 8, seed) for seed in range(20)]

    def reader(seed: int) -> None:
        ids, _ = index.search(rng_matrix(20, 8, 100 + seed), 5, ef_search=20, threads=1)
        assert (ids >= 0).all()

    with ThreadPoolExecutor(8) as pool:
        added = pool.map(index.add, batches)
        list(pool.map(reader, range(40)))
        all_ids = np.concatenate(list(added))
    assert len(index) == 100 + 20 * 50
    np.testing.assert_array_equal(np.sort(all_ids), np.arange(100, 1100))


def test_hnsw_remove_excludes_and_counts() -> None:
    base, queries = rng_matrix(1000, 16, 27), rng_matrix(30, 16, 28)
    index = strata.HnswIndex(16, M=8, ef_construction=64)
    index.add(base)
    removed = set(range(0, 1000, 4))
    for i in removed:
        index.remove(i)
    assert len(index) == 1000 and index.live_count == 750
    assert index.is_deleted(0) and not index.is_deleted(1)
    ids, _ = index.search(queries, K, ef_search=64)
    assert (ids >= 0).all() and not set(ids.ravel().tolist()) & removed
    with pytest.raises(KeyError):
        index.remove(0)  # already deleted
    with pytest.raises(KeyError):
        index.remove(5000)
    with pytest.raises(IndexError):
        index.is_deleted(5000)


def test_hnsw_save_load_is_bit_identical(tmp_path: Path) -> None:
    base, queries = rng_matrix(1000, 16, 29), rng_matrix(30, 16, 30)
    index = strata.HnswIndex(16, metric="cosine", M=8, ef_construction=64, selection="simple")
    index.add(base)
    index.remove(3)
    index.save(tmp_path / "hnsw.snap")  # pathlib.Path
    loaded = strata.HnswIndex.load(str(tmp_path / "hnsw.snap"))  # and str
    assert (loaded.dim, loaded.M, loaded.ef_construction, loaded.selection) == (
        16,
        8,
        64,
        "simple",
    )
    assert len(loaded) == 1000 and loaded.live_count == 999 and loaded.is_deleted(3)
    before = index.search(queries, K, ef_search=64)
    after = loaded.search(queries, K, ef_search=64)
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[1], before[1])  # exact float equality


def test_hnsw_save_load_then_add_matches_never_saving(tmp_path: Path) -> None:
    first, second, queries = rng_matrix(600, 8, 31), rng_matrix(600, 8, 32), rng_matrix(20, 8, 33)
    saved = strata.HnswIndex(8, M=8, ef_construction=50)
    saved.add(first)
    saved.save(tmp_path / "hnsw.snap")
    resumed = strata.HnswIndex.load(tmp_path / "hnsw.snap")
    resumed.add(second)
    straight = strata.HnswIndex(8, M=8, ef_construction=50)
    straight.add(np.vstack([first, second]))
    assert resumed.max_level == straight.max_level
    assert resumed.entry_point == straight.entry_point
    for node in range(0, 1200, 37):
        assert resumed.level(node) == straight.level(node)
        np.testing.assert_array_equal(resumed.neighbors(node, 0), straight.neighbors(node, 0))
    np.testing.assert_array_equal(
        resumed.search(queries, K, ef_search=40)[0], straight.search(queries, K, ef_search=40)[0]
    )


def test_hnsw_load_errors(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        strata.HnswIndex.load(tmp_path / "missing.snap")
    index = strata.HnswIndex(4)
    index.add(rng_matrix(50, 4, 34))
    index.save(tmp_path / "hnsw.snap")
    data = bytearray((tmp_path / "hnsw.snap").read_bytes())
    data[100] ^= 0x01
    (tmp_path / "hnsw.snap").write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="checksum"):
        strata.HnswIndex.load(tmp_path / "hnsw.snap")


# --- Concurrency ----------


def test_threaded_searches_match_serial() -> None:
    base, queries = rng_matrix(5000, 32, 12), rng_matrix(400, 32, 13)
    index = strata.BruteForceIndex(32)
    index.add(base)
    serial = index.search_batch(queries, K, threads=1)[0]
    chunks = np.array_split(queries, 4)

    def run(chunk: np.ndarray) -> np.ndarray:
        return index.search_batch(chunk, K, threads=1)[0]

    start = time.perf_counter()
    for chunk in chunks:
        run(chunk)
    one_thread = time.perf_counter() - start
    with ThreadPoolExecutor(4) as pool:
        start = time.perf_counter()
        parallel = list(pool.map(run, chunks))
        four_threads = time.perf_counter() - start
    np.testing.assert_array_equal(np.vstack(parallel), serial)
    # Timing is noisy on a loaded laptop: report, don't fail.
    if one_thread / four_threads < 1.5:
        warnings.warn(
            f"4 Python threads gave only {one_thread / four_threads:.2f}x; is the GIL released?",
            stacklevel=1,
        )


def test_concurrent_adds_and_searches() -> None:
    index = strata.BruteForceIndex(8)
    index.add(rng_matrix(100, 8))
    batches = [rng_matrix(50, 8, seed) for seed in range(20)]

    def writer(batch: np.ndarray) -> np.ndarray:
        return index.add(batch)

    def reader(seed: int) -> None:
        ids, _ = index.search_batch(rng_matrix(20, 8, 100 + seed), 5, threads=1)
        assert (ids >= 0).all()

    with ThreadPoolExecutor(8) as pool:
        added = pool.map(writer, batches)
        list(pool.map(reader, range(40)))
        all_ids = np.concatenate(list(added))
    assert len(index) == 100 + 20 * 50
    # Adds are serialized: every id assigned exactly once.
    np.testing.assert_array_equal(np.sort(all_ids), np.arange(100, 1100))


# --- Agreement with C++ on SIFT10K -------------------------------------------------------------


def find_reference_tool() -> Path | None:
    # Both the cross-platform presets and the Linux GCC 13 presets (build/linux-*).
    presets = ("release", "debug", "asan", "linux-release", "linux-debug", "linux-asan")
    for preset in presets:
        path = REPO_ROOT / "build" / preset / "tests" / "strata_reference"
        if path.exists():
            return path
    return None


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if not (SIFT / "base.fbin").exists():
        pytest.skip("SIFT10K missing: run `python scripts/prepare_datasets.py siftsmall`")
    tool = find_reference_tool()
    if tool is None:
        pytest.skip("strata_reference not built: run `cmake --build --preset release`")
    out = tmp_path_factory.mktemp("reference")
    subprocess.run([str(tool), "--data", str(SIFT), "--out", str(out)], check=True)
    info = json.loads((out / "build_info.json").read_text())
    py_info = strata.build_info()
    same_fp = all(info[key] == py_info[key] for key in FP_KEYS)
    if not same_fp:
        warnings.warn(
            f"Python and C++ builds differ ({ {k: (py_info[k], info[k]) for k in FP_KEYS} }); "
            f"comparing distances within rtol={DISTANCE_RTOL} instead of bit for bit",
            stacklevel=1,
        )
    return {"dir": out, "same_fp": same_fp, "info": info}


@pytest.fixture(scope="module")
def sift() -> dict[str, np.ndarray]:
    return {
        "base": read_bin(SIFT / "base.fbin", np.float32),
        "query": read_bin(SIFT / "query.fbin", np.float32),
    }


def assert_matches_reference(
    reference: dict, name: str, ids: np.ndarray, dists: np.ndarray
) -> None:
    ref_ids = read_bin(reference["dir"] / f"{name}_ids.ibin", np.int32).astype(np.int64)
    ref_dist = read_bin(reference["dir"] / f"{name}_dist.fbin", np.float32)
    np.testing.assert_array_equal(ids, ref_ids, err_msg=f"{name}: ids differ")
    if reference["same_fp"]:
        # Same kernel, compiler, flags, target: must be the same bits.
        np.testing.assert_array_equal(
            dists.view(np.uint32), ref_dist.view(np.uint32), err_msg=f"{name}: distance bits differ"
        )
    else:
        np.testing.assert_allclose(dists, ref_dist, rtol=DISTANCE_RTOL, err_msg=name)


def test_comparison_uses_tolerance_when_builds_differ(tmp_path: Path) -> None:
    """The fallback branch: different FP builds must match ids, and distances within rtol."""
    from prepare_datasets import write_bin

    ids = np.array([[3, 1]], dtype=np.int32)
    dists = np.array([[1000.0, 2000.0]], dtype=np.float32)
    write_bin(tmp_path / "x_ids.ibin", ids)
    write_bin(tmp_path / "x_dist.fbin", dists)
    differ = {"dir": tmp_path, "same_fp": False}
    same = {"dir": tmp_path, "same_fp": True}
    nudged = dists * np.float32(1 + 5e-6)  # within DISTANCE_RTOL, but different bits
    assert_matches_reference(differ, "x", ids.astype(np.int64), nudged)
    with pytest.raises(AssertionError, match="bits differ"):
        assert_matches_reference(same, "x", ids.astype(np.int64), nudged)
    with pytest.raises(AssertionError):
        assert_matches_reference(differ, "x", ids.astype(np.int64), dists * np.float32(1.001))
    with pytest.raises(AssertionError, match="ids differ"):
        assert_matches_reference(differ, "x", ids[:, ::-1].astype(np.int64), dists)


def test_reference_build_matches_python_build(reference: dict) -> None:
    # On a single machine building both from this CMakeLists, the FP settings must agree; if this
    # fails, the Python build picked up different flags than the C++ build.
    assert reference["same_fp"], (reference["info"], strata.build_info())


@pytest.mark.parametrize("threads", [1, None])
def test_sift_brute_force_matches_cpp(reference: dict, sift: dict, threads: int | None) -> None:
    index = strata.BruteForceIndex(128)
    index.add(sift["base"])
    ids, dists = index.search_batch(sift["query"], K, threads=threads)
    assert_matches_reference(reference, "brute", ids, dists)


@pytest.fixture(scope="module")
def sift_pq(sift: dict) -> strata.ProductQuantizer:
    return strata.ProductQuantizer.train(sift["base"], m=16, kmeans_iterations=25, seed=42)


def test_sift_pq_codes_match_cpp(
    reference: dict, sift: dict, sift_pq: strata.ProductQuantizer
) -> None:
    ref_codes = read_bin(reference["dir"] / "pq_codes.ibin", np.int32).astype(np.uint8)
    np.testing.assert_array_equal(sift_pq.encode(sift["base"][:1000]), ref_codes)


@pytest.mark.parametrize("rerank", [0, 100])
def test_sift_pq_search_matches_cpp(
    reference: dict, sift: dict, sift_pq: strata.ProductQuantizer, rerank: int
) -> None:
    index = strata.PqIndex(sift_pq, keep_originals=True)
    index.add(sift["base"])
    ids, dists = index.search(sift["query"], K, rerank=rerank, threads=None)
    assert_matches_reference(reference, f"pq_r{rerank}", ids, dists)


@pytest.mark.parametrize("upper", [10, 100])
def test_sift_filtered_matches_cpp(reference: dict, sift: dict, upper: int) -> None:
    index = strata.BruteForceIndex(128)
    index.add(sift["base"])
    buckets = splitmix64(np.arange(len(sift["base"]), dtype=np.uint64)) % np.uint64(1000)
    table = strata.AttributeTable([("bucket", "int")])
    table.extend([[int(b)] for b in buckets])
    compiled = strata.Filter.range("bucket", 0, upper - 1).compile(table)
    np.testing.assert_array_equal(compiled.evaluate(), buckets < upper)
    ids, dists = index.search_filtered(sift["query"], K, compiled, threads=None)
    assert_matches_reference(reference, f"filt_{upper}", ids, dists)


@pytest.mark.parametrize("ef", [16, 64])
@pytest.mark.parametrize("selection", ["heuristic", "simple"])
def test_sift_hnsw_matches_cpp(reference: dict, sift: dict, selection: str, ef: int) -> None:
    # Same parameters, seed, and insertion order as strata_reference: the same graph, so the same
    # results.
    index = strata.HnswIndex(128, M=16, ef_construction=200, seed=42, selection=selection)
    index.add(sift["base"])
    ids, dists = index.search(sift["query"], K, ef_search=ef, threads=None)
    assert_matches_reference(reference, f"hnsw_{selection}_ef{ef}", ids, dists)
