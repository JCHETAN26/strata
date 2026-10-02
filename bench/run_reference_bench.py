"""Benchmark reference libraries (hnswlib, FAISS) on the same data, same metric, single thread.

    uv run python bench/run_reference_bench.py --dataset siftsmall --library hnswlib
    uv run python bench/run_reference_bench.py --dataset siftsmall --library faiss --index hnsw
    uv run python bench/run_reference_bench.py --dataset siftsmall --library faiss --index flat

Methodology, matched to the Strata harness where possible:
- Search uses one thread (hnswlib set_num_threads(1), faiss.omp_set_num_threads(1)). So does the
  build, unless --build-threads N (for 10M-vector sets, where single-threaded builds take hours);
  then build_params records build_threads, as the Strata harness does.
- QPS comes from one batched call over all queries, so Python call overhead is excluded.
- Latency percentiles come from per-query calls and INCLUDE Python call overhead (a few µs); they
  overstate library latency for very fast queries. Compare QPS, not latency, across languages.
- Recall uses the same tie-aware definition as the C++ harness (bench/recall.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np
from benchmeta import REPO_ROOT
from records import save_record
from simd_info import library_simd, restrict_faiss_to_avx2

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_datasets import read_bin
from recall import recall_by_id, recall_with_ties


class Adapter:
    """Uniform wrapper: build(), set_search_param(value), search(queries, k) -> ids."""

    build_params: dict[str, Any]
    sweep_name: str | None

    def build(self, base: np.ndarray) -> None: ...
    def set_search_param(self, value: int) -> None: ...
    def search(self, queries: np.ndarray, k: int) -> np.ndarray: ...


class HnswlibAdapter(Adapter):
    def __init__(
        self, metric: str, dim: int, m: int, ef_construction: int, build_threads: int = 1
    ) -> None:
        import hnswlib

        space = {"l2": "l2", "angular": "cosine", "cosine": "cosine", "ip": "ip"}[metric]
        self.index = hnswlib.Index(space=space, dim=dim)
        self.m, self.ef_construction, self.build_threads = m, ef_construction, build_threads
        self.build_params = {"M": m, "ef_construction": ef_construction}
        if build_threads > 1:
            self.build_params["build_threads"] = build_threads
        self.sweep_name = "ef_search"

    def build(self, base: np.ndarray) -> None:
        self.index.init_index(
            max_elements=len(base), M=self.m, ef_construction=self.ef_construction, random_seed=42
        )
        self.index.set_num_threads(self.build_threads)
        self.index.add_items(base)
        self.index.set_num_threads(1)  # search is single-threaded

    def set_search_param(self, value: int) -> None:
        self.index.set_ef(value)

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        labels, _ = self.index.knn_query(queries, k=k)
        return labels.astype(np.int64)


class FaissAdapter(Adapter):
    def __init__(
        self,
        kind: str,
        metric: str,
        dim: int,
        m: int,
        ef_construction: int,
        build_threads: int = 1,
    ) -> None:
        import faiss

        self.build_threads = build_threads
        faiss.omp_set_num_threads(1)
        self.normalize = metric in ("angular", "cosine")
        faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        if kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, m, faiss_metric)
            self.index.hnsw.efConstruction = ef_construction
            self.build_params = {"M": m, "ef_construction": ef_construction}
            if build_threads > 1:
                self.build_params["build_threads"] = build_threads
            self.sweep_name = "ef_search"
        else:
            self.index = faiss.IndexFlat(dim, faiss_metric)
            self.build_params = {}
            self.sweep_name = None

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x = x / np.where(norms == 0, 1, norms)
        return x

    def build(self, base: np.ndarray) -> None:
        import faiss

        faiss.omp_set_num_threads(self.build_threads)
        self.index.add(self._prep(base))
        faiss.omp_set_num_threads(1)  # search is single-threaded

    def set_search_param(self, value: int) -> None:
        self.index.hnsw.efSearch = value

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        _, labels = self.index.search(self._prep(queries), k)
        return labels.astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--library", required=True, choices=["hnswlib", "faiss"])
    parser.add_argument("--index", default="hnsw", choices=["hnsw", "flat"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--build-threads", type=int, default=1, help="threads for the build only")
    parser.add_argument("--ef-search", default="10,20,40,80,160,320")
    parser.add_argument(
        "--simd",
        default="native",
        choices=["native", "avx2"],
        help="native: the widest SIMD the library and CPU support; avx2: held to AVX2 (FAISS "
        "via its SIMD level, hnswlib must be a build without AVX-512). Verified either way.",
    )
    args = parser.parse_args(argv)
    if args.library == "hnswlib" and args.index != "hnsw":
        parser.error("hnswlib only provides --index hnsw")
    if args.simd == "avx2":
        restrict_faiss_to_avx2()  # before FAISS is imported

    data_dir = REPO_ROOT / "data" / args.dataset
    import json

    metric = json.loads((data_dir / "meta.json").read_text())["metric"]
    base = read_bin(data_dir / "base.fbin", np.float32)
    queries = read_bin(data_dir / "query.fbin", np.float32)
    groundtruth = read_bin(data_dir / "groundtruth.ibin", np.int32)
    if args.max_queries:
        queries, groundtruth = queries[: args.max_queries], groundtruth[: args.max_queries]
    dim = base.shape[1]

    adapter: Adapter
    if args.library == "hnswlib":
        adapter = HnswlibAdapter(metric, dim, args.M, args.ef_construction, args.build_threads)
    else:
        adapter = FaissAdapter(
            args.index, metric, dim, args.M, args.ef_construction, args.build_threads
        )
    if args.simd == "avx2":
        restrict_faiss_to_avx2()  # and again now that FAISS is loaded
        adapter.build_params["simd"] = "avx2"

    # Fail before the (long) build if the library is not running the requested configuration.
    simd = library_simd(args.library, args.simd)
    if simd["problems"]:
        for problem in simd["problems"]:
            print(f"error: SIMD check failed ({args.simd}): {problem}", file=sys.stderr)
        return 1
    print(f"{args.library}: SIMD {simd['isa']} ({args.simd}, verified)", file=sys.stderr)

    t0 = time.perf_counter()
    adapter.build(base)
    build_seconds = time.perf_counter() - t0
    print(f"{args.library}/{args.index}: built in {build_seconds:.2f} s", file=sys.stderr)

    sweep = [int(x) for x in args.ef_search.split(",")] if adapter.sweep_name else [None]
    points = []
    for value in sweep:
        if value is not None:
            adapter.set_search_param(value)
        runs = []
        for _ in range(args.runs):
            adapter.search(queries, args.k)  # warmup pass, untimed
            t0 = time.perf_counter()
            labels = adapter.search(queries, args.k)
            search_seconds = time.perf_counter() - t0

            latencies = np.empty(len(queries))
            for i in range(len(queries)):
                t = time.perf_counter()
                adapter.search(queries[i : i + 1], args.k)
                latencies[i] = (time.perf_counter() - t) * 1e6

            runs.append(
                {
                    "search_seconds": search_seconds,
                    "qps": len(queries) / search_seconds,
                    "recall": recall_with_ties(base, queries, groundtruth, labels, metric, args.k),
                    "recall_by_id": recall_by_id(groundtruth, labels, args.k),
                    "latency_mean_us": float(latencies.mean()),
                    "latency_p50_us": float(np.percentile(latencies, 50, method="inverted_cdf")),
                    "latency_p95_us": float(np.percentile(latencies, 95, method="inverted_cdf")),
                    "latency_p99_us": float(np.percentile(latencies, 99, method="inverted_cdf")),
                    "latency_max_us": float(latencies.max()),
                }
            )
        search_params = {adapter.sweep_name: value} if adapter.sweep_name else {}
        print(
            f"  {search_params}: qps {np.mean([r['qps'] for r in runs]):.1f}, "
            f"recall@{args.k} {np.mean([r['recall'] for r in runs]):.4f}",
            file=sys.stderr,
        )
        points.append({"search_params": search_params, "runs": runs})

    import faiss
    import hnswlib

    versions = {"hnswlib": getattr(hnswlib, "__version__", "unknown"), "faiss": faiss.__version__}
    path = save_record(
        dataset=args.dataset,
        library=args.library,
        index=args.index,
        build_params=adapter.build_params,
        build_seconds=build_seconds,
        k=args.k,
        threads=1,
        num_queries=len(queries),
        command=sys.argv,
        points=points,
        raw={
            "library_version": versions[args.library],
            "qps_method": "batched call",
            # The instruction set its distance code ran with, the configuration requested, and
            # the check that they agree. See bench/simd_info.py.
            "simd": simd,
        },
    )
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
