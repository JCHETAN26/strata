"""Generate a synthetic clustered dataset with exact ground truth, in Strata's binary format.

Gaussian blobs around uniformly random centers. Clustered data is where HNSW's neighbor-selection
heuristic matters most (Malkov & Yashunin 2018, section 4): with "closest M" selection every
link of a node points into its own cluster, so clusters can end up connected to each other only
through the sparse upper layers.

    centers    ~ Uniform([-1, 1]^dim), one per cluster
    base       center[c] + Normal(0, spread^2 I), per_cluster points per cluster
    queries    same mixture: a uniformly random cluster, then the same noise

Ground truth is exact: top-100 by squared L2, computed in float64. Output goes to data/<name>/
in the same layout as scripts/prepare_datasets.py (base.fbin, query.fbin, groundtruth.ibin,
meta.json). The generator parameters and seed are recorded in meta.json, so the dataset is
reproducible from its name alone.

Usage:
    uv run python scripts/make_clustered.py --clusters 100 --per-cluster 1000 --dim 16 \\
        --spread 0.05 --queries 1000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
from prepare_datasets import DATA_DIR, write_bin

GROUNDTRUTH_K = 100


def dataset_name(clusters: int, per_cluster: int, dim: int, spread: float, seed: int) -> str:
    return f"clustered-c{clusters}-n{per_cluster}-d{dim}-s{spread:g}-seed{seed}"


def generate(
    clusters: int, per_cluster: int, dim: int, spread: float, queries: int, seed: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1.0, 1.0, size=(clusters, dim))
    labels = np.repeat(np.arange(clusters), per_cluster)
    # Shuffle so insertion order does not visit one cluster at a time, which would be a
    # different (and easier-to-game) experiment.
    rng.shuffle(labels)
    base = centers[labels] + rng.normal(0.0, spread, size=(labels.size, dim))
    query_labels = rng.integers(0, clusters, size=queries)
    query = centers[query_labels] + rng.normal(0.0, spread, size=(queries, dim))
    return base.astype(np.float32), query.astype(np.float32)


def exact_groundtruth(
    base: npt.NDArray[np.float32], query: npt.NDArray[np.float32], k: int, chunk: int = 256
) -> npt.NDArray[np.int32]:
    """Ids of the k nearest base vectors per query by squared L2, ties broken by id."""
    b = base.astype(np.float64)
    b_norms = np.einsum("ij,ij->i", b, b)
    ids = np.empty((query.shape[0], k), dtype=np.int32)
    for start in range(0, query.shape[0], chunk):
        q = query[start : start + chunk].astype(np.float64)
        d = b_norms[None, :] - 2.0 * q @ b.T + np.einsum("ij,ij->i", q, q)[:, None]
        part = np.argpartition(d, k, axis=1)[:, : k + 1]
        for row in range(q.shape[0]):
            cand = part[row]
            order = np.lexsort((cand, d[row, cand]))  # by distance, then id
            ids[start + row] = cand[order][:k]
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--per-cluster", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--spread", type=float, default=0.05, help="per-coordinate noise stdev")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite an existing dataset")
    args = parser.parse_args(argv)

    name = dataset_name(args.clusters, args.per_cluster, args.dim, args.spread, args.seed)
    out_dir = args.data_dir / name
    if (out_dir / "meta.json").exists() and not args.force:
        print(f"{name}: already present in {out_dir} (use --force to rebuild)")
        return 0

    base, query = generate(
        args.clusters, args.per_cluster, args.dim, args.spread, args.queries, args.seed
    )
    groundtruth = exact_groundtruth(base, query, GROUNDTRUTH_K)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    write_bin(out_dir / "base.fbin", base)
    write_bin(out_dir / "query.fbin", query)
    write_bin(out_dir / "groundtruth.ibin", groundtruth)
    meta = {
        "name": name,
        "source_url": "synthetic: scripts/make_clustered.py",
        "generator": {
            "clusters": args.clusters,
            "per_cluster": args.per_cluster,
            "dim": args.dim,
            "spread": args.spread,
            "queries": args.queries,
            "seed": args.seed,
        },
        "metric": "l2",
        "base_shape": list(base.shape),
        "query_shape": list(query.shape),
        "groundtruth_shape": list(groundtruth.shape),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
