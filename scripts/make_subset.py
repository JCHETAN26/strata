"""Make a smaller dataset from a prepared one: the first N base vectors and first Q queries, with
exact ground truth recomputed for the subset.

Used for development runs on the fanless M2, where full SIFT1M builds run long enough to
overheat it. The source dataset's ground truth is for its full base set and does not apply to
a subset, so it is recomputed exactly (float64, ties broken by id; see make_clustered.py).

    uv run python scripts/make_subset.py --source sift1m --base 200000 --queries 1000

Writes data/<source>-<base/1000>k-q<queries>/ in the usual layout, with the source and sizes
recorded in meta.json. L2 datasets only (the ground truth is computed by squared L2).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from make_clustered import GROUNDTRUTH_K, exact_groundtruth
from prepare_datasets import DATA_DIR, read_bin, write_bin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default="sift1m", help="prepared dataset under data/")
    parser.add_argument("--base", type=int, default=200_000, help="number of base vectors")
    parser.add_argument("--queries", type=int, default=1000, help="number of queries")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite an existing subset")
    args = parser.parse_args(argv)

    src = args.data_dir / args.source
    src_meta = json.loads((src / "meta.json").read_text())
    if src_meta["metric"] != "l2":
        parser.error(f"{args.source} uses {src_meta['metric']}; only l2 subsets are supported")
    name = f"{args.source}-{args.base // 1000}k-q{args.queries}"
    out_dir = args.data_dir / name
    if (out_dir / "meta.json").exists() and not args.force:
        print(f"{name}: already present in {out_dir} (use --force to rebuild)")
        return 0

    base = np.ascontiguousarray(read_bin(src / "base.fbin", np.float32)[: args.base])
    query = np.ascontiguousarray(read_bin(src / "query.fbin", np.float32)[: args.queries])
    if len(base) < args.base or len(query) < args.queries:
        parser.error(f"{args.source} has only {len(base)} base vectors and {len(query)} queries")
    groundtruth = exact_groundtruth(base, query, GROUNDTRUTH_K)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    write_bin(out_dir / "base.fbin", base)
    write_bin(out_dir / "query.fbin", query)
    write_bin(out_dir / "groundtruth.ibin", groundtruth)
    meta = {
        "name": name,
        "source_url": f"subset of data/{args.source}: scripts/make_subset.py",
        "subset_of": {
            "name": args.source,
            "source_sha256": src_meta.get("source_sha256"),
            "base": args.base,
            "queries": args.queries,
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
