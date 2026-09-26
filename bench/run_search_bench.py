"""Run the C++ search harness (Strata indexes) and save a result record under results/search/.

    uv run python bench/run_search_bench.py --dataset siftsmall --index brute_force
    uv run python bench/run_search_bench.py --dataset siftsmall --index hnsw \\
        --M 16 --ef-construction 200 --ef-search 10,20,40,80,160,320

See bench/records.py for the record format; bench/make_tables.py and bench/plot_recall_qps.py
render them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmeta import REPO_ROOT, git_info
from records import save_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, help="directory name under data/")
    parser.add_argument("--index", default="brute_force", choices=["brute_force", "hnsw"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=0, help="0 = all queries")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", default="10,20,40,80,160,320")
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "release")
    parser.add_argument(
        "--allow-debug", action="store_true", help="allow non-Release builds (not for results)"
    )
    args = parser.parse_args(argv)

    data_dir = REPO_ROOT / "data" / args.dataset
    dataset_meta = json.loads((data_dir / "meta.json").read_text())
    harness = args.build_dir / "bench" / "strata_search"
    if not harness.exists():
        parser.error(f"{harness} not found; build with `cmake --build --preset release`")

    cmd = [
        str(harness),
        "--data", str(data_dir),
        "--metric", dataset_meta["metric"],
        "--index", args.index,
        "--k", str(args.k),
        "--runs", str(args.runs),
        "--max-queries", str(args.max_queries),
    ]  # fmt: skip
    if args.index == "hnsw":
        cmd += [
            "--M", str(args.M),
            "--ef-construction", str(args.ef_construction),
            "--ef-search", args.ef_search,
        ]  # fmt: skip
    print("$", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    raw: dict[str, Any] = json.loads(proc.stdout)

    if raw["build_type"] != "Release" or raw["asserts"]:
        msg = f"harness is a {raw['build_type']} build with asserts={raw['asserts']}"
        if not args.allow_debug:
            print(f"error: {msg}; results would be misleading", file=sys.stderr)
            return 1
        print(f"warning: {msg}", file=sys.stderr)
    if git_info()["dirty"]:
        print("warning: working tree has uncommitted changes", file=sys.stderr)

    path = save_record(
        dataset=args.dataset,
        library="strata",
        index=raw["index"],
        build_params=raw["build_params"],
        build_seconds=raw["build_seconds"],
        k=raw["k"],
        threads=raw["threads"],
        num_queries=raw["num_queries"],
        command=cmd,
        points=[{"search_params": p["search_params"], "runs": p["runs"]} for p in raw["points"]],
        raw={key: raw[key] for key in ("build_type", "asserts", "compiler", "warmup_passes")},
    )
    record = json.loads(path.read_text())
    for point in record["points"]:
        s = point["summary"]
        print(
            f"{args.dataset} strata/{args.index} {point['search_params'] or ''}: "
            f"qps {s['qps']['mean']:.1f} ± {s['qps']['stdev']:.1f}, "
            f"recall@{args.k} {s['recall']['mean']:.4f}"
        )
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
