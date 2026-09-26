"""Run the filtered-search benchmark and save a result record under results/search/.

uv run python bench/run_filter_bench.py --dataset siftsmall --selectivity 0.01,0.1,0.5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmeta import REPO_ROOT
from records import save_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--selectivity", default="0.001,0.01,0.1,0.5,1.0")
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "release")
    args = parser.parse_args(argv)

    data_dir = REPO_ROOT / "data" / args.dataset
    metric = json.loads((data_dir / "meta.json").read_text())["metric"]
    cmd = [
        str(args.build_dir / "bench" / "strata_filter_bench"),
        "--data", str(data_dir),
        "--metric", metric,
        "--k", str(args.k),
        "--runs", str(args.runs),
        "--max-queries", str(args.max_queries),
        "--selectivity", args.selectivity,
    ]  # fmt: skip
    print("$", " ".join(cmd), file=sys.stderr)
    raw = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout)
    if raw["build_type"] != "Release" or raw["asserts"]:
        print("error: not a Release build; results would be misleading", file=sys.stderr)
        return 1
    path = save_record(
        dataset=args.dataset,
        library="strata",
        index="brute_force_filtered",
        build_params={"kernel": raw["kernel"]},
        build_seconds=0.0,
        k=raw["k"],
        threads=1,
        num_queries=raw["num_queries"],
        command=cmd,
        points=raw["points"],
        raw={key: value for key, value in raw.items() if key != "points"},
    )
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
