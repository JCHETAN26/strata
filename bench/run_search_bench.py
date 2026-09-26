"""Run the C++ search harness and save raw results + summary under results/.

    uv run python bench/run_search_bench.py --dataset siftsmall --index brute_force --runs 5

Output: results/search/<dataset>/<index>-<timestamp>.json containing the harness output (every
run), mean/stddev summaries, the dataset's meta.json, and commit/hardware metadata.
Tables are generated from these files by bench/make_tables.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmeta import REPO_ROOT, metadata, timestamp_slug

SUMMARY_FIELDS = [
    "qps",
    "recall",
    "build_seconds",
    "latency_mean_us",
    "latency_p50_us",
    "latency_p95_us",
    "latency_p99_us",
]


def summarize(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for field in SUMMARY_FIELDS:
        values = [r[field] for r in runs]
        summary[field] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, help="directory name under data/")
    parser.add_argument("--index", default="brute_force")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=0, help="0 = all queries")
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "release")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "search")
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
    print("$", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    result: dict[str, Any] = json.loads(proc.stdout)

    if result["build_type"] != "Release" or result["asserts"]:
        msg = f"harness is a {result['build_type']} build with asserts={result['asserts']}"
        if not args.allow_debug:
            print(f"error: {msg}; results would be misleading", file=sys.stderr)
            return 1
        print(f"warning: {msg}", file=sys.stderr)

    meta = metadata()
    if meta["git"]["dirty"]:
        print("warning: working tree has uncommitted changes", file=sys.stderr)

    record = {
        **meta,
        "dataset": {"name": args.dataset, **dataset_meta},
        "command": cmd,
        "result": result,
        "summary": summarize(result["runs"]),
    }
    out_dir = args.out_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.index}-{timestamp_slug()}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")

    s = record["summary"]
    print(
        f"{args.dataset} {args.index}: "
        f"qps {s['qps']['mean']:.1f} ± {s['qps']['stdev']:.1f}, "
        f"recall@{args.k} {s['recall']['mean']:.4f}, "
        f"p99 {s['latency_p99_us']['mean']:.1f} µs"
    )
    print(f"saved {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
