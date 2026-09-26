"""Run the storage benchmark and save raw results under results/storage/.

    uv run python bench/run_storage_bench.py --dataset siftsmall --runs 3

The benchmark directory should live on the disk you care about; the default is the system temp
directory. Its filesystem is recorded with the result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new
from records import summarize_fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fsync-inserts", type=int, default=500)
    parser.add_argument("--dir", type=Path, default=Path(tempfile.gettempdir()) / "strata_storage")
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "release")
    args = parser.parse_args(argv)

    binary = args.build_dir / "bench" / "strata_storage_bench"
    cmd = [
        str(binary),
        "--data", str(REPO_ROOT / "data" / args.dataset),
        "--dir", str(args.dir),
        "--runs", str(args.runs),
        "--fsync-inserts", str(args.fsync_inserts),
    ]  # fmt: skip
    print("$", " ".join(cmd), file=sys.stderr)
    raw = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout)
    if raw["asserts"]:
        print("error: benchmark built with asserts; use the release preset", file=sys.stderr)
        return 1

    df = subprocess.run(["df", "-P", str(args.dir.parent)], capture_output=True, text=True)
    record = {
        **metadata(),
        "dataset": args.dataset,
        "command": cmd,
        "filesystem": df.stdout.strip().splitlines()[-1] if df.returncode == 0 else "",
        "result": raw,
        "summary": summarize_fields(raw["runs"]),
    }
    path = REPO_ROOT / "results" / "storage" / f"{args.dataset}-{timestamp_slug()}.json"
    write_new(path, json.dumps(record, indent=2) + "\n")
    s = record["summary"]
    print(
        f"inserts/s: {s['insert_nosync_ops']['mean']:.0f} (no sync), "
        f"{s['insert_fsync_ops']['mean']:.0f} (fsync); "
        f"recovery: {s['recovery_wal_seconds']['mean'] * 1e3:.1f} ms (WAL), "
        f"{s['recovery_snapshot_seconds']['mean'] * 1e3:.1f} ms (snapshot)"
    )
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
