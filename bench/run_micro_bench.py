"""Run Google Benchmark microbenchmarks with repetitions and save raw JSON under results/.

    uv run python bench/run_micro_bench.py --filter 'scalar/.*' --repetitions 5

Output: results/micro/<name>-<timestamp>.json (Google Benchmark's JSON plus commit/hardware).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmeta import REPO_ROOT, metadata, timestamp_slug


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="distance")
    parser.add_argument("--filter", default=".*")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-time", default="0.2s")
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "release")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "micro")
    args = parser.parse_args(argv)

    binary = args.build_dir / "bench" / "strata_bench"
    if not binary.exists():
        parser.error(f"{binary} not found; build with `cmake --build --preset release`")

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        cmd = [
            str(binary),
            f"--benchmark_filter={args.filter}",
            f"--benchmark_repetitions={args.repetitions}",
            f"--benchmark_min_time={args.min_time}",
            "--benchmark_report_aggregates_only=false",
            "--benchmark_out_format=json",
            f"--benchmark_out={tmp.name}",
        ]
        print("$", " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True, stdout=sys.stderr)
        gbench = json.loads(Path(tmp.name).read_text())

    if gbench["context"].get("library_build_type") == "debug":
        print("warning: Google Benchmark library is a debug build", file=sys.stderr)

    record = {**metadata(), "command": cmd, "benchmark": gbench}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.name}-{timestamp_slug()}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"saved {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
