"""Recall@10 vs QPS: Strata HNSW against hnswlib and FAISS HNSW, with a brute-force baseline.

    uv run python bench/run_hnsw_curves.py                        # SIFT10K + SIFT1M, then report
    uv run python bench/run_hnsw_curves.py --datasets siftsmall
    uv run python bench/run_hnsw_curves.py --report-only

Same build parameters for all three HNSW libraries (M, ef_construction; Strata uses its default
neighbor-selection heuristic), the same ef_search sweep, single thread, via
bench/run_search_bench.py (Strata) and bench/run_reference_bench.py (hnswlib, FAISS). Raw
records go to results/search/<dataset>/. Then writes:

    results/hnsw/<name>.md                    per-library recall and QPS, build time
    results/plots/<name>_<dataset>.png        recall vs QPS, one chart per dataset

<name> is --name (default hnsw_vs_reference). The report takes each library's latest record for
the dataset and parameters, so give AWS runs their own --name to keep the Mac tables.

Mac protocol (the default): 3 runs per point and one build configuration, with a pause between
builds so one library's heat does not slow the next. On the fanless M2, recall is valid but QPS
is indicative only; the outputs say so. Final speed comparisons run on dedicated hardware.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Any

from benchmeta import (
    REPO_ROOT,
    git_info,
    hardware_note,
    is_development_machine,
    preflight,
    thermal_warnings,
)
from plot_recall_qps import plot_dataset
from records import latest_records, load_records

LIBRARIES = [("strata", "hnsw"), ("hnswlib", "hnsw"), ("faiss", "hnsw"), ("strata", "brute_force")]
# Brute force on SIFT1M runs at ~50 QPS on the M2; a query subset keeps each run under a minute.
# On x86 (AVX2, one thread) GloVe-100 is similar to SIFT1M, and a 10M set is ~10x slower per query.
BRUTE_FORCE_MAX_QUERIES = {"sift1m": 1000, "glove100": 1000, "bigann10m": 200}
BENCH = REPO_ROOT / "bench"


def run(args: argparse.Namespace) -> None:
    # Every step runs in its own process: hnswlib and FAISS each ship an OpenMP runtime, and
    # loading both into one process aborts on macOS (OMP Error #15). Separate processes also
    # free each index (hundreds of MB on SIFT1M) before the next build starts.
    hnsw_args = ["--M", str(args.M), "--ef-construction", str(args.ef_construction)]
    if args.build_threads > 1:
        hnsw_args += ["--build-threads", str(args.build_threads)]
    sweep = ["--ef-search", args.ef_search, "--runs", str(args.runs)]
    for dataset in args.datasets:
        bench = ["--dataset", dataset]
        steps = [
            ["run_search_bench.py", *bench, "--index", "hnsw", *hnsw_args, *sweep],
            ["run_reference_bench.py", *bench, "--library", "hnswlib", *hnsw_args, *sweep],
            ["run_reference_bench.py", *bench, "--library", "faiss", "--index", "hnsw",
             *hnsw_args, *sweep],
            ["run_search_bench.py", *bench, "--index", "brute_force", "--runs", str(args.runs),
             "--max-queries", str(BRUTE_FORCE_MAX_QUERIES.get(dataset, 0))],
        ]  # fmt: skip
        for script, *step_args in steps:
            cmd = [sys.executable, "-u", str(BENCH / script), *step_args]
            preflight(" ".join(step_args[:4]))
            print("$", " ".join(cmd), file=sys.stderr, flush=True)
            subprocess.run(cmd, check=True)
            if warning := thermal_warnings():
                raise SystemExit(f"thermal warning after {script}; stopping:\n{warning}")
            print(f"cooling down {args.cooldown} s", file=sys.stderr, flush=True)
            time.sleep(args.cooldown)


def matches(record: dict[str, Any], dataset: str, args: argparse.Namespace) -> bool:
    if record["dataset"]["name"] != dataset or record["threads"] != 1:
        return False
    key = (record["library"], record["index"])
    params = record["build_params"]
    if key == ("strata", "brute_force"):
        # SIMD brute force only; records from before the kernel was recorded were scalar.
        return params.get("kernel") not in (None, "scalar")
    if key not in LIBRARIES:
        return False
    if params.get("M") != args.M or params.get("ef_construction") != args.ef_construction:
        return False
    if params.get("build_threads", 1) != args.build_threads:
        return False
    return key != ("strata", "hnsw") or params.get("selection") == "heuristic"


def select(dataset: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    records = [r for r in latest_records(load_records()) if matches(r, dataset, args)]
    found = {(r["library"], r["index"]) for r in records}
    missing = [f"{lib}/{idx}" for lib, idx in LIBRARIES if (lib, idx) not in found]
    if missing:
        raise SystemExit(
            f"{dataset}: no records for {', '.join(missing)}; run without --report-only"
        )
    return sorted(records, key=lambda r: LIBRARIES.index((r["library"], r["index"])))


def fmt(summary: dict[str, float], digits: int) -> str:
    return f"{summary['mean']:.{digits}f} ± {summary['stdev']:.{digits}f}"


def write_table(by_dataset: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    every = [r for records in by_dataset.values() for r in records]
    commits = sorted(
        {f"{r['library']} run at {r['git']['commit'][:10]}{'-dirty' if r['git']['dirty'] else ''}"
         for r in every}
    )  # fmt: skip
    lines = [
        "# Strata HNSW vs hnswlib and FAISS: recall@10 vs QPS",
        "",
        "Generated by `bench/run_hnsw_curves.py`. Do not edit by hand.",
        "",
        f"- **Hardware:** {hardware_note(every[0]['hardware'])}",
        f"- Build: M={args.M}, ef_construction={args.ef_construction}, "
        + (
            "single thread for build and search."
            if args.build_threads == 1
            else f"{args.build_threads} threads for the build (every library), one for search."
        )
        + " Strata uses the paper's neighbor-selection heuristic.",
        f"- {len(every[0]['points'][0]['runs'])} runs per point, mean ± stdev. Recall is "
        "tie-aware recall@10 against exact ground truth.",
        "- Strata QPS is measured in C++; hnswlib and FAISS QPS come from one batched Python call "
        "over all queries (Python overhead amortized away). See `bench/run_reference_bench.py`.",
    ]
    if every[0]["hardware"]["machine"] in ("arm64", "aarch64"):
        lines += [
            "- **The hnswlib speed comparison favors Strata on ARM:** hnswlib's distance code has "
            "no NEON path (its SIMD is SSE/AVX only), so here it computes distances in scalar "
            "code while Strata and FAISS use NEON. Fair speed comparisons against both libraries "
            "come from the x86 runs in Phase 9.",
        ]
    lines += [
        f"- Commits: {'; '.join(commits)}",
        "",
    ]
    for dataset, records in by_dataset.items():
        lines += [f"## {dataset}", ""]
        lines += ["| library | build (s) | queries | ef_search | recall@10 | QPS |"]
        lines += ["|---|---:|---:|---:|---:|---:|"]
        for record in records:
            name = f"{record['library']} {record['index'].replace('_', ' ')}"
            for i, point in enumerate(record["points"]):
                s = point["summary"]
                ef = point["search_params"].get("ef_search", "")
                build = f"{record['build_seconds']:.1f}" if i == 0 else ""
                queries = str(record["num_queries"]) if i == 0 else ""
                lines.append(
                    f"| {name if i == 0 else ''} | {build} | {queries} | {ef} "
                    f"| {fmt(s['recall'], 4)} | {fmt(s['qps'], 0)} |"
                )
        lines.append("")
    out_table = REPO_ROOT / "results" / "hnsw" / f"{args.name}.md"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text("\n".join(lines))
    print(f"wrote {out_table.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=["siftsmall", "sift1m"])
    parser.add_argument("--report-only", action="store_true", help="skip runs; report only")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", default="10,20,40,80,160,320")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cooldown", type=int, default=60, help="seconds between builds")
    parser.add_argument(
        "--build-threads",
        type=int,
        default=1,
        help="parallel builds for every library (10M sets); search stays single-threaded",
    )
    parser.add_argument(
        "--name",
        default="hnsw_vs_reference",
        help="output name: results/hnsw/<name>.md and results/plots/<name>_<dataset>.png "
        "(keeps AWS and Mac outputs apart)",
    )
    args = parser.parse_args(argv)
    if not args.report_only:
        if git_info()["dirty"]:
            print("warning: uncommitted changes; records will be marked dirty", file=sys.stderr)
        run(args)
    by_dataset = {dataset: select(dataset, args) for dataset in args.datasets}
    write_table(by_dataset, args)
    for dataset, records in by_dataset.items():
        out = REPO_ROOT / "results" / "plots" / f"{args.name}_{dataset}.png"
        plot_dataset(
            dataset,
            records,
            out=out,
            note=hardware_note(records[0]["hardware"])
            + (
                " hnswlib has no NEON path, so on ARM its speed is understated; fair speed "
                "comparisons against both libraries come from the x86 runs in Phase 9."
                if is_development_machine(records[0]["hardware"])
                else ""
            ),
            source="bench/run_hnsw_curves.py",
            title_suffix=(
                "Mac development results, not final"
                if is_development_machine(records[0]["hardware"])
                else None
            ),
        )
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
