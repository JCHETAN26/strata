"""HNSW under deletion: search QPS and recall with tombstones versus a rebuilt index.

    uv run python bench/run_hnsw_delete_bench.py --dataset sift1m-200k-q1000
    uv run python bench/run_hnsw_delete_bench.py --dataset sift1m-200k-q1000 --report-only

One harness process (bench/search_harness.cpp, --delete-fractions and --compare-rebuild) builds
the index once, then deletes 0%, 25%, 50%, and 90% of the ids in turn (nested sets, chosen by a
hash of the id). At each fraction it recomputes exact ground truth over the live vectors, sweeps
ef_search on the tombstoned index, and builds a fresh index over only the live vectors and sweeps
that too. Writes:

    results/hnsw_deletes/<dataset>/strata-hnsw-<timestamp>.json   raw record
    results/hnsw_deletes/deletes_<dataset>.md                     table and rebuild guideline
    results/plots/hnsw_deletes_<dataset>.png                      recall vs. QPS per fraction

The guideline compares QPS at matched recall (interpolated along each curve), not at matched
ef_search: under deletion the same ef_search gives higher recall, so matching ef would flatter
the tombstoned index.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmeta import REPO_ROOT, git_info, hardware_note, preflight, thermal_warnings
from plot_recall_qps import MUTED, SURFACE, TEXT
from records import latest_records, load_records, save_record

OUT_DIR = REPO_ROOT / "results" / "hnsw_deletes"
TARGETS = [0.95, 0.99]
# One color per deleted fraction; solid = tombstoned, dashed = rebuilt.
COLORS = {0.0: "#2a78d6", 0.25: "#1baf7a", 0.5: "#eda100", 0.9: "#e0457b"}


def run(args: argparse.Namespace) -> None:
    harness = args.build_dir / "bench" / "strata_search"
    cmd = [
        str(harness), "--data", str(REPO_ROOT / "data" / args.dataset), "--metric", "l2",
        "--index", "hnsw", "--M", str(args.M), "--ef-construction", str(args.ef_construction),
        "--ef-search", args.ef_search, "--runs", str(args.runs),
        "--delete-fractions", args.fractions, "--compare-rebuild", "1",
    ]  # fmt: skip
    if git_info()["dirty"]:
        print("warning: uncommitted changes; the record will say so", file=sys.stderr)
    preflight("the delete sweep")
    print("$", " ".join(cmd), file=sys.stderr, flush=True)
    raw = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout)
    if warning := thermal_warnings():
        print(f"thermal warning after the run:\n{warning}", file=sys.stderr)
    if raw["build_type"] != "Release" or raw["asserts"]:
        raise SystemExit("harness is not a Release build without asserts")
    path = save_record(
        dataset=args.dataset,
        library="strata",
        index="hnsw",
        build_params={**raw["build_params"], "kernel": raw["kernel"], "sweep": "deletes"},
        build_seconds=raw["build_seconds"],
        k=raw["k"],
        threads=raw["threads"],
        num_queries=raw["num_queries"],
        command=cmd,
        points=[{"search_params": p["search_params"], "runs": p["runs"]} for p in raw["points"]],
        raw={key: value for key, value in raw.items() if key != "points"},
        out_dir=OUT_DIR,
    )
    print(f"saved {path.relative_to(REPO_ROOT)}")


def curves(record: dict[str, Any]) -> dict[tuple[float, bool], list[dict[str, Any]]]:
    """Points grouped by (deleted fraction, rebuilt), sorted by ef_search."""
    out: dict[tuple[float, bool], list[dict[str, Any]]] = {}
    for p in record["points"]:
        sp = p["search_params"]
        key = (round(sp["deleted_fraction"], 4), bool(sp.get("rebuilt", False)))
        out.setdefault(key, []).append(p)
    for pts in out.values():
        pts.sort(key=lambda p: p["search_params"]["ef_search"])
    return out


def qps_at_recall(points: list[dict[str, Any]], target: float) -> tuple[float, bool] | None:
    """QPS where the curve reaches `target` recall, interpolating log(QPS) linearly in recall
    between the two sweep points that bracket it. The flag is True when even the smallest
    ef_search measured already exceeds the target: that QPS is then only a lower bound (a smaller
    ef_search would be faster at the target). None if the curve never reaches the target."""
    prev = None
    for p in points:
        r, q = p["summary"]["recall"]["mean"], p["summary"]["qps"]["mean"]
        if r >= target:
            if prev is None:
                return q, True
            if prev[0] >= r:
                return q, False
            t = (target - prev[0]) / (r - prev[0])
            return math.exp(math.log(prev[1]) + t * (math.log(q) - math.log(prev[1]))), False
        prev = (r, q)
    return None


def qps_cell(summary: dict[str, Any]) -> str:
    return f"{summary['qps']['mean']:,.0f} ± {summary['qps']['stdev']:,.0f}"


def report(args: argparse.Namespace) -> None:
    records = [
        r
        for r in latest_records(load_records(OUT_DIR))
        if r["dataset"]["name"] == args.dataset and r["build_params"].get("sweep") == "deletes"
    ]
    if not records:
        raise SystemExit(f"no delete-sweep record for {args.dataset}; run without --report-only")
    record = max(records, key=lambda r: r["timestamp"])
    by = curves(record)
    fractions = sorted({f for f, _ in by})
    git = record["git"]
    lines = [
        f"# HNSW under deletion: tombstones vs. rebuild ({args.dataset})",
        "",
        "Generated by `bench/run_hnsw_delete_bench.py`. Do not edit by hand.",
        "",
        f"- **Hardware:** {hardware_note(record['hardware'])}",
        f"- Commit: {git['commit'][:10]}{' (dirty)' if git['dirty'] else ''}",
        f"- M={record['build_params']['M']}, ef_construction="
        f"{record['build_params']['ef_construction']}, single thread, "
        f"{len(record['points'][0]['runs'])} runs per point; recall@{record['k']} against exact "
        "ground truth over the live vectors at each fraction.",
        "- Deleted ids are chosen by a hash of the id, and the sets are nested. *Tombstoned* = the "
        "original index with tombstones. *Rebuilt* = a new index over only the live vectors, same "
        "parameters.",
        "",
    ]
    for f in fractions:
        tomb, rebuilt = by.get((f, False), []), by.get((f, True), [])
        live = tomb[0]["search_params"]["live_fraction"] if tomb else float("nan")
        lines += [f"## {f:.0%} deleted ({live:.1%} live)", ""]
        if rebuilt:
            secs = rebuilt[0]["search_params"]["rebuild_seconds"]
            lines += [f"Rebuild time: {secs:.1f} s (single thread).", ""]
        lines += [
            "| ef_search | recall (tombstoned) | QPS (tombstoned) "
            "| recall (rebuilt) | QPS (rebuilt) |",
            "|---:|---:|---:|---:|---:|",
        ]
        rebuilt_by_ef = {p["search_params"]["ef_search"]: p for p in rebuilt}
        for p in tomb:
            ef = p["search_params"]["ef_search"]
            s = p["summary"]
            row = f"| {ef} | {s['recall']['mean']:.4f} | {qps_cell(s)} |"
            if ef in rebuilt_by_ef:
                rs = rebuilt_by_ef[ef]["summary"]
                row += f" {rs['recall']['mean']:.4f} | {qps_cell(rs)} |"
            else:
                row += " — | — |"
            lines.append(row)
        lines.append("")

    lines += [
        "## QPS at matched recall",
        "",
        "Interpolated along each curve. *vs. 0%* compares with the index before any deletes; "
        "*vs. rebuilt* compares with a fresh index over the same live vectors.",
        "",
        "| deleted | recall | QPS tombstoned | vs. 0% | QPS rebuilt | tombstoned / rebuilt |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    base = by.get((0.0, False), [])

    def fmt(value: tuple[float, bool] | None, present: bool) -> str:
        if not present:
            return "—"
        if value is None:
            return "not reached"
        return f"{'≥ ' if value[1] else ''}{value[0]:,.0f}"

    def ratio(a: tuple[float, bool] | None, b: tuple[float, bool] | None) -> str:
        if a is None or b is None:
            return "—"
        # A lower bound on the numerator only keeps the ratio a lower bound. A lower bound on the
        # denominator makes the ratio meaningless either way.
        if b[1]:
            return "—"
        return f"{'≥ ' if a[1] and not b[1] else ''}{a[0] / b[0]:.2f}x"

    guideline: list[tuple[float, float, str]] = []
    for f in fractions:
        tomb, rebuilt = by.get((f, False), []), by.get((f, True), [])
        for target in TARGETS:
            qt, qr, q0 = (qps_at_recall(c, target) for c in (tomb, rebuilt, base))
            lines.append(
                f"| {f:.0%} | {target} | {fmt(qt, bool(tomb))} | {ratio(qt, q0)} "
                f"| {fmt(qr, bool(rebuilt))} | {ratio(qt, qr)} |"
            )
            if f > 0:
                guideline.append((f, target, ratio(qt, qr)))
    lines += [
        "",
        "*≥* marks a lower bound: the curve's smallest ef_search already exceeds the target "
        "recall.",
        "",
        "## Tombstoned / rebuilt QPS at matched recall",
        "",
        "| deleted | " + " | ".join(f"recall {t}" for t in TARGETS) + " |",
        "|---:|" + "---:|" * len(TARGETS),
    ]
    for f in sorted({g[0] for g in guideline}):
        cells = [r for g_f, _, r in guideline if g_f == f]
        lines.append(f"| {f:.0%} | " + " | ".join(cells) + " |")
    lines.append("")
    table = OUT_DIR / f"deletes_{args.dataset}.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines))
    print(f"wrote {table.relative_to(REPO_ROOT)}")
    plot(args.dataset, record, by, fractions)


def plot(
    dataset: str,
    record: dict[str, Any],
    by: dict[tuple[float, bool], list[dict[str, Any]]],
    fractions: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for f in fractions:
        color = COLORS.get(f, MUTED)
        for rebuilt in (False, True):
            pts = by.get((f, rebuilt), [])
            if not pts:
                continue
            xs = [p["summary"]["recall"]["mean"] for p in pts]
            ys = [p["summary"]["qps"]["mean"] for p in pts]
            ax.plot(
                xs, ys, color=color, linestyle="--" if rebuilt else "-", linewidth=2,
                marker="s" if rebuilt else "o", markersize=5, markeredgecolor=SURFACE,
                label=f"{f:.0%} deleted, {'rebuilt' if rebuilt else 'tombstoned'}",
            )  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel(f"Recall@{record['k']} (over live vectors)", color=TEXT)
    ax.set_ylabel("Queries per second (log scale, single thread)", color=TEXT)
    ax.set_title(
        f"{dataset}: HNSW under deletion, tombstones vs. rebuild\n"
        "Mac development results, not final",
        color=TEXT, loc="left",
    )  # fmt: skip
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower left")
    fig.text(0.01, 0.035, hardware_note(record["hardware"]), fontsize=7, color=MUTED)
    fig.text(
        0.99, 0.01, "Up and to the right is better. bench/run_hnsw_delete_bench.py",
        ha="right", fontsize=7, color=MUTED,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = REPO_ROOT / "results" / "plots" / f"hnsw_deletes_{dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="sift1m-200k-q1000")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", default="10,20,40,80,160,320")
    parser.add_argument("--fractions", default="0,0.25,0.5,0.9")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "release")
    args = parser.parse_args(argv)
    if not args.report_only:
        run(args)
    report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
