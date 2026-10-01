"""Filtered HNSW search: graph vs pre-filter vs auto, random and cluster-correlated filters.

    uv run python bench/run_hnsw_filter_bench.py --dataset sift1m-200k-q1000
    uv run python bench/run_hnsw_filter_bench.py --dataset sift1m-200k-q1000 --report-only
    uv run python bench/run_hnsw_filter_bench.py --label cached-selectivity --threshold 0.013 \
        --selectivities 0.1,0.5

Phase 1 (graph): for each filter kind, one bench/hnsw_filter_bench.cpp process measures the exact
pre-filter and the graph strategy over an ef_search sweep at each selectivity. The crossover is
then derived from those numbers: for each kind and target recall, the selectivity below which the
pre-filter's QPS beats the graph's QPS at that recall. Phase 2 (auto): the auto strategy is
measured at the derived threshold (the largest crossover: auto cannot tell at query time whether a
filter is correlated, and the pre-filter is exact, so erring toward it is the safe side), reporting
how often it chose the pre-filter and how often the graph fell back.

Every process loads the same index snapshot (built once with the parallel build and saved under
data/), so all numbers come from one graph. One heavy job at a time: benchmeta.preflight before
each process, a thermal check after, and a cool-down between. Writes:

    results/hnsw_filter/<dataset>/{graph,auto}-{random,correlated}.json   raw output
    results/hnsw_filter/<dataset>/meta.json                                hardware, commit
    results/hnsw_filter/filter_<dataset>.md                                tables, crossover
    results/plots/hnsw_filter_<dataset>.png                                recall vs QPS

With --label, a re-measurement: both phases run into results/hnsw_filter/<dataset>/<label>/ at the
fixed --threshold (no derivation), and the report is a side-by-side of auto versus forced graph at
each ef_search, written to results/hnsw_filter/filter_<dataset>_<label>.md. Graph and auto come
from the same session, so the comparison is not skewed by thermal state or a different commit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmeta import REPO_ROOT, git_info, hardware_note, metadata, preflight, thermal_warnings
from plot_recall_qps import MUTED, SURFACE, TEXT

OUT_ROOT = REPO_ROOT / "results" / "hnsw_filter"
KINDS = ["random", "correlated"]
TARGETS = [0.95, 0.99]
COLORS = {0.001: "#e0457b", 0.01: "#eda100", 0.1: "#1baf7a", 0.5: "#2a78d6"}


def run_bench(args: argparse.Namespace, kind: str, phase: str, out: Path, threshold: float) -> None:
    if out.exists():
        return  # resume
    preflight(out.name)
    data = REPO_ROOT / "data" / args.dataset
    cmd = [
        str(args.build_dir / "bench" / "strata_hnsw_filter_bench"), "--data", str(data),
        "--snapshot", str(data / "hnsw_filter_index.snap"),
        "--clusters-cache", str(data / "hnsw_filter_clusters.bin"),
        "--kind", kind, "--phase", phase, "--selectivities", args.selectivities,
        "--ef-search", args.ef_search, "--runs", str(args.runs),
        "--max-queries", str(args.max_queries), "--threshold", str(threshold),
        "--build-threads", str(args.build_threads),
    ]  # fmt: skip
    print("$", " ".join(cmd), file=sys.stderr, flush=True)
    raw = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout)
    if raw["asserts"]:
        raise SystemExit("benchmark built with asserts; use the release preset")
    out.write_text(json.dumps(raw, indent=2) + "\n")
    if warning := thermal_warnings():
        raise SystemExit(f"thermal warning after {out.name}; stopping:\n{warning}")
    time.sleep(args.cooldown)


def mean(point: dict[str, Any], field: str) -> float:
    return statistics.fmean(r[field] for r in point["runs"])


def sd(point: dict[str, Any], field: str) -> float:
    values = [r[field] for r in point["runs"]]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def by_selectivity(raw: dict[str, Any]) -> dict[float, list[dict[str, Any]]]:
    out: dict[float, list[dict[str, Any]]] = {}
    for p in raw["points"]:
        out.setdefault(p["target_selectivity"], []).append(p)
    return out


def qps_at_recall(curve: list[dict[str, Any]], target: float) -> tuple[float, bool] | None:
    """Graph QPS at `target` recall, interpolating log(QPS) linearly in recall between bracketing
    ef_search points. The flag marks a lower bound (the smallest ef already exceeds the target).
    None if the curve never reaches the target."""
    prev = None
    for p in sorted(curve, key=lambda p: p["ef_search"]):
        r, q = mean(p, "recall"), mean(p, "qps")
        if r >= target:
            if prev is None:
                return q, True
            if prev[0] >= r:
                return q, False
            t = (target - prev[0]) / (r - prev[0])
            return math.exp(math.log(prev[1]) + t * (math.log(q) - math.log(prev[1]))), False
        prev = (r, q)
    return None


def crossover(raw: dict[str, Any], target: float) -> tuple[float, list[str]]:
    """The selectivity below which the pre-filter beats the graph at `target` recall, interpolated
    in log(selectivity) between measured points, plus one explanation line per selectivity."""
    rows = []
    notes = []
    for _, points in sorted(by_selectivity(raw).items()):
        pre = next(p for p in points if p["strategy"] == "prefilter")
        graph = [p for p in points if p["strategy"] == "graph"]
        actual = pre["selectivity"]
        gq = qps_at_recall(graph, target)
        pq = mean(pre, "qps")
        ratio = 0.0 if gq is None else gq[0] / pq
        rows.append((actual, ratio))
        shown = "not reached" if gq is None else f"{'>= ' if gq[1] else ''}{gq[0]:,.0f}"
        notes.append(f"{actual:.4%}: graph {shown} vs pre-filter {pq:,.0f} QPS")
    # rows sorted by selectivity; ratio (graph / pre-filter) grows with selectivity.
    for (s0, r0), (s1, r1) in itertools.pairwise(rows):
        if r0 < 1 <= r1:
            if r0 <= 0:
                return s1, notes
            t = (math.log(1) - math.log(r0)) / (math.log(r1) - math.log(r0))
            return math.exp(math.log(s0) + t * (math.log(s1) - math.log(s0))), notes
    if rows and rows[0][1] >= 1:
        return rows[0][0], notes  # the graph wins everywhere measured
    return rows[-1][0] if rows else 0.0, notes  # the pre-filter wins everywhere measured


def report(args: argparse.Namespace, out_dir: Path, meta: dict[str, Any], threshold: float) -> None:
    git = meta["git"]
    lines = [
        f"# Filtered HNSW search: {args.dataset}",
        "",
        "Generated by `bench/run_hnsw_filter_bench.py`. Do not edit by hand.",
        "",
        f"- **Hardware:** {hardware_note(meta['hardware'])}",
        f"- Commit: {git['commit'][:10]}{' (dirty)' if git['dirty'] else ''}",
        f"- One index (M=16, ef_construction=200, parallel build, saved and reloaded by every "
        f"process); {args.max_queries} queries; {args.runs} timed runs per point after a warmup; "
        "single-thread search. Recall@10 is tie-aware against exact filtered search.",
        "- Filters are CompiledFilters evaluated per query inside the timed region, so the "
        "pre-filter pays one filter test per id per query. **Random**: `hash(id) % 100000 < "
        "s * 100000`. **Correlated**: vectors assigned to the nearest of 1000 k-means centroids; "
        "the filter takes clusters in a fixed shuffled order until they cover s of the vectors. "
        "*Own cluster* = whether the query's nearest centroid is among them.",
        "- The crossover depends on index size (the pre-filter scales with n, the graph with "
        "ef / selectivity); it is re-measured at 1M and 10M in the AWS session.",
        "",
    ]
    graph_raw = {k: json.loads((out_dir / f"graph-{k}.json").read_text()) for k in KINDS}
    auto_raw = {
        k: json.loads(p.read_text()) for k in KINDS if (p := out_dir / f"auto-{k}.json").exists()
    }
    lines += [
        "## Crossover",
        "",
        "| kind | target recall | crossover selectivity |",
        "|---|---:|---:|",
    ]
    detail = []
    for kind in KINDS:
        for target in TARGETS:
            s, notes = crossover(graph_raw[kind], target)
            lines.append(f"| {kind} | {target} | {s:.3%} |")
            detail.append(f"- {kind}, recall {target}: " + "; ".join(notes))
    lines += ["", *detail, "", f"Auto threshold used below: **{threshold:.3%}** (the largest).", ""]

    ceilings = []
    for kind in KINDS:
        lines += [f"## {kind} filters", ""]
        for s, points in sorted(by_selectivity(graph_raw[kind]).items()):
            pre = next(p for p in points if p["strategy"] == "prefilter")
            own = pre["queries_own_match"]
            lines += [
                f"### {pre['selectivity']:.3%} of vectors match ({pre['matching']:,})"
                + (
                    f", {own} of {args.max_queries} queries' own cluster matches"
                    if kind == "correlated"
                    else ""
                ),
                "",
                "| strategy | ef_search | recall@10 | QPS | own cluster matches | does not | "
                "pre-filter used | fell back |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            rows = [pre, *[p for p in points if p["strategy"] == "graph"]]
            rows += [p for p in by_selectivity(auto_raw.get(kind, {"points": []})).get(s, [])]
            for p in rows:
                split = [mean(p, "recall_own_match"), mean(p, "recall_own_nomatch")]
                split_text = [("—" if v < 0 else f"{v:.4f}") for v in split]
                auto = p["strategy"] == "auto"
                lines.append(
                    f"| {p['strategy']} | {p['ef_search'] or '—'} | {mean(p, 'recall'):.4f} "
                    f"| {mean(p, 'qps'):,.0f} ± {sd(p, 'qps'):,.0f} | {split_text[0]} "
                    f"| {split_text[1]} | "
                    + (f"{mean(p, 'prefilter_fraction'):.0%}" if auto else "—")
                    + " | "
                    + (f"{mean(p, 'fallback_fraction'):.1%}" if auto else "—")
                    + " |"
                )
            graph = sorted(
                (p for p in points if p["strategy"] == "graph"), key=lambda p: p["ef_search"]
            )
            if len(graph) >= 2:
                top, below = mean(graph[-1], "recall"), mean(graph[-2], "recall")
                if top < 0.99 and top - below < 0.005:
                    ceilings.append(
                        f"- {kind}, {pre['selectivity']:.3%}: graph recall levels off at "
                        f"{top:.4f} (ef {graph[-2]['ef_search']} -> {graph[-1]['ef_search']} adds "
                        f"{top - below:+.4f})"
                    )
            lines.append("")
    lines += ["## Recall ceilings (graph strategy)", ""]
    lines += ceilings or ["None found: raising ef_search lifted recall toward 1 at every point."]
    lines.append("")
    out = OUT_ROOT / f"filter_{args.dataset}.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    plot(args, graph_raw, auto_raw, meta)


def compare_report(
    args: argparse.Namespace, out_dir: Path, meta: dict[str, Any], threshold: float
) -> None:
    """Auto versus forced graph at each selectivity and ef_search, from one labelled session."""
    git = meta["git"]
    lines = [
        f"# Filtered HNSW search, auto versus forced graph: {args.dataset} ({args.label})",
        "",
        "Generated by `bench/run_hnsw_filter_bench.py --label`. Do not edit by hand.",
        "",
        f"- **Hardware:** {hardware_note(meta['hardware'])}",
        f"- Commit: {git['commit'][:10]}{' (dirty)' if git['dirty'] else ''}",
        f"- Same index and filters as `filter_{args.dataset}.md`; {args.max_queries} queries; "
        f"{args.runs} timed runs per point after a warmup; single thread; auto threshold fixed "
        f"at {threshold:.3%}. Graph and auto were measured in the same session.",
        "- *auto / graph* is the ratio of mean QPS; 1.00 means auto costs nothing over forcing "
        "the strategy it chose.",
        "",
    ]
    for kind in KINDS:
        graph_raw = json.loads((out_dir / f"graph-{kind}.json").read_text())
        auto_raw = json.loads((out_dir / f"auto-{kind}.json").read_text())
        auto_by = by_selectivity(auto_raw)
        lines += [f"## {kind} filters", ""]
        for s, points in sorted(by_selectivity(graph_raw).items()):
            graph = {p["ef_search"]: p for p in points if p["strategy"] == "graph"}
            auto = {p["ef_search"]: p for p in auto_by.get(s, [])}
            lines += [
                f"### {points[0]['selectivity']:.3%} of vectors match",
                "",
                "| ef_search | graph QPS | auto QPS | auto / graph | graph recall | auto recall "
                "| pre-filter used | fell back |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for ef in sorted(graph.keys() & auto.keys()):
                g, a = graph[ef], auto[ef]
                lines.append(
                    f"| {ef} | {mean(g, 'qps'):,.0f} ± {sd(g, 'qps'):,.0f} "
                    f"| {mean(a, 'qps'):,.0f} ± {sd(a, 'qps'):,.0f} "
                    f"| {mean(a, 'qps') / mean(g, 'qps'):.2f} | {mean(g, 'recall'):.4f} "
                    f"| {mean(a, 'recall'):.4f} | {mean(a, 'prefilter_fraction'):.0%} "
                    f"| {mean(a, 'fallback_fraction'):.1%} |"
                )
            lines.append("")
    out = OUT_ROOT / f"filter_{args.dataset}_{args.label}.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def plot(
    args: argparse.Namespace,
    graph_raw: dict[str, Any],
    auto_raw: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, kind in zip(axes, KINDS, strict=True):
        ax.set_facecolor(SURFACE)
        for s, points in sorted(by_selectivity(graph_raw[kind]).items()):
            color = COLORS.get(s, MUTED)
            graph = sorted(
                (p for p in points if p["strategy"] == "graph"), key=lambda p: p["ef_search"]
            )
            ax.plot(
                [mean(p, "recall") for p in graph], [mean(p, "qps") for p in graph],
                color=color, marker="o", markersize=4, linewidth=2,
                label=f"{points[0]['selectivity']:.2%} graph",
            )  # fmt: skip
            pre = next(p for p in points if p["strategy"] == "prefilter")
            ax.scatter(
                [mean(pre, "recall")], [mean(pre, "qps")], color=color, marker="D", s=40,
                edgecolor=SURFACE, zorder=3, label=f"{pre['selectivity']:.2%} pre-filter",
            )  # fmt: skip
            auto = sorted(
                by_selectivity(auto_raw.get(kind, {"points": []})).get(s, []),
                key=lambda p: p["ef_search"],
            )
            if auto:
                ax.plot(
                    [mean(p, "recall") for p in auto], [mean(p, "qps") for p in auto],
                    color=color, linestyle="--", linewidth=1.2, alpha=0.8,
                )  # fmt: skip
        ax.set_yscale("log")
        ax.set_title(f"{kind} filters", color=TEXT, loc="left", fontsize=10)
        ax.set_xlabel("Recall@10 (over matching vectors)", color=TEXT)
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False, fontsize=7, labelcolor=TEXT, loc="lower left")
    axes[0].set_ylabel("Queries per second (log scale, single thread)", color=TEXT)
    fig.suptitle(
        f"{args.dataset}: filtered HNSW search (solid = graph, diamond = pre-filter, dashed = "
        "auto)\nMac development results, not final",
        color=TEXT, x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.text(0.01, 0.035, hardware_note(meta["hardware"]), fontsize=7, color=MUTED)
    fig.text(0.99, 0.01, "bench/run_hnsw_filter_bench.py", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = REPO_ROOT / "results" / "plots" / f"hnsw_filter_{args.dataset}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="sift1m-200k-q1000")
    parser.add_argument("--selectivities", default="0.001,0.01,0.1,0.5")
    parser.add_argument("--ef-search", default="10,20,40,80,160,320")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--cooldown", type=int, default=60)
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "release")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--build-threads", type=int, default=4, help="threads for the one-time index build"
    )
    parser.add_argument("--label", help="re-measurement: separate outputs, comparison report")
    parser.add_argument("--threshold", type=float, help="fixed auto threshold (needs --label)")
    args = parser.parse_args(argv)
    if (args.label is None) != (args.threshold is None):
        parser.error("--label and --threshold go together")

    out_dir = OUT_ROOT / args.dataset / (args.label or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        if git_info()["dirty"]:
            print("warning: uncommitted changes; the record will say so", file=sys.stderr)
        meta = {**metadata(), "args": {k: str(v) for k, v in vars(args).items()}}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if not args.report_only:
        for kind in KINDS:
            run_bench(args, kind, "graph", out_dir / f"graph-{kind}.json", 0.0)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        graph_raw = {k: json.loads((out_dir / f"graph-{k}.json").read_text()) for k in KINDS}
        threshold = max(crossover(graph_raw[k], t)[0] for k in KINDS for t in TARGETS)
        print(f"derived auto threshold: {threshold:.4%}", file=sys.stderr)
    if not args.report_only:
        for kind in KINDS:
            run_bench(args, kind, "auto", out_dir / f"auto-{kind}.json", threshold)
    if args.label:
        compare_report(args, out_dir, meta, threshold)
    else:
        report(args, out_dir, meta, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
