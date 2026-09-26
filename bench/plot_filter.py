"""Plot filtered-search QPS vs. selectivity per strategy (the crossover chart).

    uv run python bench/plot_filter.py

Writes results/plots/filter_<dataset>.png from the newest brute_force_filtered record (and, once
it exists, HNSW filtered records). Numbers are in results/tables.md.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmeta import REPO_ROOT
from records import latest_records, load_records

SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
# Fixed per strategy (validated categorical palette), plus a marker for non-color encoding.
STYLES = {
    "prefilter": ("#2a78d6", "o"),
    "scan_check": ("#eb6834", "s"),
    "hnsw_in_graph": ("#1baf7a", "^"),
    "hnsw_postfilter": ("#eda100", "D"),
}


def plot(dataset: str, records: list[dict[str, Any]]) -> str:
    curves: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for record in records:
        for point in record["points"]:
            params = point["search_params"]
            s = point["summary"]["qps"]
            curves[params["strategy"]].append((params["actual_selectivity"], s["mean"], s["stdev"]))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for strategy, pts in sorted(curves.items()):
        pts.sort()
        color, marker = STYLES.get(strategy, (MUTED, "x"))
        xs, ys, errs = zip(*pts, strict=True)
        name = strategy.replace("_", " ")
        ax.errorbar(xs, ys, yerr=errs, color=color, marker=marker, markersize=7, linewidth=2,
                    markeredgecolor=SURFACE, elinewidth=1, label=name)  # fmt: skip
        ax.annotate(name, (xs[0], ys[0]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color=TEXT)  # fmt: skip

    hardware = records[0]["hardware"].get("cpu") or records[0]["hardware"]["machine"]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Filter selectivity (fraction of vectors that match, log scale)", color=TEXT)
    ax.set_ylabel("Queries per second (log scale, single thread)", color=TEXT)
    ax.set_title(f"{dataset}: filtered search  ·  {hardware}", color=TEXT, loc="left")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper right")
    fig.text(0.99, 0.01, "Filter evaluation included per query. bench/plot_filter.py",
             ha="right", fontsize=7, color=MUTED)  # fmt: skip
    fig.tight_layout()
    out = REPO_ROOT / "results" / "plots" / f"filter_{dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return str(out.relative_to(REPO_ROOT))


def main() -> int:
    records = [r for r in latest_records(load_records()) if r["index"].endswith("_filtered")]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]["name"]].append(r)
    for dataset, group in sorted(by_dataset.items()):
        print("wrote", plot(dataset, group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
