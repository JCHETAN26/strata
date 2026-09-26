"""Plot PQ memory vs. recall from saved result records.

    uv run python bench/plot_pq_memory.py

Writes results/plots/pq_memory_<dataset>.png: recall@k against bytes per vector for each PQ
code size m, with ADC only (rerank=0) and with reranking, and the uncompressed float32 baseline.
Reranking reads full-precision vectors; its memory column counts codes only, as if the originals
lived on disk. The numbers are in results/tables.md.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmeta import REPO_ROOT
from records import latest_records, load_records

SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
ADC_COLOR, RERANK_COLOR, BASE_COLOR = "#2a78d6", "#eb6834", "#eda100"
RERANK_DEPTH = 100


def recall_at(record: dict[str, Any], rerank: int) -> float | None:
    for point in record["points"]:
        if point["search_params"].get("rerank") == rerank:
            return point["summary"]["recall"]["mean"]
    return None


def plot(dataset: str, pq: list[dict[str, Any]], brute: dict[str, Any] | None) -> str:
    pq = sorted(pq, key=lambda r: r["build_params"]["m"])
    k = pq[0]["k"]
    dim = pq[0]["raw"]["dim"]
    xs = [r["build_params"]["m"] for r in pq]  # 1 byte per sub-quantizer
    adc = [recall_at(r, 0) for r in pq]
    reranked = [recall_at(r, RERANK_DEPTH) for r in pq]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(xs, adc, color=ADC_COLOR, marker="o", markersize=7, linewidth=2,
            markeredgecolor=SURFACE, label="PQ, ADC only")  # fmt: skip
    ax.plot(xs, reranked, color=RERANK_COLOR, marker="s", markersize=7, linewidth=2,
            markeredgecolor=SURFACE, label=f"PQ + rerank top {RERANK_DEPTH}")  # fmt: skip
    for x, y in zip(xs, adc, strict=True):
        ax.annotate(f"{dim * 4 // x}x", (x, y), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED)  # fmt: skip
    if brute is not None:
        full = dim * 4
        ax.scatter(
            [full],
            [brute["points"][0]["summary"]["recall"]["mean"]],
            color=BASE_COLOR,
            marker="D",
            s=50,
            zorder=3,
            edgecolors=SURFACE,
            label="float32 (brute force)",
        )
        ax.annotate("float32", (full, 1.0), xytext=(-8, 6), textcoords="offset points",
                    ha="right", fontsize=8, color=TEXT)  # fmt: skip
    ax.annotate("PQ + rerank", (xs[0], reranked[0]), xytext=(6, -12), textcoords="offset points",
                fontsize=8, color=TEXT)  # fmt: skip
    ax.annotate("ADC only", (xs[0], adc[0]), xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=TEXT)  # fmt: skip

    hardware = pq[0]["hardware"].get("cpu") or pq[0]["hardware"]["machine"]
    ax.set_xscale("log", base=2)
    ax.set_xticks([*xs, dim * 4], [str(x) for x in [*xs, dim * 4]])
    ax.set_xlabel("Bytes per vector (log scale; labels under points = compression ratio)",
                  color=TEXT)  # fmt: skip
    ax.set_ylabel(f"Recall@{k}", color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset}: product quantization, memory vs. recall  ·  {hardware}",
                 color=TEXT, loc="left")  # fmt: skip
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    footer = "Rerank memory counts codes only (originals on disk). bench/plot_pq_memory.py"
    fig.text(0.99, 0.01, footer, ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    out = REPO_ROOT / "results" / "plots" / f"pq_memory_{dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return str(out.relative_to(REPO_ROOT))


def main() -> int:
    records = [r for r in latest_records(load_records()) if r["threads"] == 1]
    datasets = {r["dataset"]["name"] for r in records if r["index"] == "pq"}
    for dataset in sorted(datasets):
        pq = [r for r in records if r["dataset"]["name"] == dataset and r["index"] == "pq"]
        brute = next(
            (r for r in records if r["dataset"]["name"] == dataset and r["index"] == "brute_force"
             and r["build_params"].get("kernel") != "scalar"),
            None,
        )  # fmt: skip
        print("wrote", plot(dataset, pq, brute))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
