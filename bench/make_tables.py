"""Generate markdown tables from saved results. No hand-copied numbers.

    uv run python bench/make_tables.py

Reads results/search/**/*.json and results/micro/*.json and writes results/tables.md.
For each configuration (see records.config_key) only the newest result is shown.
"""

from __future__ import annotations

import json
from typing import Any

from benchmeta import REPO_ROOT
from records import latest_records, load_records

RESULTS = REPO_ROOT / "results"


def fmt(summary: dict[str, float], digits: int) -> str:
    return f"{summary['mean']:.{digits}f} ± {summary['stdev']:.{digits}f}"


def machine(record: dict[str, Any]) -> str:
    return record["hardware"].get("cpu") or record["hardware"]["machine"]


def search_table() -> list[str]:
    records = latest_records(load_records())
    if not records:
        return []
    lines = [
        "## Search",
        "",
        "| Dataset | Library | Index | Build params | Search params | Machine | Threads | Runs "
        "| Recall@k | Recall@k by id | QPS | p50 µs | p99 µs | Build s | Commit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: (r["dataset"]["name"], r["library"], r["index"])):
        build = ", ".join(f"{k}={v}" for k, v in r["build_params"].items()) or "—"
        commit = r["git"]["commit"][:8] + ("*" if r["git"]["dirty"] else "")
        for point in r["points"]:
            s = point["summary"]
            search = ", ".join(f"{k}={v}" for k, v in point["search_params"].items()) or "—"
            lines.append(
                f"| {r['dataset']['name']} | {r['library']} | {r['index']} | {build} | {search} "
                f"| {machine(r)} | {r['threads']} | {len(point['runs'])} "
                f"| {s['recall']['mean']:.4f} (k={r['k']}) "
                f"| {s['recall_by_id']['mean'] if 'recall_by_id' in s else float('nan'):.4f} "
                f"| {fmt(s['qps'], 1)} | {fmt(s['latency_p50_us'], 1)} "
                f"| {fmt(s['latency_p99_us'], 1)} | {r['build_seconds']:.2f} | `{commit}` |"
            )
    notes = [
        "",
        "Recall@k is tie-aware (ann-benchmarks definition): a result counts if it is no farther",
        "than the k-th true neighbor. Recall by id is strict id matching; it can be lower when",
        "several vectors tie at the k-th distance. hnswlib/FAISS latencies include Python call",
        "overhead (their QPS does not). `*` = uncommitted changes when measured.",
        "",
    ]
    return lines + notes


def micro_table() -> list[str]:
    records = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("micro/*.json"))]
    if not records:
        return []
    lines = [
        "## Microbenchmarks",
        "",
        "| Benchmark | Machine | ns/op (mean ± stdev) | Commit |",
        "|---|---|---|---|",
    ]
    rows: dict[tuple[str, str], str] = {}
    for record in sorted(records, key=lambda r: r["timestamp"]):
        benches = record["benchmark"]["benchmarks"]
        stats = {
            (b["run_name"], b["aggregate_name"]): b["cpu_time"]
            for b in benches
            if b.get("run_type") == "aggregate"
        }
        commit = record["git"]["commit"][:8] + ("*" if record["git"]["dirty"] else "")
        for (name, agg), mean in stats.items():
            if agg != "mean":
                continue
            stdev = stats.get((name, "stddev"), 0.0)
            rows[(name, machine(record))] = (
                f"| {name} | {machine(record)} | {mean:.2f} ± {stdev:.2f} | `{commit}` |"
            )
    return [*lines, *(rows[key] for key in sorted(rows)), ""]


def storage_table() -> list[str]:
    records = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("storage/*.json"))]
    if not records:
        return []
    latest_by_machine: dict[tuple[str, str], dict[str, Any]] = {}
    for r in sorted(records, key=lambda r: r["timestamp"]):
        latest_by_machine[(r["dataset"], machine(r))] = r
    lines = [
        "## Storage (Collection: WAL + snapshot)",
        "",
        "| Dataset | Machine | Vectors | Inserts/s no sync | Inserts/s fsync | fsync p99 µs "
        "| Checkpoint ms | Recover from WAL ms | Recover from snapshot ms | Commit |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (dataset, mach), r in sorted(latest_by_machine.items()):
        s = r["summary"]
        commit = r["git"]["commit"][:8] + ("*" if r["git"]["dirty"] else "")
        ms = {k: {m: v * 1e3 for m, v in s[k].items()} for k in s if k.endswith("seconds")}
        lines.append(
            f"| {dataset} | {mach} | {r['result']['num_vectors']} "
            f"| {fmt(s['insert_nosync_ops'], 0)} | {fmt(s['insert_fsync_ops'], 0)} "
            f"| {fmt(s['insert_fsync_p99_us'], 0)} | {fmt(ms['checkpoint_seconds'], 1)} "
            f"| {fmt(ms['recovery_wal_seconds'], 1)} | {fmt(ms['recovery_snapshot_seconds'], 1)} "
            f"| `{commit}` |"
        )
    notes = [
        "",
        "fsync = `F_FULLFSYNC` on macOS (flushes the drive cache), `fsync` on Linux; one per",
        "insert, no group commit. The fsync column uses the first `fsync_inserts` vectors only.",
        "",
    ]
    return lines + notes


def main() -> int:
    body = ["# Results", "", "Generated by `bench/make_tables.py`. Do not edit by hand.", ""]
    body += search_table() + storage_table() + micro_table()
    out = RESULTS / "tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body))
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
