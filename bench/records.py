"""Result records shared by every benchmark driver (Strata, hnswlib, FAISS).

A record is one index build on one dataset plus a sweep of search-parameter points:

    {
      "timestamp", "git", "hardware",              # benchmeta.metadata()
      "dataset": {"name", "metric", ...},          # data/<name>/meta.json
      "library": "strata" | "hnswlib" | "faiss",
      "index": "brute_force" | "hnsw" | ...,
      "build_params": {...}, "build_seconds": float,
      "k", "threads", "num_queries", "command",
      "points": [{"search_params": {...}, "runs": [...], "summary": {...}}],
      "raw": {...}                                 # driver-specific raw output
    }
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from benchmeta import REPO_ROOT, metadata, timestamp_slug, write_new

SUMMARY_FIELDS = [
    "qps",
    "recall",
    "recall_by_id",
    "latency_mean_us",
    "latency_p50_us",
    "latency_p95_us",
    "latency_p99_us",
]

RESULTS_DIR = REPO_ROOT / "results" / "search"


def summarize(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return summarize_fields(runs, SUMMARY_FIELDS)


def summarize_fields(
    runs: list[dict[str, float]], fields: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """mean/median/stdev/min/max for each numeric field (all fields of the first run by default)."""
    fields = fields if fields is not None else list(runs[0])
    summary = {}
    for field in fields:
        values = [r[field] for r in runs if field in r]
        if not values:
            continue
        summary[field] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def save_record(
    *,
    dataset: str,
    library: str,
    index: str,
    build_params: dict[str, Any],
    build_seconds: float,
    k: int,
    threads: int,
    num_queries: int,
    command: list[str],
    points: list[dict[str, Any]],
    raw: dict[str, Any] | None = None,
    out_dir: Path = RESULTS_DIR,
) -> Path:
    dataset_meta = json.loads((REPO_ROOT / "data" / dataset / "meta.json").read_text())
    record = {
        **metadata(),
        "dataset": {**dataset_meta, "name": dataset},
        "library": library,
        "index": index,
        "build_params": build_params,
        "build_seconds": build_seconds,
        "k": k,
        "threads": threads,
        "num_queries": num_queries,
        "command": command,
        "points": [{**p, "summary": summarize(p["runs"])} for p in points],
        "raw": raw or {},
    }
    path = out_dir / dataset / f"{library}-{index}-{timestamp_slug()}.json"
    write_new(path, json.dumps(record, indent=2) + "\n")
    return path


def load_records(out_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("**/*.json"))]


def config_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Identifies a configuration; newer records with the same key supersede older ones."""
    return (
        record["dataset"]["name"],
        record["library"],
        record["index"],
        json.dumps(record["build_params"], sort_keys=True),
        record["k"],
        record["threads"],
        record["hardware"].get("cpu") or record["hardware"]["machine"],
    )


def latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in sorted(records, key=lambda r: r["timestamp"]):
        by_key[config_key(record)] = record
    return list(by_key.values())
