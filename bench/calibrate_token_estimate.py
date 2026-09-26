"""Fit the offline input-token estimator to exact counts from a HotpotQA (BEIR) run.

    uv run python bench/calibrate_token_estimate.py [results/rag/<run>.json]

Model: tokens = per_request + per_char * characters, where characters is request_chars() (system
prompt, document titles and sentences, question). Fitted by least squares on the exact
count_tokens values recorded in the run's cost estimate. Reports the fit, a cross-condition check
(fit on one condition, predict the other), the old estimator's error, and how the exact counts
compare with the input tokens the API actually billed (usage.input_tokens).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from benchmeta import REPO_ROOT


def fit(chars: np.ndarray, tokens: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(chars), chars])
    (intercept, slope), *_ = np.linalg.lstsq(design, tokens, rcond=None)
    return float(intercept), float(slope)


def error(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    rel = (pred - true) / true
    return {
        "mean_abs_pct": float(np.mean(np.abs(rel)) * 100),
        "max_abs_pct": float(np.max(np.abs(rel)) * 100),
        "total_pct": float((pred.sum() - true.sum()) / true.sum() * 100),
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = (
        Path(argv[0])
        if argv
        else max((REPO_ROOT / "results" / "rag").glob("hotpotqa-subset-*[0-9].json"))
    )
    record = json.loads(path.read_text())
    rows = {}
    for cond, est in record["cost_estimate"].items():
        billed = {r["id"]: r["input_tokens"] for r in record["results"][cond]["records"]}
        for qid, r in est["per_request"].items():
            if r["method"] != "count_tokens":
                raise SystemExit(f"{path.name}: {cond} was not counted exactly")
            rows[(cond, qid)] = (r["chars"], r["input_tokens"], r["offline_estimate"], billed[qid])
    conds = sorted({c for c, _ in rows})
    arr = {c: np.array([v for (cc, _), v in rows.items() if cc == c], dtype=float) for c in conds}
    everything = np.vstack(list(arr.values()))
    chars, exact, old, billed = everything.T

    intercept, slope = fit(chars, exact)
    new = intercept + slope * chars
    out = {
        "run": path.name,
        "requests": len(rows),
        "fit": {
            "per_request_tokens": intercept,
            "tokens_per_char": slope,
            "chars_per_token": 1 / slope,
        },
        "old_estimator_vs_exact": error(old, exact),
        "new_estimator_vs_exact": error(new, exact),
        "exact_vs_billed": {
            **error(exact, billed),
            "identical_requests": int(np.sum(exact == billed)),
        },
        "cross_condition": {},
    }
    for train in conds:
        for test in conds:
            if train == test:
                continue
            a, b = fit(arr[train][:, 0], arr[train][:, 1])
            out["cross_condition"][f"fit_{train}_predict_{test}"] = error(
                a + b * arr[test][:, 0], arr[test][:, 1]
            )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
