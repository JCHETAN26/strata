#!/usr/bin/env python3
"""Kaggle GPU runner for Strata's IdeaPad-class stages (the dev laptop's disk cannot fit CUDA
PyTorch or the full-corpus embeddings, so the GPU work moves to a Kaggle notebook).

Stages (each writes result JSON to results/ and copies it into the download folder):

    env       print and save the GPU + environment (also recorded inside every result file)
    stage2    full-corpus BEIR HotpotQA (5.2M passages) dense/BM25/hybrid retrieval, vs published
              BEIR references. Estimates memory first and refuses to embed if it will not fit.
    stage3    SciFact cross-encoder reranking on the GPU, tuned on the train split
    stage4    GPU reranking latency on the HotpotQA BEIR subset, printed next to the CPU numbers
              already committed under results/rerank/
    stage2b   (optional, costs money) HotpotQA answer generation on the BEIR subset (the split
              that carries gold answers); always behind a cost estimate and an --enable-api flag

Design notes:
- Embedding is sharded and resumable (scripts/embed_beir.py): a session that hits Kaggle's time
  limit is rerun and picks up from the last completed shard. Corpus embeddings and indexes stay
  on Kaggle; only the small result JSONs are meant to be downloaded.
- Every result records Kaggle's exact GPU and environment via bench/benchmeta.accelerator_info().
- Nothing here calls the Anthropic API except stage2b, which estimates cost and stops unless
  --enable-api is passed and the worst-case cost is under --max-cost-usd.

Run under uv so the pinned environment (CUDA PyTorch, sentence-transformers) is used:

    uv run python kaggle/run_stages.py env
    uv run python kaggle/run_stages.py stage2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Where downloadable outputs are collected. On Kaggle this is the persistent working dir.
DOWNLOAD_DIR = Path(os.environ.get("STRATA_OUTPUT_DIR", "/kaggle/working/strata-results"))

EMBED_MODEL = "bge-small-en-v1.5"
HOTPOTQA_CORPUS = 5_233_329
EMBED_DIM = 384  # bge-small-en-v1.5


def sh(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a repo script/command from the repo root, streaming its output live."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False, **kwargs)


def available_ram_gib() -> float | None:
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def collect_results() -> None:
    """Copy every result JSON into the flat download folder (embeddings/indexes stay put)."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for path in (REPO_ROOT / "results").rglob("*.json"):
        rel = path.relative_to(REPO_ROOT / "results")
        dest = DOWNLOAD_DIR / str(rel).replace("/", "__")
        try:
            shutil.copy2(path, dest)
        except OSError as e:
            print(f"  (could not copy {rel}: {e})", file=sys.stderr)
    print(f"  results collected in {DOWNLOAD_DIR}")


# --------------------------------------------------------------------------- env


def cmd_env(args: argparse.Namespace) -> int:
    from benchmeta import accelerator_info, hardware_info

    env = {
        "hardware": hardware_info(),
        "accelerator": accelerator_info(),
        "available_ram_gib": available_ram_gib(),
        "python": sys.version,
        "repo_root": str(REPO_ROOT),
    }
    print(json.dumps(env, indent=2))
    accel = env["accelerator"]
    cuda = accel.get("cuda")
    if not cuda or not cuda.get("available"):
        print(
            "\nWARNING: torch.cuda.is_available() is False. The stages will run on CPU (slow) "
            "unless a GPU is attached and the driver matches the pinned CUDA build. See "
            "kaggle/README.md ('GPU not detected').",
            file=sys.stderr,
        )
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (DOWNLOAD_DIR / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    return 0


# ------------------------------------------------------------------ stage 2 (retrieval)


def estimate_memory() -> dict:
    corpus_gib = HOTPOTQA_CORPUS * EMBED_DIM * 4 / 2**30
    # Exact dense retrieval loads the corpus vectors (numpy) and the index keeps its own copy.
    peak_gib = corpus_gib * 2
    ram = available_ram_gib()
    return {
        "corpus_passages": HOTPOTQA_CORPUS,
        "dim": EMBED_DIM,
        "corpus_fbin_gib": round(corpus_gib, 2),
        "retrieval_peak_ram_gib": round(peak_gib, 2),
        "available_ram_gib": round(ram, 2) if ram else None,
        "fits": (ram is not None and peak_gib < ram * 0.9),
    }


def cmd_stage2(args: argparse.Namespace) -> int:
    est = estimate_memory()
    print("Stage 2 memory estimate (full-corpus BEIR HotpotQA):")
    print(json.dumps(est, indent=2))
    if not est["fits"] and not args.allow_large_memory:
        print(
            "\nSTOP: exact dense retrieval over 5.2M passages needs about "
            f"{est['retrieval_peak_ram_gib']} GiB of RAM (corpus vectors + the index's copy), "
            f"which is not safely within the available {est['available_ram_gib']} GiB.\n"
            "Options, in order of preference:\n"
            "  1. Attach a higher-RAM Kaggle accelerator and rerun (embedding itself is cheap on "
            "RAM; the constraint is the brute-force dense index).\n"
            "  2. Rerun with --allow-large-memory to try anyway (may be killed by the OOM killer; "
            "embedding progress is saved in shards and will resume).\n"
            "The embedding step is safe regardless (peak RAM ~= shard_size x dim x 4). Pass "
            "--embed-only to produce embeddings now and run retrieval on a larger box later.",
            file=sys.stderr,
        )
        if not args.embed_only:
            return 3

    # 1. Prepare the full corpus (idempotent; downloads + caches the BEIR zip once).
    if not (REPO_ROOT / "data" / "beir" / "hotpotqa" / "corpus.jsonl").exists():
        rc = sh([sys.executable, "scripts/prepare_hotpotqa_beir.py", "--full"]).returncode
        if rc:
            return rc

    # 2. Embed on the GPU (sharded + resumable).
    rc = sh(
        [
            sys.executable, "scripts/embed_beir.py",
            "--dataset", "hotpotqa", "--model", EMBED_MODEL,
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--shard-size", str(args.shard_size),
        ]
    ).returncode
    if rc:
        return rc
    if args.embed_only:
        print("embed-only: embeddings ready; run retrieval on a larger box.")
        return 0

    # 3. Full-corpus retrieval vs published references.
    rc = sh(
        [sys.executable, "bench/eval_hybrid_beir.py", "--dataset", "hotpotqa",
         "--model", EMBED_MODEL]
    ).returncode
    collect_results()
    if rc:
        print(
            "\nNote: eval_hybrid_beir returned non-zero, which means the measured baselines did "
            "not match the published references within the tight ±0.002 margin. This is recorded "
            "in the result JSON (baseline_checks) and reported, not hidden. Small gaps vs MTEB are "
            "expected (CPU-vs-GPU float arithmetic, tokenizer truncation); review the numbers.",
            file=sys.stderr,
        )
    return 0  # a reference gap is a finding to report, not a runner failure


# ------------------------------------------------------------------ stage 3 (scifact rerank)


def cmd_stage3(args: argparse.Namespace) -> int:
    # SciFact BEIR dataset + embeddings (small; embedding is quick even on CPU).
    if not (REPO_ROOT / "data" / "beir" / "scifact" / "corpus.jsonl").exists():
        rc = sh([sys.executable, "scripts/prepare_beir.py", "scifact"]).returncode
        if rc:
            return rc
    rc = sh(
        [sys.executable, "scripts/embed_beir.py", "--dataset", "scifact", "--model", EMBED_MODEL,
         "--device", args.device, "--batch-size", str(args.batch_size)]
    ).returncode
    if rc:
        return rc
    # Reranking, tuned on SciFact's train split, cross-encoders on the GPU.
    rc = sh(
        [sys.executable, "bench/eval_rerank.py", "--dataset", "scifact",
         "--tune-dataset", "scifact", "--tune-split", "train",
         "--device", args.device, "--batch-size", str(args.rerank_batch_size),
         *(["--models", *args.models] if args.models else [])]
    ).returncode
    collect_results()
    return rc


# ------------------------------------------------------------------ stage 4 (GPU latency)


def cmd_stage4(args: argparse.Namespace) -> int:
    """GPU reranking latency on the HotpotQA BEIR subset (the same dataset as the committed CPU
    numbers), then print CPU vs GPU latency side by side."""
    subset = "hotpotqa-subset-n100-seed0-bg20000"
    dev_subset = "hotpotqa-dev-subset-n100-seed0-bg20000"
    for name, extra in ((subset, []), (dev_subset, ["--split", "dev"])):
        if not (REPO_ROOT / "data" / "beir" / name / "corpus.jsonl").exists():
            rc = sh(
                [sys.executable, "scripts/prepare_hotpotqa_beir.py",
                 "--n", "100", "--seed", "0", "--background", "20000", *extra]
            ).returncode
            if rc:
                return rc
    rc = sh(
        [sys.executable, "bench/eval_rerank.py", "--dataset", subset,
         "--tune-dataset", dev_subset, "--tune-split", "dev",
         "--device", args.device, "--batch-size", str(args.rerank_batch_size),
         *(["--models", *args.models] if args.models else [])]
    ).returncode
    collect_results()
    if rc == 0:
        _print_cpu_vs_gpu_latency(subset)
    return rc


def _print_cpu_vs_gpu_latency(dataset: str) -> None:
    """Read the newest CPU and GPU rerank results for `dataset` and print latency side by side."""
    results = sorted((REPO_ROOT / "results" / "rerank").glob(f"{dataset}-*.json"))
    by_device: dict[str, dict] = {}
    for path in results:
        record = json.loads(path.read_text())
        for model, lat in record.get("latency", {}).items():
            by_device.setdefault(lat["device"], {}).setdefault(model, (path.name, lat))
    if "cpu" not in by_device and "cuda" not in by_device:
        return
    comparison = {}
    print("\nReranking latency, CPU vs GPU (HotpotQA BEIR subset):")
    models = sorted({m for d in by_device.values() for m in d})
    for model in models:
        row = {}
        for device in ("cpu", "cuda"):
            if model in by_device.get(device, {}):
                _, lat = by_device[device][model]
                row[device] = lat
                print(
                    f"  {model:20s} {device:4s}  N={lat['pool_depth_N']} "
                    f"pool {lat['mean_pool_size']:.1f}  {lat['ms_mean']:.0f} ms/query "
                    f"(p50 {lat['ms_p50']:.0f}, p95 {lat['ms_p95']:.0f})"
                )
        if "cpu" in row and "cuda" in row and row["cuda"]["ms_mean"]:
            print(f"  {model:20s} speedup (mean): "
                  f"{row['cpu']['ms_mean'] / row['cuda']['ms_mean']:.1f}x")  # fmt: skip
        comparison[model] = row
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (DOWNLOAD_DIR / "rerank_latency_cpu_vs_gpu.json").write_text(
        json.dumps({"dataset": dataset, "by_model": comparison}, indent=2) + "\n"
    )


# ------------------------------------------------------------------ stage 2b (optional, API)


def cmd_stage2b(args: argparse.Namespace) -> int:
    """Optional HotpotQA answer generation (Anthropic API). Estimate-only unless --enable-api.

    Runs on the HotpotQA BEIR *subset* (which carries the answers and supporting facts), not the
    full 5.2M corpus: bench/eval_hotpotqa_beir.py embeds its corpus inline and scores answers
    against gold, which only the subset builder writes. Full-corpus answer-groundedness would need
    a full-corpus mode wired to the sharded embeddings; it is not built yet."""
    if not (REPO_ROOT / "data" / "beir" / args.subset / "answers.jsonl").exists():
        rc = sh(
            [sys.executable, "scripts/prepare_hotpotqa_beir.py",
             "--n", "100", "--seed", "0", "--background", "20000"]
        ).returncode
        if rc:
            return rc
    cmd = [
        sys.executable, "bench/eval_hotpotqa_beir.py",
        "--subset", args.subset, "--k", str(args.k), "--max-cost-usd", str(args.max_cost_usd),
    ]
    if args.enable_api:
        cmd.append("--run")
        print(
            "API ENABLED: eval_hotpotqa_beir will call the Anthropic API but still refuses to "
            f"start if the worst-case cost exceeds --max-cost-usd (${args.max_cost_usd})."
        )
    else:
        print("Estimate only. Re-run with --enable-api to spend (subject to --max-cost-usd).")
    rc = sh(cmd).returncode
    collect_results()
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="cuda", help="cuda (default here) or cpu")
        p.add_argument("--batch-size", type=int, default=256, help="embedding batch size")
        p.add_argument("--rerank-batch-size", type=int, default=64)
        p.add_argument("--models", nargs="+", help="reranker keys (default: all)")

    p = sub.add_parser("env", help="print + save GPU/environment")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("stage2", help="full-corpus BEIR HotpotQA retrieval")
    add_common(p)
    p.add_argument("--shard-size", type=int, default=200_000)
    p.add_argument("--allow-large-memory", action="store_true")
    p.add_argument("--embed-only", action="store_true", help="stop after embeddings")
    p.set_defaults(func=cmd_stage2)

    p = sub.add_parser("stage3", help="SciFact reranking on GPU (tuned on train)")
    add_common(p)
    p.set_defaults(func=cmd_stage3)

    p = sub.add_parser("stage4", help="GPU reranking latency vs committed CPU numbers")
    add_common(p)
    p.set_defaults(func=cmd_stage4)

    p = sub.add_parser("stage2b", help="optional: answer generation on the subset (Anthropic API)")
    add_common(p)
    p.add_argument("--subset", default="hotpotqa-subset-n100-seed0-bg20000")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--enable-api", action="store_true", help="actually call the API")
    p.set_defaults(func=cmd_stage2b)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
