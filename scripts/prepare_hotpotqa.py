"""Download HotpotQA (distractor setting, dev split) and draw a fixed evaluation subset.

    uv run python scripts/prepare_hotpotqa.py --n 100 --seed 0

Source: Hugging Face dataset hotpotqa/hotpot_qa, pinned to REVISION (the same data as the
original hotpot_dev_distractor_v1.json: 7,405 questions, each with 10 paragraphs of which 2 are
gold, plus sentence-level supporting facts). Writes data/hotpotqa/:
    distractor_validation.parquet       the full dev split (SHA-256 recorded)
    subset-n<N>-seed<S>.json            the sampled questions
    subset-n<N>-seed<S>.meta.json       revision, checksum, sampling method, id list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "hotpotqa"
REPO = "hotpotqa/hotpot_qa"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
FILES = {
    "validation": ["distractor/validation-00000-of-00001.parquet"],
    "train": ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"],
}
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/" + FILES["validation"][0]


def download_split(split: str) -> list[tuple[Path, str]]:
    """Download (once) and checksum the parquet files of a HotpotQA split."""
    out = []
    for name in FILES[split]:
        path = OUT / name.replace("distractor/", "distractor_").replace("-00000-of-00001", "")
        if not path.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
            request = urllib.request.Request(url, headers={"User-Agent": "strata/0.1"})
            with urllib.request.urlopen(request) as response:
                path.write_bytes(response.read())
        out.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
    return out


def download() -> tuple[Path, str]:
    """The dev ("validation") split, as used by the distractor-setting evaluation."""
    return download_split("validation")[0]


def load_questions(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    questions = []
    for r in rows:
        facts = r["supporting_facts"]
        context = r["context"]
        questions.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "type": r["type"],
                "level": r["level"],
                "supporting_facts": [
                    [t, int(s)] for t, s in zip(facts["title"], facts["sent_id"], strict=True)
                ],
                "context": [
                    {"title": t, "sentences": list(s)}
                    for t, s in zip(context["title"], context["sentences"], strict=True)
                ],
            }
        )
    return questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    path, sha = download()
    questions = load_questions(path)
    rng = np.random.default_rng(args.seed)
    picked = sorted(rng.choice(len(questions), size=args.n, replace=False).tolist())
    subset = [questions[i] for i in picked]

    name = f"subset-n{args.n}-seed{args.seed}"
    (OUT / f"{name}.json").write_text(json.dumps(subset, indent=1) + "\n")
    meta = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "file": "distractor/validation-00000-of-00001.parquet",
        "sha256": sha,
        "total_questions": len(questions),
        "n": args.n,
        "seed": args.seed,
        "sampling": "numpy.random.default_rng(seed).choice(total, n, replace=False), sorted",
        "ids": [q["id"] for q in subset],
        "types": {t: sum(q["type"] == t for q in subset) for t in {q["type"] for q in subset}},
    }
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(questions)} dev questions; wrote {name} ({meta['types']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
