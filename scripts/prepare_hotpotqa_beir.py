"""Build a small BEIR-HotpotQA subset for developing the RAG evaluation on the Mac.

    uv run python scripts/prepare_hotpotqa_beir.py --n 100 --seed 0 --background 20000

The full BEIR HotpotQA corpus (5,233,329 Wikipedia abstracts) is for the IdeaPad. This subset keeps
the BEIR setting (BEIR corpus passages, BEIR queries, BEIR test qrels) at a Mac-friendly size:

- queries: n BEIR test queries, sampled with `seed` (BEIR's HotpotQA test split is HotpotQA's
  dev set: all 7,405 ids match the pinned Hugging Face copy, which supplies answers and
  sentence-level supporting facts);
- corpus: for each query, its 2 gold passages (qrels) and its 8 HotpotQA distractor passages
  (matched to BEIR documents by title; they were chosen by TF-IDF retrieval over full Wikipedia,
  so they are hard negatives), plus `background` further passages sampled uniformly from the
  whole corpus (a passage is kept when a seeded hash of its id falls below background / N).

Retrieval numbers on the subset are NOT comparable to full-corpus BEIR numbers: the corpus is
~200x smaller. They are for developing and checking the pipeline.

Sentences: gold and distractor passages carry HotpotQA's official sentence splits (their BEIR
text equals the joined sentences up to whitespace/NFKC normalization; checked for every one), so
cited sentence indices map exactly onto supporting facts. Background passages are split with a
simple regex; any citation of them is a false positive whatever the sentence index.

Writes data/beir/hotpotqa-subset-n<N>-seed<S>-bg<B>/: corpus.jsonl, queries.jsonl,
qrels/test.tsv, answers.jsonl, meta.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_hotpotqa import download_split, load_questions  # noqa: E402

# BEIR split -> the HotpotQA split its questions come from. BEIR's test split is HotpotQA's dev
# ("validation") set; BEIR's dev split is carved from HotpotQA's train set.
HOTPOTQA_SPLIT = {"test": "validation", "dev": "train"}

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip"
BEIR_ZIP = REPO_ROOT / "data" / "beir" / "hotpotqa.zip"
BEIR_SHA256 = "c62459dbc91d31329584507b1fa564b3b1abcee58800967113054ee0f5a15c62"
CORPUS_SIZE = 5_233_329
SENTENCE_END = re.compile(r"(?<=[.!?])(?=\s+[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """Regex sentence split; the pieces join back to `text` exactly."""
    pieces, start = [], 0
    for m in SENTENCE_END.finditer(text):
        pieces.append(text[start : m.end()])
        start = m.end()
    pieces.append(text[start:])
    return [p for p in pieces if p]


def normalized(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).split())


def ensure_zip() -> None:
    if not BEIR_ZIP.exists():
        BEIR_ZIP.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(BEIR_URL, headers={"User-Agent": "strata/0.1"})
        with urllib.request.urlopen(request) as response, BEIR_ZIP.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    sha = hashlib.sha256()
    with BEIR_ZIP.open("rb") as f:
        while chunk := f.read(1 << 24):
            sha.update(chunk)
    if sha.hexdigest() != BEIR_SHA256:
        raise SystemExit(f"{BEIR_ZIP}: unexpected SHA-256 {sha.hexdigest()}")


def keep_background(doc_id: str, seed: int, fraction: float) -> bool:
    h = hashlib.sha1(f"{seed}:{doc_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < fraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--split",
        default="test",
        choices=sorted(HOTPOTQA_SPLIT),
        help="BEIR split to sample queries from (dev is for tuning)",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--background", type=int, default=20_000)
    args = parser.parse_args(argv)

    ensure_zip()
    hf_files = download_split(HOTPOTQA_SPLIT[args.split])
    hf = {q["id"]: q for path, _ in hf_files for q in load_questions(path)}
    hf_sha = {path.name: sha for path, sha in hf_files}
    z = zipfile.ZipFile(BEIR_ZIP)

    qrels: dict[str, dict[str, int]] = {}
    with z.open(f"hotpotqa/qrels/{args.split}.tsv") as f:
        reader = csv.reader(io.TextIOWrapper(f), delimiter="\t")
        next(reader)
        for qid, doc, score in reader:
            qrels.setdefault(qid, {})[doc] = int(score)
    if not set(qrels) <= set(hf):
        raise SystemExit(
            f"BEIR {args.split} queries are not all in HotpotQA {HOTPOTQA_SPLIT[args.split]}"
        )

    rng = np.random.default_rng(args.seed)
    all_ids = sorted(qrels)
    qids = sorted(all_ids[i] for i in rng.choice(len(all_ids), size=args.n, replace=False))
    gold_ids = {d for q in qids for d in qrels[q]}
    # Official sentence splits for every paragraph HotpotQA gave these questions.
    official: dict[str, list[str]] = {}
    for q in qids:
        for c in hf[q]["context"]:
            official[c["title"]] = c["sentences"]

    corpus: dict[str, dict] = {}
    fraction = args.background / CORPUS_SIZE
    scanned = 0
    with z.open("hotpotqa/corpus.jsonl") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            scanned += 1
            d = json.loads(line)
            if d["_id"] in gold_ids:
                source = "gold"
            elif d["title"] in official:
                source = "distractor"
            elif keep_background(d["_id"], args.seed, fraction):
                source = "background"
            else:
                continue
            sentences = official.get(d["title"])
            if sentences is not None and normalized("".join(sentences)) != normalized(d["text"]):
                raise SystemExit(f"{d['_id']} ({d['title']}): text differs from HotpotQA sentences")
            corpus[d["_id"]] = {
                "_id": d["_id"],
                "title": d["title"],
                "text": d["text"],
                "sentences": sentences if sentences is not None else split_sentences(d["text"]),
                "sentence_source": "hotpotqa" if sentences is not None else "regex",
                "source": source,
            }
    if scanned != CORPUS_SIZE:
        raise SystemExit(f"corpus has {scanned} passages, expected {CORPUS_SIZE}")
    missing = gold_ids - set(corpus)
    if missing:
        raise SystemExit(f"gold passages missing from the corpus: {sorted(missing)[:5]}")

    queries = {}
    with z.open("hotpotqa/queries.jsonl") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            q = json.loads(line)
            if q["_id"] in qrels and q["_id"] in set(qids):
                queries[q["_id"]] = q

    prefix = "hotpotqa" if args.split == "test" else f"hotpotqa-{args.split}"
    name = f"{prefix}-subset-n{args.n}-seed{args.seed}-bg{args.background}"
    out = REPO_ROOT / "data" / "beir" / name
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w") as f:
        for doc_id in sorted(corpus):
            f.write(json.dumps(corpus[doc_id]) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for q in qids:
            f.write(json.dumps(queries[q]) + "\n")
    with (out / "qrels" / f"{args.split}.tsv").open("w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in qids:
            for d, s in sorted(qrels[q].items()):
                f.write(f"{q}\t{d}\t{s}\n")
    with (out / "answers.jsonl").open("w") as f:
        for q in qids:
            h = hf[q]
            f.write(
                json.dumps(
                    {
                        "_id": q,
                        "answer": h["answer"],
                        "type": h["type"],
                        "level": h["level"],
                        "supporting_facts": h["supporting_facts"],
                    }
                )
                + "\n"
            )
    counts = {
        s: sum(d["source"] == s for d in corpus.values())
        for s in ("gold", "distractor", "background")
    }
    meta = {
        "setting": f"BEIR HotpotQA ({args.split} split), subset corpus",
        "split": args.split,
        "beir_source": BEIR_URL,
        "beir_sha256": BEIR_SHA256,
        "beir_corpus_size": CORPUS_SIZE,
        "hotpotqa_source": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        "hotpotqa_revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "hotpotqa_split": HOTPOTQA_SPLIT[args.split],
        "hotpotqa_sha256": hf_sha,
        "n": args.n,
        "seed": args.seed,
        "background_target": args.background,
        "sampling": "queries: numpy default_rng(seed).choice over sorted test ids; background: "
        "sha1(f'{seed}:{id}') < background / corpus size",
        "corpus_passages": len(corpus),
        "corpus_by_source": counts,
        "comparable_to_full_beir": False,
        "sentence_splits": "HotpotQA official for gold/distractor passages; regex for background",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}: {len(qids)} queries, {len(corpus)} passages {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
