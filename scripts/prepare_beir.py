"""Download BEIR datasets into data/beir/<name>/.

    uv run python scripts/prepare_beir.py scifact nfcorpus fiqa

Each dataset directory holds the files as BEIR distributes them:
    corpus.jsonl        {"_id", "title", "text"} per line
    queries.jsonl       {"_id", "text"} per line
    qrels/test.tsv      query-id, corpus-id, score (with a header row)
plus meta.json with the source URL and the SHA-256 of the downloaded zip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BEIR_DIR = REPO_ROOT / "data" / "beir"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATASETS = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"]


def prepare(name: str, force: bool) -> Path:
    out = BEIR_DIR / name
    if (out / "meta.json").exists() and not force:
        print(f"{name}: already present in {out}")
        return out
    url = URL.format(name=name)
    print(f"downloading {url}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}.zip"
        sha = hashlib.sha256()
        request = urllib.request.Request(url, headers={"User-Agent": "strata-dataset-fetch/0.1"})
        with urllib.request.urlopen(request) as response, archive.open("wb") as f:
            while chunk := response.read(1 << 20):
                sha.update(chunk)
                f.write(chunk)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(Path(tmp) / name), out)
    (out / "meta.json").write_text(
        json.dumps({"name": name, "source_url": url, "source_sha256": sha.hexdigest()}, indent=2)
        + "\n"
    )
    print(f"{name}: wrote {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="+", choices=DATASETS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for name in args.names:
        prepare(name, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
