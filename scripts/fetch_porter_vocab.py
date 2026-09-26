"""Download Martin Porter's official stemmer test vocabulary into data/porter/.

    uv run python scripts/fetch_porter_vocab.py

voc.txt holds 23,531 words and output.txt their stems from Porter's reference implementation
(the same algorithm, with his later departures from the paper, that Lucene's PorterStemmer
ports). tests/python/test_bm25.py checks strata.porter_stem against every pair when present.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "porter"
BASE = "https://tartarus.org/martin/PorterStemmer/"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("voc.txt", "output.txt"):
        request = urllib.request.Request(BASE + name, headers={"User-Agent": "strata/0.1"})
        with urllib.request.urlopen(request) as response:
            (OUT / name).write_bytes(response.read())
        print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
