"""Download benchmark datasets and convert them to Strata's raw binary format.

Output format (same as big-ann-benchmarks .fbin/.ibin), little-endian:

    uint32 num_vectors
    uint32 dimension
    num_vectors * dimension values (float32 for .fbin, int32 for .ibin), row-major

The C++ code reads these with a single header read + one contiguous read (or mmap),
so it needs no HDF5 dependency.

Each dataset is written to data/<name>/:

    base.fbin          vectors to index
    query.fbin         query vectors
    groundtruth.ibin   exact nearest-neighbor ids for each query (row i = query i)
    meta.json          source URL, metric, shapes, SHA-256 of the downloaded file

Usage:
    uv run python scripts/prepare_datasets.py siftsmall
    uv run python scripts/prepare_datasets.py sift1m glove100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

HEADER_DTYPE = np.dtype("<u4")


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    kind: str  # "texmex" (tar.gz of .fvecs/.ivecs) or "hdf5" (ann-benchmarks)
    metric: str  # "l2" or "angular"
    texmex_prefix: str = ""


DATASETS: dict[str, Dataset] = {
    "siftsmall": Dataset(
        name="siftsmall",
        url="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        kind="texmex",
        metric="l2",
        texmex_prefix="siftsmall",
    ),
    "sift1m": Dataset(
        name="sift1m",
        url="http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        kind="hdf5",
        metric="l2",
    ),
    "glove100": Dataset(
        name="glove100",
        url="http://ann-benchmarks.com/glove-100-angular.hdf5",
        kind="hdf5",
        metric="angular",
    ),
}


# --- Binary format -------------------------------------------------------------------------


def write_bin(path: Path, array: npt.NDArray[np.float32] | npt.NDArray[np.int32]) -> None:
    """Write a 2-D float32 or int32 array with a (num_vectors, dimension) uint32 header."""
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    if array.dtype not in (np.float32, np.int32):
        raise ValueError(f"expected float32 or int32, got {array.dtype}")
    n, d = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.array([n, d], dtype=HEADER_DTYPE).tofile(f)
        np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")).tofile(f)


def read_bin(path: Path, dtype: npt.DTypeLike) -> npt.NDArray[np.generic]:
    """Read a file written by write_bin. dtype is np.float32 (.fbin) or np.int32 (.ibin)."""
    with path.open("rb") as f:
        n, d = (int(x) for x in np.fromfile(f, dtype=HEADER_DTYPE, count=2))
        data = np.fromfile(f, dtype=np.dtype(dtype).newbyteorder("<"), count=n * d)
    if data.size != n * d:
        raise ValueError(f"{path}: header says {n}x{d}, file holds {data.size} values")
    return data.reshape(n, d)


def read_xvecs(path: Path, dtype: npt.DTypeLike) -> npt.NDArray[np.generic]:
    """Read a TEXMEX .fvecs/.ivecs file: each row is int32 dimension followed by the values."""
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        raise ValueError(f"{path}: empty file")
    d = int(raw[0])
    if d <= 0 or raw.size % (d + 1) != 0:
        raise ValueError(f"{path}: size {raw.size} is not a multiple of row length {d + 1}")
    rows = raw.reshape(-1, d + 1)
    if not np.all(rows[:, 0] == d):
        raise ValueError(f"{path}: rows have inconsistent dimensions")
    return rows[:, 1:].copy().view(np.dtype(dtype).newbyteorder("<"))


# --- Download and convert ------------------------------------------------------------------


def download(url: str, dest: Path) -> str:
    """Download url to dest and return the SHA-256 hex digest."""
    print(f"downloading {url}", file=sys.stderr)
    sha = hashlib.sha256()
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        while chunk := response.read(1 << 20):
            sha.update(chunk)
            out.write(chunk)
    return sha.hexdigest()


def convert_texmex(archive: Path, prefix: str) -> dict[str, npt.NDArray[np.generic]]:
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        src = Path(tmp) / prefix
        return {
            "base": read_xvecs(src / f"{prefix}_base.fvecs", np.float32),
            "query": read_xvecs(src / f"{prefix}_query.fvecs", np.float32),
            "groundtruth": read_xvecs(src / f"{prefix}_groundtruth.ivecs", np.int32),
        }


def convert_hdf5(path: Path) -> dict[str, npt.NDArray[np.generic]]:
    import h5py  # imported lazily so the texmex path works without it

    with h5py.File(path, "r") as f:
        return {
            "base": np.asarray(f["train"], dtype=np.float32),
            "query": np.asarray(f["test"], dtype=np.float32),
            "groundtruth": np.asarray(f["neighbors"], dtype=np.int32),
        }


def prepare(dataset: Dataset, data_dir: Path, force: bool) -> Path:
    out_dir = data_dir / dataset.name
    if (out_dir / "meta.json").exists() and not force:
        print(f"{dataset.name}: already present in {out_dir} (use --force to rebuild)")
        return out_dir

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / Path(dataset.url).name
        sha256 = download(dataset.url, archive)
        if dataset.kind == "texmex":
            arrays = convert_texmex(archive, dataset.texmex_prefix)
        else:
            arrays = convert_hdf5(archive)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    write_bin(out_dir / "base.fbin", arrays["base"])
    write_bin(out_dir / "query.fbin", arrays["query"])
    write_bin(out_dir / "groundtruth.ibin", arrays["groundtruth"])

    meta = {
        "name": dataset.name,
        "source_url": dataset.url,
        "source_sha256": sha256,
        "metric": dataset.metric,
        "base_shape": list(arrays["base"].shape),
        "query_shape": list(arrays["query"].shape),
        "groundtruth_shape": list(arrays["groundtruth"].shape),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{dataset.name}: wrote {out_dir}")
    for key in ("base", "query", "groundtruth"):
        print(f"  {key:12s} {arrays[key].shape} {arrays[key].dtype}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="re-download and overwrite")
    args = parser.parse_args(argv)
    for name in args.names:
        prepare(DATASETS[name], args.data_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
