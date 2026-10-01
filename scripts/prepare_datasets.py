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
    uv run python scripts/prepare_datasets.py bigann10m      # AWS: 1.3 GB download, 5.1 GB on disk

bigann10m is the first 10M vectors of BIGANN (SIFT1B), the big-ann-benchmarks "BIGANN-10M" set:
the base is read with an HTTP range request from the 1B-vector file (uint8, converted to float32),
with its public 10K queries and published 10M ground truth, which is spot-checked here against an
exact search for a few queries.
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
from http.client import HTTPResponse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

HEADER_DTYPE = np.dtype("<u4")


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    kind: str  # "texmex" (tar.gz of .fvecs/.ivecs), "hdf5" (ann-benchmarks), or "bigann"
    metric: str  # "l2" or "angular"
    texmex_prefix: str = ""
    # kind == "bigann": the base is the first num_base rows of `url` (u8bin), plus these files.
    query_url: str = ""
    groundtruth_url: str = ""
    num_base: int = 0


BIGANN = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks"


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
    "bigann10m": Dataset(
        name="bigann10m",
        url=f"{BIGANN}/bigann/base.1B.u8bin",
        kind="bigann",
        metric="l2",
        query_url=f"{BIGANN}/bigann/query.public.10K.u8bin",
        groundtruth_url=f"{BIGANN}/GT_10M/bigann-10M",
        num_base=10_000_000,
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
    # ann-benchmarks.com returns 403 to urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "strata-dataset-fetch/0.1"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as out:
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


def _open(url: str, byte_range: tuple[int, int] | None = None) -> HTTPResponse:
    headers = {"User-Agent": "strata-dataset-fetch/0.1"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def _read_u8bin_header(url: str) -> tuple[int, int]:
    with _open(url, (0, 7)) as response:
        n, d = np.frombuffer(response.read(), dtype=HEADER_DTYPE)
    return int(n), int(d)


def prepare_bigann(dataset: Dataset, out_dir: Path, verify_queries: int = 20) -> dict[str, Any]:
    """Streams the first num_base rows of the 1B-vector base (uint8) into base.fbin as float32,
    in chunks, so memory stays at a few hundred MB; then queries and ground truth."""
    total, dim = _read_u8bin_header(dataset.url)
    n = dataset.num_base
    if n > total:
        raise ValueError(f"{dataset.url} holds {total} vectors, fewer than {n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    rows_per_chunk = 1 << 16
    print(f"downloading {n:,} x {dim} bytes from {dataset.url}", file=sys.stderr)
    with (
        _open(dataset.url, (8, 8 + n * dim - 1)) as response,
        (out_dir / "base.fbin").open("wb") as out,
    ):
        np.array([n, dim], dtype=HEADER_DTYPE).tofile(out)
        done = 0
        while done < n:
            rows = min(rows_per_chunk, n - done)
            raw = response.read(rows * dim)
            if len(raw) != rows * dim:
                raise OSError(f"short read at row {done}: expected {rows * dim} bytes")
            sha.update(raw)
            np.frombuffer(raw, dtype=np.uint8).astype("<f4").tofile(out)
            done += rows
            if done % (1 << 20) < rows_per_chunk:
                print(f"  {done:,} / {n:,}", file=sys.stderr, flush=True)

    with _open(dataset.query_url) as response:
        q = np.frombuffer(response.read(), dtype=np.uint8, offset=8)
    nq, qd = _read_u8bin_header(dataset.query_url)
    queries = q.reshape(nq, qd).astype(np.float32)
    write_bin(out_dir / "query.fbin", queries)

    # big-ann-benchmarks ground truth: header (nq, k), nq*k uint32 ids, then nq*k float32 distances.
    with _open(dataset.groundtruth_url) as response:
        blob = response.read()
    gq, gk = (int(x) for x in np.frombuffer(blob[:8], dtype=HEADER_DTYPE))
    ids = np.frombuffer(blob, dtype="<u4", count=gq * gk, offset=8).reshape(gq, gk)
    distances = np.frombuffer(blob, dtype="<f4", count=gq * gk, offset=8 + 4 * gq * gk)
    groundtruth = ids.astype(np.int32)
    write_bin(out_dir / "groundtruth.ibin", groundtruth)

    _spot_check_groundtruth(out_dir, queries, ids, distances.reshape(gq, gk), verify_queries)
    return {
        "source_sha256": sha.hexdigest(),
        "base_shape": [n, dim],
        "query_shape": list(queries.shape),
        "groundtruth_shape": list(groundtruth.shape),
        "query_url": dataset.query_url,
        "groundtruth_url": dataset.groundtruth_url,
        "source_rows": f"first {n} of {total}",
        "groundtruth_spot_check_queries": verify_queries,
    }


def _spot_check_groundtruth(
    out_dir: Path,
    queries: npt.NDArray[np.float32],
    ids: npt.NDArray[np.uint32],
    distances: npt.NDArray[np.float32],
    count: int,
    k: int = 10,
) -> None:
    """Exact search for the first `count` queries over base.fbin (memory-mapped, in blocks); the
    k-th distance must match the published ground truth (squared L2, so integer-exact for BIGANN).
    Comparing distances rather than ids tolerates ties."""
    if count == 0:
        return
    base = np.memmap(out_dir / "base.fbin", dtype="<f4", mode="r", offset=8)
    n, dim = (int(x) for x in np.fromfile(out_dir / "base.fbin", dtype=HEADER_DTYPE, count=2))
    base = base.reshape(n, dim)
    q = queries[:count].astype(np.float64)
    best = np.full((count, k), np.inf)
    block = 1 << 20
    for start in range(0, n, block):
        chunk = np.asarray(base[start : start + block], dtype=np.float64)
        d = (q**2).sum(1)[:, None] - 2 * q @ chunk.T + (chunk**2).sum(1)[None, :]
        best = np.sort(np.concatenate([best, np.partition(d, k, axis=1)[:, :k]], axis=1), 1)[:, :k]
    exact_kth = np.round(best[:, k - 1])
    published_kth = np.round(distances[:count, k - 1].astype(np.float64))
    if not np.array_equal(exact_kth, published_kth):
        bad = np.nonzero(exact_kth != published_kth)[0]
        raise ValueError(f"ground truth disagrees with exact search for queries {bad.tolist()}")
    print(f"  ground truth matches exact search on {count} queries", file=sys.stderr)


def prepare(dataset: Dataset, data_dir: Path, force: bool) -> Path:
    out_dir = data_dir / dataset.name
    if (out_dir / "meta.json").exists() and not force:
        print(f"{dataset.name}: already present in {out_dir} (use --force to rebuild)")
        return out_dir

    if dataset.kind == "bigann":
        if out_dir.exists():
            shutil.rmtree(out_dir)
        info = prepare_bigann(dataset, out_dir)
        meta = {"name": dataset.name, "source_url": dataset.url, "metric": dataset.metric, **info}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{dataset.name}: wrote {out_dir}")
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
