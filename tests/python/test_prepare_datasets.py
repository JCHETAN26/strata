from pathlib import Path

import numpy as np
import pytest
from prepare_datasets import read_bin, read_xvecs, write_bin


def write_xvecs(path: Path, array: np.ndarray) -> None:
    n, d = array.shape
    rows = np.empty((n, d + 1), dtype="<i4")
    rows[:, 0] = d
    rows[:, 1:] = array.view("<i4")
    rows.tofile(path)


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_bin_roundtrip(tmp_path: Path, dtype: type) -> None:
    array = np.arange(12, dtype=dtype).reshape(4, 3)
    path = tmp_path / "x.bin"
    write_bin(path, array)
    assert path.stat().st_size == 8 + array.nbytes
    np.testing.assert_array_equal(read_bin(path, dtype), array)


def test_bin_header_is_little_endian_uint32(tmp_path: Path) -> None:
    path = tmp_path / "x.fbin"
    write_bin(path, np.zeros((5, 7), dtype=np.float32))
    assert path.read_bytes()[:8] == (5).to_bytes(4, "little") + (7).to_bytes(4, "little")


def test_bin_empty_array(tmp_path: Path) -> None:
    path = tmp_path / "empty.fbin"
    write_bin(path, np.zeros((0, 128), dtype=np.float32))
    assert read_bin(path, np.float32).shape == (0, 128)


def test_write_bin_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_bin(tmp_path / "a", np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        write_bin(tmp_path / "b", np.zeros((2, 2), dtype=np.float64))


def test_read_bin_detects_truncation(tmp_path: Path) -> None:
    path = tmp_path / "x.fbin"
    write_bin(path, np.ones((4, 4), dtype=np.float32))
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="header says"):
        read_bin(path, np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_read_xvecs(tmp_path: Path, dtype: type) -> None:
    array = (np.arange(10, dtype=dtype) * 3).reshape(2, 5)
    path = tmp_path / "x.xvecs"
    write_xvecs(path, array)
    result = read_xvecs(path, dtype)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, array)


def test_read_xvecs_rejects_inconsistent_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "bad.fvecs"
    np.array([2, 0, 0, 3, 0, 0], dtype="<i4").tofile(path)
    with pytest.raises(ValueError):
        read_xvecs(path, np.float32)
