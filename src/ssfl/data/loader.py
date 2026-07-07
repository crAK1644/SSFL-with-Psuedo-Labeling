"""Fast, memory-mapped access to the mini-N-BaIoT cache.

This module is the only sanctioned data access path for the rest of the
project. It never touches the raw CSVs: the arrays inside ``cache/mini.npz``
are stored uncompressed, so each one is memory-mapped in place at its offset
within the zip archive. If the cache is missing, loading fails fast with build
instructions — it is never rebuilt implicitly (the build is an explicit,
minutes-long command).

Views:
- :func:`load_client`   per-client private data (X, y) for a scenario
- :func:`load_open`     the shared open set — X only, labels are not exposed
- :func:`load_test`     the test set (X, y)
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path("cache")
_BUILD_HINT = (
    "dataset cache not found at {path!s}: "
    "run `uv run python -m ssfl.data.build` first (the cache is never built implicitly)"
)


class CacheMissingError(FileNotFoundError):
    """Raised when cache/ has not been built yet."""


def _require(cache_dir: str | Path | None, filename: str) -> Path:
    cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    path = cache_dir / filename
    if not path.is_file():
        raise CacheMissingError(_BUILD_HINT.format(path=cache_dir))
    return path


def _memmap_npz_member(npz_path: Path, name: str) -> np.memmap:
    """Memory-map one uncompressed array inside an .npz without extraction."""
    member = name + ".npy"
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{npz_path}:{member} is compressed; cannot memory-map")
        header_offset = info.header_offset
    with open(npz_path, "rb") as fh:
        fh.seek(header_offset)
        local_header = fh.read(30)  # fixed part of the zip local file header
        name_len, extra_len = struct.unpack("<HH", local_header[26:30])
        fh.seek(header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(fh)
        if version != (1, 0):
            raise ValueError(f"unsupported .npy format version {version}")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fh)
        data_offset = fh.tell()
    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def load_arrays(
    cache_dir: str | Path | None = None,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    """Memory-mapped (X, y, device_id) for the whole mini dataset."""
    path = _require(cache_dir, "mini.npz")
    return (
        _memmap_npz_member(path, "X"),
        _memmap_npz_member(path, "y"),
        _memmap_npz_member(path, "device_id"),
    )


def load_meta(cache_dir: str | Path | None = None) -> dict:
    return json.loads(_require(cache_dir, "meta.json").read_text())


def load_splits(cache_dir: str | Path | None = None) -> dict:
    return json.loads(_require(cache_dir, "splits.json").read_text())


def load_partition(
    scenario: int, cache_dir: str | Path | None = None
) -> dict[str, list[int]]:
    """client_id -> private sample indices for one scenario."""
    return json.loads(_require(cache_dir, f"scenario_{scenario}.json").read_text())


def num_clients(scenario: int, cache_dir: str | Path | None = None) -> int:
    return len(load_partition(scenario, cache_dir))


def _split_indices(cache_dir: str | Path | None, split: str) -> list[int]:
    splits = load_splits(cache_dir)
    return sorted(i for entry in splits["subsets"].values() for i in entry[split])


def load_client(
    scenario: int, client_id: int, cache_dir: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One client's private (X, y) under the given scenario partition."""
    partition = load_partition(scenario, cache_dir)
    try:
        idx = partition[str(client_id)]
    except KeyError:
        raise KeyError(
            f"scenario {scenario} has clients 0..{len(partition) - 1}, got {client_id}"
        ) from None
    X, y, _ = load_arrays(cache_dir)
    return X[idx], y[idx]


def load_open(cache_dir: str | Path | None = None) -> np.ndarray:
    """The shared open set: X only. Open-split labels are never exposed (CON-6)."""
    X, _, _ = load_arrays(cache_dir)
    return X[_split_indices(cache_dir, "open")]


def load_test(cache_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The held-out test set (X, y)."""
    idx = _split_indices(cache_dir, "test")
    X, y, _ = load_arrays(cache_dir)
    return X[idx], y[idx]
