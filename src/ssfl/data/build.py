"""One-time mini-N-BaIoT cache builder.

Turns the raw N-BaIoT CSVs (``data/*.csv``, 89 device-category traffic files)
into a deterministic, cached dataset under ``cache/``:

- ``mini.npz``      X float32 [N, 23, 5], y int64 [N], device_id int8 [N]
- ``splits.json``   per-subset 70/10/20 private/open/test global index lists
- ``meta.json``     seed, global 11-class map, per-device class counts,
                    raw-data content hash
- ``scenario_<s>.json``  client_id -> private sample indices, s in {1, 2, 3}

Judgment calls (ADR-8): mini-N-BaIoT sampling is the *first* ``rows_per_subset``
rows of each device-category file; min-max normalization is global per feature
over the mini dataset; the 23x5 reshape follows Eq. 19's column layout
(features 0-22 -> column 0, 23-45 -> column 1, ...).

Building is an explicit command::

    uv run python -m ssfl.data.build

Consumers (see ``ssfl.data.loader``) never rebuild implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

#: Global 11-class map — fixed for the whole project, independent of which
#: classes a given device actually has (devices 3 and 7 lack mirai traffic).
CANONICAL_CLASSES = [
    "benign",
    "gafgyt.combo",
    "gafgyt.junk",
    "gafgyt.scan",
    "gafgyt.tcp",
    "gafgyt.udp",
    "mirai.ack",
    "mirai.scan",
    "mirai.syn",
    "mirai.udp",
    "mirai.udpplain",
]
CLASS_MAP: dict[str, int] = {cat: i for i, cat in enumerate(CANONICAL_CLASSES)}

N_FEATURES = 115
RESHAPE_ROWS, RESHAPE_COLS = 23, 5  # Eq. 19
SPLIT_FRACTIONS = (0.7, 0.1)  # private, open; test is the remainder

#: Traffic files look like ``<device>.<category>.csv``; the metadata files
#: (data_summary.csv, device_info.csv, features.csv) do not match.
_TRAFFIC_FILE_RE = re.compile(r"^(\d+)\.(benign|gafgyt\.[a-z]+|mirai\.[a-z]+)\.csv$")


def discover_subsets(data_dir: Path) -> list[tuple[int, str, Path]]:
    """Enumerate (device, category, path) traffic subsets in deterministic order."""
    subsets = []
    for path in sorted(data_dir.glob("*.csv")):
        m = _TRAFFIC_FILE_RE.match(path.name)
        if m is None:
            continue  # metadata file (data_summary / device_info / features)
        device, category = int(m.group(1)), m.group(2)
        if category not in CLASS_MAP:
            raise ValueError(f"unknown traffic category in {path.name!r}")
        subsets.append((device, category, path))
    if not subsets:
        raise FileNotFoundError(f"no traffic CSVs found in {data_dir}")
    subsets.sort(key=lambda t: (t[0], t[1]))
    return subsets


def _write_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """np.savez replacement with fixed zip timestamps -> byte-identical rebuilds.

    Members are stored uncompressed so loaders can memory-map them in place.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr))
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())


def _write_json_deterministic(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, indent=1) + "\n")


def _read_subset(path: Path, rows: int) -> np.ndarray:
    """Read the first ``rows`` data rows of one traffic CSV (opened read-only)."""
    with open(path, "r", newline="") as fh:
        df = pd.read_csv(fh, nrows=rows, dtype=np.float64)
    if df.shape[1] != N_FEATURES:
        raise ValueError(f"{path.name}: expected {N_FEATURES} features, got {df.shape[1]}")
    return df.to_numpy(dtype=np.float64)


def _min_max_normalize(raw: np.ndarray) -> np.ndarray:
    """Global per-feature min-max to [0, 1]; constant features map to 0."""
    lo = raw.min(axis=0)
    span = raw.max(axis=0) - lo
    span[span == 0.0] = 1.0
    return (raw - lo) / span


def _reshape_eq19(flat: np.ndarray) -> np.ndarray:
    """[N, 115] -> [N, 23, 5]: features 0-22 fill column 0, 23-45 column 1, ..."""
    n = flat.shape[0]
    return flat.reshape(n, RESHAPE_COLS, RESHAPE_ROWS).transpose(0, 2, 1)


def _make_splits(
    subset_rows: list[tuple[int, str, int]], rng: np.random.Generator
) -> dict:
    """70/10/20 private/open/test split per subset; global, mutually disjoint indices."""
    subsets: dict[str, dict] = {}
    start = 0
    for device, category, n in subset_rows:
        perm = start + rng.permutation(n)
        n_priv = round(SPLIT_FRACTIONS[0] * n)
        n_open = round(SPLIT_FRACTIONS[1] * n)
        subsets[f"{device}.{category}"] = {
            "private": sorted(int(i) for i in perm[:n_priv]),
            "open": sorted(int(i) for i in perm[n_priv : n_priv + n_open]),
            "test": sorted(int(i) for i in perm[n_priv + n_open :]),
        }
        start += n
    return {"subsets": subsets}


def build(
    data_dir: str | Path = "data",
    cache_dir: str | Path = "cache",
    seed: int = 0,
    rows_per_subset: int = 1000,
) -> Path:
    """Build the mini-N-BaIoT cache. All randomness derives from ``seed``."""
    from ssfl.data.partition import make_scenarios

    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    subsets = discover_subsets(data_dir)

    hasher = hashlib.sha256()
    blocks, labels, devices, subset_rows = [], [], [], []
    per_device_counts: dict[str, dict[str, int]] = {}
    for device, category, path in subsets:
        block = _read_subset(path, rows_per_subset)
        hasher.update(f"{device}.{category}:{block.shape[0]}".encode())
        hasher.update(block.tobytes())
        blocks.append(block)
        labels.append(np.full(block.shape[0], CLASS_MAP[category], dtype=np.int64))
        devices.append(np.full(block.shape[0], device, dtype=np.int8))
        subset_rows.append((device, category, block.shape[0]))
        per_device_counts.setdefault(str(device), {})[category] = int(block.shape[0])

    X = _reshape_eq19(_min_max_normalize(np.concatenate(blocks))).astype(np.float32)
    y = np.concatenate(labels)
    device_id = np.concatenate(devices)

    # Independent, seed-derived RNG streams: splits get one, scenarios the rest.
    split_seq, scenario_seq = np.random.SeedSequence(seed).spawn(2)
    splits = _make_splits(subset_rows, np.random.default_rng(split_seq))
    scenarios = make_scenarios(splits, y, device_id, scenario_seq)

    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_npz_deterministic(cache_dir / "mini.npz", {"X": X, "y": y, "device_id": device_id})
    _write_json_deterministic(cache_dir / "splits.json", splits)
    for s, clients in scenarios.items():
        _write_json_deterministic(cache_dir / f"scenario_{s}.json", clients)
    _write_json_deterministic(
        cache_dir / "meta.json",
        {
            "seed": seed,
            "rows_per_subset": rows_per_subset,
            "class_map": CLASS_MAP,
            "per_device_class_counts": per_device_counts,
            "raw_data_sha256": hasher.hexdigest(),
            "num_samples": int(X.shape[0]),
            "num_subsets": len(subsets),
        },
    )
    return cache_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ssfl.data.build",
        description="Build the mini-N-BaIoT cache from raw data/*.csv (one-time, explicit).",
    )
    parser.add_argument("--data-dir", default="data", help="raw CSV directory (read-only)")
    parser.add_argument("--cache-dir", default="cache", help="output cache directory")
    parser.add_argument("--seed", type=int, default=0, help="seed for all randomness")
    parser.add_argument(
        "--rows-per-subset", type=int, default=1000,
        help="rows taken from the head of each device-category file",
    )
    args = parser.parse_args(argv)
    cache = build(args.data_dir, args.cache_dir, args.seed, args.rows_per_subset)
    print(f"cache built at {cache.resolve()}")


if __name__ == "__main__":
    main()
