"""Shared helpers for the mini-N-BaIoT data-core tests.

Builds a small synthetic replica of the raw N-BaIoT layout: 89 traffic CSVs
(devices 3 and 7 have no mirai files) plus the 3 metadata CSVs that the
builder must ignore.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_FEATURES = 115
ROWS_PER_SUBSET = 50  # small stand-in for the real 1000
EXTRA_ROWS = 5  # rows beyond the cutoff that the builder must never read

CATEGORIES = [
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
NO_MIRAI_DEVICES = {3, 7}


def subset_names() -> list[tuple[int, str]]:
    """The 89 (device, category) subsets, in build order."""
    out = []
    for device in range(1, 10):
        for cat in CATEGORIES:
            if device in NO_MIRAI_DEVICES and cat.startswith("mirai"):
                continue
            out.append((device, cat))
    return out


def make_raw_data(data_dir: Path, rows: int = ROWS_PER_SUBSET) -> None:
    """Write the 89 synthetic traffic CSVs + 3 metadata CSVs into data_dir.

    Layout guarantees per traffic file:
      row 0 = all zeros, row 1 = all ones  (per-feature min/max anchors, so
              global min-max normalization is the identity on the rest)
      1.benign.csv row 2 = arange(115)/114 (reshape probe row)
      rows beyond `rows` contain the sentinel 5.0 (must never be read).
    """
    rng = np.random.default_rng(123)
    header = ",".join(f"f{i}" for i in range(N_FEATURES))
    for device, cat in subset_names():
        vals = rng.uniform(size=(rows + EXTRA_ROWS, N_FEATURES))
        vals[0] = 0.0
        vals[1] = 1.0
        if device == 1 and cat == "benign":
            vals[2] = np.arange(N_FEATURES) / (N_FEATURES - 1)
        vals[rows:] = 5.0  # sentinel: reading past the cutoff is detectable
        lines = [header] + [",".join(repr(float(v)) for v in row) for row in vals]
        (data_dir / f"{device}.{cat}.csv").write_text("\n".join(lines) + "\n")

    # Metadata files that must be excluded from the build.
    (data_dir / "features.csv").write_text(
        "Feature Name,Feature Description\nf0,dummy feature\n"
    )
    (data_dir / "data_summary.csv").write_text(
        "File Name, Data Count, Feature Count\n1.benign.csv,100,115\n"
    )
    (data_dir / "device_info.csv").write_text("DeviceID,DeviceName\n1,Danmini_Doorbell\n")


def build_fixture(tmp_path_factory, seed: int = 7):
    """Create raw data + build the cache once; returns (data_dir, cache_dir, seed)."""
    from ssfl.data.build import build

    root = tmp_path_factory.mktemp("mini_nbaiot")
    data_dir = root / "data"
    data_dir.mkdir()
    make_raw_data(data_dir)
    cache_dir = root / "cache"
    build(data_dir=data_dir, cache_dir=cache_dir, seed=seed, rows_per_subset=ROWS_PER_SUBSET)
    return data_dir, cache_dir, seed
