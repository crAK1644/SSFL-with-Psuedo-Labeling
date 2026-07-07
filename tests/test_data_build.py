"""Tests for the mini-N-BaIoT cache builder (ssfl.data.build)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from helpers_data import (
    CATEGORIES,
    NO_MIRAI_DEVICES,
    ROWS_PER_SUBSET,
    subset_names,
)
from helpers_data import build_fixture

N_SUBSETS = 89
N_TOTAL = N_SUBSETS * ROWS_PER_SUBSET


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_fixture(tmp_path_factory)


@pytest.fixture(scope="module")
def arrays(built):
    _, cache_dir, _ = built
    with np.load(cache_dir / "mini.npz") as npz:
        return {k: npz[k] for k in npz.files}


def test_cache_files_exist(built):
    _, cache_dir, _ = built
    for name in ["mini.npz", "splits.json", "meta.json"]:
        assert (cache_dir / name).exists(), f"missing cache/{name}"


def test_array_shapes_and_dtypes(arrays):
    X, y, device_id = arrays["X"], arrays["y"], arrays["device_id"]
    assert X.shape == (N_TOTAL, 23, 5)
    assert X.dtype == np.float32
    assert y.shape == (N_TOTAL,)
    assert y.dtype == np.int64
    assert device_id.shape == (N_TOTAL,)
    assert device_id.dtype == np.int8


def test_metadata_csvs_excluded_and_rows_capped(arrays):
    # 89 traffic subsets x ROWS_PER_SUBSET rows: nothing more (metadata files
    # skipped), nothing past the per-file cutoff (sentinel rows never read).
    assert arrays["X"].shape[0] == N_TOTAL


def test_values_min_max_normalized_to_unit_range(arrays):
    X = arrays["X"]
    assert np.isfinite(X).all()
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_reshape_follows_eq19_column_layout(arrays):
    # Global row 2 is 1.benign row 2, whose raw features are k/114 and whose
    # per-feature min/max anchors make normalization the identity.
    # Eq. 19: features 0-22 -> column 0, 23-45 -> column 1, ...
    probe = arrays["X"][2]
    expected = np.arange(115, dtype=np.float64).reshape(5, 23).T / 114.0
    np.testing.assert_allclose(probe, expected.astype(np.float32), atol=1e-6)


def test_labels_follow_global_class_map(built, arrays):
    _, cache_dir, _ = built
    meta = json.loads((cache_dir / "meta.json").read_text())
    class_map = meta["class_map"]
    y, device_id = arrays["y"], arrays["device_id"]
    row = 0
    for device, cat in subset_names():
        block = slice(row, row + ROWS_PER_SUBSET)
        assert (y[block] == class_map[cat]).all()
        assert (device_id[block] == device).all()
        row += ROWS_PER_SUBSET


def test_splits_70_10_20_disjoint_per_subset(built):
    _, cache_dir, _ = built
    splits = json.loads((cache_dir / "splits.json").read_text())
    subsets = splits["subsets"]
    assert len(subsets) == N_SUBSETS
    n_priv = round(0.7 * ROWS_PER_SUBSET)
    n_open = round(0.1 * ROWS_PER_SUBSET)
    n_test = ROWS_PER_SUBSET - n_priv - n_open
    row = 0
    for device, cat in subset_names():
        entry = subsets[f"{device}.{cat}"]
        priv, opn, tst = entry["private"], entry["open"], entry["test"]
        assert len(priv) == n_priv
        assert len(opn) == n_open
        assert len(tst) == n_test
        all_idx = set(priv) | set(opn) | set(tst)
        assert len(all_idx) == ROWS_PER_SUBSET, "splits overlap"
        assert all_idx == set(range(row, row + ROWS_PER_SUBSET)), (
            "splits must cover exactly this subset's rows"
        )
        row += ROWS_PER_SUBSET


def test_meta_contents(built, arrays):
    _, cache_dir, seed = built
    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["seed"] == seed
    # Global 11-class map, always full even though devices 3/7 lack mirai.
    assert meta["class_map"] == {cat: i for i, cat in enumerate(sorted(CATEGORIES))}
    counts = meta["per_device_class_counts"]
    assert set(counts) == {str(d) for d in range(1, 10)}
    for device in range(1, 10):
        expect = {
            cat: ROWS_PER_SUBSET
            for cat in CATEGORIES
            if not (device in NO_MIRAI_DEVICES and cat.startswith("mirai"))
        }
        assert counts[str(device)] == expect
    h = meta["raw_data_sha256"]
    assert isinstance(h, str) and len(h) == 64
    int(h, 16)  # valid hex


def test_rebuild_same_seed_is_byte_identical(built, tmp_path):
    from ssfl.data.build import build

    data_dir, cache_dir, seed = built
    cache2 = tmp_path / "cache2"
    build(data_dir=data_dir, cache_dir=cache2, seed=seed, rows_per_subset=ROWS_PER_SUBSET)
    names = sorted(p.name for p in cache_dir.iterdir())
    assert names == sorted(p.name for p in cache2.iterdir())
    for name in names:
        assert (cache_dir / name).read_bytes() == (cache2 / name).read_bytes(), (
            f"cache/{name} differs between identical builds"
        )


def test_different_seed_changes_splits(built, tmp_path):
    from ssfl.data.build import build

    data_dir, cache_dir, seed = built
    cache2 = tmp_path / "cache_other_seed"
    build(data_dir=data_dir, cache_dir=cache2, seed=seed + 1, rows_per_subset=ROWS_PER_SUBSET)
    assert (cache_dir / "splits.json").read_bytes() != (cache2 / "splits.json").read_bytes()


def test_build_is_an_explicit_module_command(built, tmp_path):
    import subprocess
    import sys

    data_dir, _, _ = built
    cache_dir = tmp_path / "cli_cache"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ssfl.data.build",
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(cache_dir),
            "--seed",
            "7",
            "--rows-per-subset",
            str(ROWS_PER_SUBSET),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (cache_dir / "mini.npz").exists()
    assert (cache_dir / "meta.json").exists()
