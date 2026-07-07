"""Tests for the memory-mapped cache loaders (ssfl.data.loader)."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from helpers_data import build_fixture


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_fixture(tmp_path_factory)


@pytest.fixture(scope="module")
def cache_dir(built):
    return built[1]


@pytest.fixture(scope="module")
def raw(cache_dir):
    with np.load(cache_dir / "mini.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    splits = json.loads((cache_dir / "splits.json").read_text())
    return arrays, splits


def test_load_arrays_returns_memory_mapped_views(cache_dir):
    from ssfl.data.loader import load_arrays

    X, y, device_id = load_arrays(cache_dir)
    for arr in (X, y, device_id):
        assert isinstance(arr, np.memmap), "loader must memory-map, not read into RAM"
    assert X.dtype == np.float32 and X.shape[1:] == (23, 5)
    assert y.dtype == np.int64
    assert device_id.dtype == np.int8


def test_memmap_matches_npz_contents(cache_dir, raw):
    from ssfl.data.loader import load_arrays

    arrays, _ = raw
    X, y, device_id = load_arrays(cache_dir)
    np.testing.assert_array_equal(np.asarray(X), arrays["X"])
    np.testing.assert_array_equal(np.asarray(y), arrays["y"])
    np.testing.assert_array_equal(np.asarray(device_id), arrays["device_id"])


def test_open_view_is_label_free(cache_dir, raw):
    from ssfl.data.loader import load_open

    arrays, splits = raw
    X_open = load_open(cache_dir)
    # X only — a bare array, no labels anywhere in the return value.
    assert isinstance(X_open, np.ndarray) and not isinstance(X_open, tuple)
    open_idx = sorted(
        i for entry in splits["subsets"].values() for i in entry["open"]
    )
    np.testing.assert_array_equal(X_open, arrays["X"][open_idx])
    # And the API cannot leak labels: load_open returns exactly one value.
    sig = inspect.signature(load_open)
    assert "y" not in sig.parameters


def test_test_view(cache_dir, raw):
    from ssfl.data.loader import load_test

    arrays, splits = raw
    X_test, y_test = load_test(cache_dir)
    test_idx = sorted(
        i for entry in splits["subsets"].values() for i in entry["test"]
    )
    np.testing.assert_array_equal(X_test, arrays["X"][test_idx])
    np.testing.assert_array_equal(y_test, arrays["y"][test_idx])


@pytest.mark.parametrize("scenario,n_clients", [(1, 27), (2, 89), (3, 89)])
def test_client_views_match_scenario_partitions(cache_dir, raw, scenario, n_clients):
    from ssfl.data.loader import load_client, num_clients

    arrays, _ = raw
    assert num_clients(scenario, cache_dir) == n_clients
    partition = json.loads((cache_dir / f"scenario_{scenario}.json").read_text())
    for cid in (0, n_clients - 1):
        X_c, y_c = load_client(scenario, cid, cache_dir)
        idx = partition[str(cid)]
        np.testing.assert_array_equal(X_c, arrays["X"][idx])
        np.testing.assert_array_equal(y_c, arrays["y"][idx])


def test_loading_does_not_touch_raw_csvs(built):
    """The loader module must not depend on CSV parsing at all."""
    import ssfl.data.loader as loader_mod

    data_dir, cache_dir, _ = built
    src = inspect.getsource(loader_mod)
    assert "pandas" not in src and "read_csv" not in src
    # Loads still work when the raw data directory no longer exists.
    from ssfl.data.loader import load_open, load_test

    assert load_open(cache_dir).shape[0] > 0
    assert load_test(cache_dir)[0].shape[0] > 0


def test_missing_cache_fails_fast_with_build_instructions(tmp_path):
    from ssfl.data.loader import CacheMissingError, load_arrays, load_open

    for fn in (load_arrays, load_open):
        with pytest.raises(CacheMissingError) as exc:
            fn(tmp_path / "nope")
        msg = str(exc.value)
        assert "ssfl.data.build" in msg, "error must tell the user how to build"
    assert issubclass(CacheMissingError, FileNotFoundError)


def test_missing_cache_never_rebuilds_implicitly(tmp_path):
    from ssfl.data.loader import CacheMissingError, load_arrays

    empty = tmp_path / "empty_cache"
    empty.mkdir()
    with pytest.raises(CacheMissingError):
        load_arrays(empty)
    assert list(empty.iterdir()) == [], "loader must not create cache files"
