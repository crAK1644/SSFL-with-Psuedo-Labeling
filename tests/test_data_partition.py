"""Tests for the scenario partitioners (ssfl.data.partition)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from helpers_data import NO_MIRAI_DEVICES, build_fixture


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_fixture(tmp_path_factory)


@pytest.fixture(scope="module")
def cache(built):
    _, cache_dir, _ = built
    with np.load(cache_dir / "mini.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    splits = json.loads((cache_dir / "splits.json").read_text())
    scenarios = {
        s: json.loads((cache_dir / f"scenario_{s}.json").read_text()) for s in (1, 2, 3)
    }
    return arrays, splits, scenarios


def _private_pool(splits):
    pool = set()
    for entry in splits["subsets"].values():
        pool.update(entry["private"])
    return pool


def _device_private(cache, device):
    """Per-subset private index lists belonging to one device."""
    _, splits, _ = cache
    return [
        entry["private"]
        for name, entry in splits["subsets"].items()
        if int(name.split(".", 1)[0]) == device
    ]


def test_scenario_files_written(built):
    _, cache_dir, _ = built
    for s in (1, 2, 3):
        assert (cache_dir / f"scenario_{s}.json").exists()


@pytest.mark.parametrize("scenario,expected", [(1, 27), (2, 89), (3, 89)])
def test_client_counts(cache, scenario, expected):
    _, _, scenarios = cache
    assert len(scenarios[scenario]) == expected


@pytest.mark.parametrize("scenario", [1, 2, 3])
def test_clients_disjoint_and_cover_private_pool(cache, scenario):
    _, splits, scenarios = cache
    seen: set[int] = set()
    total = 0
    for idx in scenarios[scenario].values():
        seen.update(idx)
        total += len(idx)
    assert total == len(seen), "clients overlap"
    assert seen == _private_pool(splits), "clients must exactly cover the private pool"


@pytest.mark.parametrize("scenario", [1, 2, 3])
def test_every_client_is_single_device_and_nonempty(cache, scenario):
    arrays, _, scenarios = cache
    device_id = arrays["device_id"]
    for cid, idx in scenarios[scenario].items():
        assert len(idx) > 0, f"scenario {scenario} client {cid} is empty"
        assert len(set(device_id[idx].tolist())) == 1, (
            f"scenario {scenario} client {cid} mixes devices"
        )


def test_scenario1_three_clients_per_device_from_two_shards(cache):
    arrays, _, scenarios = cache
    device_id, y = arrays["device_id"], arrays["y"]
    clients = scenarios[1]
    # 3 clients per device.
    per_device: dict[int, int] = {}
    for idx in clients.values():
        d = int(device_id[idx[0]])
        per_device[d] = per_device.get(d, 0) + 1
    assert per_device == {d: 3 for d in range(1, 10)}
    # Each client must be the union of exactly 2 shards of its device's
    # label-sorted private pool (6 shards per device, 2 per client).
    device_shards: dict[int, list[frozenset[int]]] = {}
    for d in range(1, 10):
        priv = np.sort(
            np.concatenate([np.array(i, dtype=int) for i in _device_private(cache, d)])
        )
        by_label = priv[np.argsort(y[priv], kind="stable")]
        device_shards[d] = [frozenset(int(i) for i in s) for s in np.array_split(by_label, 6)]
    for cid, idx in clients.items():
        d = int(device_id[idx[0]])
        members = set(idx)
        matched = [s for s in device_shards[d] if s <= members]
        assert len(matched) == 2, f"client {cid} is not exactly 2 label-sorted shards"
        assert set().union(*matched) == members


def test_scenario2_one_class_per_client_and_devices_3_7_have_6(cache):
    arrays, _, scenarios = cache
    device_id, y = arrays["device_id"], arrays["y"]
    per_device: dict[int, int] = {}
    for cid, idx in scenarios[2].items():
        assert len(set(y[idx].tolist())) == 1, f"client {cid} holds more than one class"
        d = int(device_id[idx[0]])
        per_device[d] = per_device.get(d, 0) + 1
    for d in range(1, 10):
        assert per_device[d] == (6 if d in NO_MIRAI_DEVICES else 11)


def test_scenario3_dirichlet_is_nonuniform(cache):
    arrays, _, scenarios = cache
    y = arrays["y"]
    # alpha=0.1 concentrates mass: client class distributions must be skewed,
    # i.e. not every client holds (near-)equal shares of every device class.
    sizes = sorted(len(idx) for idx in scenarios[3].values())
    assert sizes[0] < sizes[-1], "Dirichlet(0.1) should produce unequal client sizes"
    # And at least some clients miss some classes entirely.
    n_classes_held = [len(set(y[idx].tolist())) for idx in scenarios[3].values()]
    assert min(n_classes_held) < 6, "alpha=0.1 should starve some clients of classes"
