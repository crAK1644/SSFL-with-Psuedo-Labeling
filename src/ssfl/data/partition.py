"""Scenario partitioners: private pool -> client_id -> sample indices.

Three non-IID scenarios from the paper, all per device:

- Scenario 1: label-sorted shards, 2 shards per client, 3 clients per device
  -> 27 clients.
- Scenario 2: one class per client per device -> 89 clients (devices 3 and 7
  have no mirai traffic, so they contribute 6 clients each, not 11).
- Scenario 3: Dirichlet(alpha=0.1) over each device's present classes, one
  client per present class -> 89 clients.

All functions are pure numpy; randomness comes from the caller's SeedSequence.
"""

from __future__ import annotations

import numpy as np

SHARDS_PER_CLIENT = 2
CLIENTS_PER_DEVICE = 3
DIRICHLET_ALPHA = 0.1


def _private_by_device(
    splits: dict, device_id: np.ndarray
) -> dict[int, np.ndarray]:
    """Device -> sorted global private indices, devices in ascending order."""
    by_device: dict[int, list[int]] = {}
    for entry in splits["subsets"].values():
        for i in entry["private"]:
            by_device.setdefault(int(device_id[i]), []).append(i)
    return {d: np.array(sorted(idx)) for d, idx in sorted(by_device.items())}


def scenario_shards(
    splits: dict,
    y: np.ndarray,
    device_id: np.ndarray,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    """Scenario 1: per device, sort private samples by label, cut into
    CLIENTS_PER_DEVICE * SHARDS_PER_CLIENT shards, deal 2 random shards per client."""
    clients: dict[int, list[int]] = {}
    cid = 0
    for _, priv in _private_by_device(splits, device_id).items():
        by_label = priv[np.argsort(y[priv], kind="stable")]
        n_shards = CLIENTS_PER_DEVICE * SHARDS_PER_CLIENT
        shards = np.array_split(by_label, n_shards)
        order = rng.permutation(n_shards)
        for k in range(CLIENTS_PER_DEVICE):
            picks = order[k * SHARDS_PER_CLIENT : (k + 1) * SHARDS_PER_CLIENT]
            clients[cid] = sorted(int(i) for s in picks for i in shards[s])
            cid += 1
    return clients


def scenario_one_class(
    splits: dict, y: np.ndarray, device_id: np.ndarray
) -> dict[int, list[int]]:
    """Scenario 2: one client per (device, present class). Deterministic."""
    clients: dict[int, list[int]] = {}
    cid = 0
    for _, priv in _private_by_device(splits, device_id).items():
        for label in np.unique(y[priv]):
            clients[cid] = [int(i) for i in priv[y[priv] == label]]
            cid += 1
    return clients


def scenario_dirichlet(
    splits: dict,
    y: np.ndarray,
    device_id: np.ndarray,
    rng: np.random.Generator,
    alpha: float = DIRICHLET_ALPHA,
) -> dict[int, list[int]]:
    """Scenario 3: per device, split each present class across as many clients
    as the device has classes, with Dirichlet(alpha) proportions.

    Devices 3/7 (6 classes) simply get 6 clients. Empty clients (possible at
    alpha=0.1) are topped up deterministically from the largest client so every
    client trains on at least one sample.
    """
    clients: dict[int, list[int]] = {}
    cid_base = 0
    for _, priv in _private_by_device(splits, device_id).items():
        labels = np.unique(y[priv])
        k = len(labels)
        parts: list[list[int]] = [[] for _ in range(k)]
        for label in labels:
            idx = rng.permutation(priv[y[priv] == label])
            proportions = rng.dirichlet(np.full(k, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
            for part, chunk in zip(parts, np.split(idx, cuts)):
                part.extend(int(i) for i in chunk)
        # Deterministic top-up: no client may end up empty.
        while any(len(p) == 0 for p in parts):
            donor = max(range(k), key=lambda j: len(parts[j]))
            taker = next(j for j in range(k) if len(parts[j]) == 0)
            parts[taker].append(parts[donor].pop())
        for j, part in enumerate(parts):
            clients[cid_base + j] = sorted(part)
        cid_base += k
    return clients


def make_scenarios(
    splits: dict,
    y: np.ndarray,
    device_id: np.ndarray,
    seed_seq: np.random.SeedSequence,
) -> dict[int, dict[str, list[int]]]:
    """All three scenario partitions, keyed by scenario number, with string
    client ids (JSON-ready). Each scenario draws from its own seed stream."""
    s1_seq, s3_seq = seed_seq.spawn(2)
    scenarios = {
        1: scenario_shards(splits, y, device_id, np.random.default_rng(s1_seq)),
        2: scenario_one_class(splits, y, device_id),
        3: scenario_dirichlet(splits, y, device_id, np.random.default_rng(s3_seq)),
    }
    return {s: {str(c): idx for c, idx in cl.items()} for s, cl in scenarios.items()}
