---
id: dc1
title: Mini-N-BaIoT data core (build, partition, load)
type: feature
dependencies: []
---
# Mini-N-BaIoT data core (build, partition, load)

## Goal
One-time builder that turns the raw N-BaIoT CSVs into a cached, deterministic mini-N-BaIoT dataset, plus scenario partitioners and fast loaders.

## Requirements
- Build `cache/` from `data/*.csv` (89 device-category files, read-only): take the first 1000 rows of each device-category subset, min-max normalize all features to [0,1], reshape each 115-feature row to 23×5 following the paper's Eq. 19 column layout (features 0–22 → column 1, 23–45 → column 2, …).
- Split every subset 70/10/20 into private/open/test index sets that are mutually disjoint; the open split must be exposed to consumers without labels.
- Persist: `cache/mini.npz` (X float32 [N,23,5], y int64 [N], device_id int8 [N]), `cache/splits.json`, `cache/meta.json` (seed, global 11-class map, per-device class counts, raw-data content hash).
- Partition the private pool for three scenarios, written to `cache/scenario_<s>.json` (client_id → sample indices): Scenario 1 = label-sorted shards, 2 shards per client, 3 clients per device (27 clients); Scenario 2 = one class per client per device (89 clients); Scenario 3 = Dirichlet α=0.1 per device (89 clients).
- Devices 3 and 7 have no mirai traffic (6 classes each); the builder and partitioners must handle this without error, and Scenario 2 yields exactly 89 clients because of it.
- Building twice with the same seed produces byte-identical cache files; all randomness derives from a single seed argument.
- Loaders expose memory-mapped views: per-client private data (X, y), the shared open set (X only), and the test set (X, y), without re-reading any CSV.
- Building is an explicit command (`python -m ssfl.data.build`); consumers fail fast with a clear message if the cache is missing, never rebuild implicitly.
- Unit tests cover the invariants above (disjointness, value range, shapes, client counts, determinism, label-free open view).

## Constraints
- Conventions and directory layout per `.start/specs/001-ssfl-paper-reproduction/solution.md` (ADR-5; Directory Map).
- No Flower imports; numpy/pandas only for the build, numpy for loaders.
- Raw CSVs are opened read-only and parsed exactly once (at build time).
- Full test coverage for all requirements.

## Interfaces
- Produces the `cache/` layout consumed by fl1/fd1/ds1/ss1 (via loaders), tr1, and ts1.
- Loader API is the only sanctioned data access path for other units.
