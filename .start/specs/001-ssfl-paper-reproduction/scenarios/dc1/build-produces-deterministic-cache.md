---
unit: dc1
feature: Mini-N-BaIoT dataset builder
priority: P0
---
# Build produces a valid, deterministic cache

## Scenario
Run `uv run python -m ssfl.data.build --seed 42` on the raw `data/*.csv`. Record checksums of all produced cache files. Delete `cache/` and run the identical command again.

## Expected
`cache/mini.npz`, `cache/splits.json`, `cache/scenario_1.json`, `cache/scenario_2.json`, `cache/scenario_3.json`, `cache/meta.json` all exist.
In `mini.npz`: X is float32 with shape [N, 23, 5], all values within [0.0, 1.0]; y is int64 with 11 distinct classes overall; N equals 1000 × number of device-category subsets (89,000).
In `splits.json`: for every subset, private/open/test index lists are pairwise disjoint and sized 70%/10%/20% of 1000.
The second build produces byte-identical files to the first (matching checksums).
