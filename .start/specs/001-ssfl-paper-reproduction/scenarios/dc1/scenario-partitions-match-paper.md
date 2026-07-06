---
unit: dc1
feature: Non-IID scenario partitioning
priority: P1
---
# Scenario partitions match the paper's client structure

## Scenario
After a successful build, load `cache/scenario_1.json`, `cache/scenario_2.json`, and `cache/scenario_3.json` and inspect the client → indices mappings against the labels in `mini.npz`.

## Expected
Scenario 1 has exactly 27 clients; scenario 2 has exactly 89; scenario 3 has exactly 89.
In scenario 2, every client's samples carry exactly one class label.
In scenario 1, every client's samples come from exactly one device and span at most 2 label shards.
In every scenario, each client's indices are a subset of the private split only (no open or test indices).
All indices for devices 3 and 7 map to at most 6 distinct classes.
