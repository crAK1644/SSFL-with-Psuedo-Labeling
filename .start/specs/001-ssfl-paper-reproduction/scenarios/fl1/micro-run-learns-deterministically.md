---
unit: fl1
feature: Framework-free FL round driver
priority: P1
---
# A 2-round micro-run learns and is deterministic

## Scenario
Using the FL round driver with 3 clients drawn from a tiny subset of the built cache (≤200 samples each), run 2 rounds with seed 0 and record per-round test accuracy of the aggregated model. Repeat the identical run.

## Expected
Both rounds complete without any FL framework (no flwr/ray import required by the driver).
Round-2 global-model training loss is lower than round-1's initial loss.
The repeated run reproduces identical per-round accuracies.
