---
unit: ts1
feature: Pre-flight smoke gate
priority: P0
---
# The smoke gate passes in under 5 minutes without Ray

## Scenario
With the dataset cache built, run the documented smoke command (`uv run pytest -m smoke` or equivalent) and time it. Inspect which modules were imported during the run.

## Expected
All smoke tests pass: data invariants (disjoint splits, [0,1] range, shapes, 27/89/89 client counts, 6 classes for devices 3/7, label-free open view) and 2-round micro-runs for FL, FD, DS-FL, SSFL, and at least one SSFL ablation variant.
Total wall-clock is under 5 minutes on this machine.
Neither ray nor the Flower simulation engine is imported by any smoke test.
A failing invariant (e.g., corrupted cache) produces a failure message naming the violated invariant.
