---
unit: fd1
feature: Framework-free FD round driver
priority: P1
---
# A 2-round FD micro-run completes with best-client evaluation

## Scenario
Run the FD round driver for 2 rounds with 3 tiny clients from the built cache, seed 0, including the distillation step with received logit targets.

## Expected
Both rounds complete without flwr/ray.
Evaluation reports the best client model's test accuracy as a float in [0, 1].
A client holding a single class participates without error (its other class rows are zero vectors).
