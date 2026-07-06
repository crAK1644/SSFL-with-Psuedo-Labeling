---
unit: ds1
feature: Framework-free DS-FL round driver
priority: P1
---
# A 2-round DS-FL micro-run evaluates the server model without touching open labels

## Scenario
Run the DS-FL round driver for 2 rounds with 3 tiny clients from the built cache, seed 0. Both client distillation and server-model distillation execute. Monitor which data the driver requests from the loader API.

## Expected
Both rounds complete without flwr/ray.
Reported per-round accuracy comes from the server-held model on the test split.
No ground-truth labels for open-split samples are requested from any loader at any point.
