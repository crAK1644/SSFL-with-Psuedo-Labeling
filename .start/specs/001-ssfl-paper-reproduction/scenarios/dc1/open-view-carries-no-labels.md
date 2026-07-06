---
unit: dc1
feature: Semisupervised split integrity
priority: P1
---
# Open-set loader view exposes no labels

## Scenario
Using the public loader API on a built cache, request the shared open-set view and inspect what it returns. Then attempt to obtain labels for open-set samples through every loader entry point.

## Expected
The open-set view provides features (X) only — no label array is present in the returned object.
No public loader function returns ground-truth labels for open-split indices.
The private and test views do return (X, y) pairs.
