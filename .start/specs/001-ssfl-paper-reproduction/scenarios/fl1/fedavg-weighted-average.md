---
unit: fl1
feature: FedAvg aggregation (Eq. 1)
priority: P0
---
# Sample-weighted averaging matches a hand-computed example

## Scenario
Through the FL method's public aggregate function: client A holds 300 samples and weights all equal to 1.0; client B holds 100 samples and weights all equal to 5.0. Aggregate the two weight sets.

## Expected
Every aggregated parameter equals (300×1.0 + 100×5.0) / 400 = 2.0.
Aggregating a single client returns that client's weights unchanged.
