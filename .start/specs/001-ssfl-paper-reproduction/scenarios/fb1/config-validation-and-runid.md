---
unit: fb1
feature: Run configuration and validation
priority: P0
---
# Invalid configs are rejected; run-ids are stable

## Scenario
Through the public config API: (a) create a config with method "fl" and `no_voting=True`; (b) create a config with threshold "0.85"; (c) create two identical valid SSFL configs and compare their run-ids; (d) create a valid default config.

## Expected
(a) and (b) are rejected before any launch, with error messages listing the allowed values.
(c) both configs derive the identical run-id string following `{method}-{model}-s{scenario}-seed{seed}` (plus flag suffixes when set).
(d) defaults equal the paper's values: rounds 200, lr 1e-4, batch 80, local epochs 5.
