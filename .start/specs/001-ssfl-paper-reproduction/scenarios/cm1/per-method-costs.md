---
unit: cm1
feature: Analytic communication accounting
priority: P0
---
# Per-method cumulative costs match hand computation

## Scenario
Through the accounting API, compute cumulative upload cost over 200 rounds for FL, DS-FL, and SSFL (hard labels) in Scenario 1 (27 clients, N_o = 8900, L = 11), plus the one-time open-set distribution cost for DS-FL/SSFL.

## Expected
FL cumulative cost equals 200 × 27 × (4 × CNN-parameter-count) bytes for uploads (and scales identically for downloads).
SSFL cumulative upload cost equals 200 × 27 × 8 × 8900 bytes — several orders of magnitude below FL's.
DS-FL sits between the two.
The open-set distribution cost (C@D^o) is reported once (not per round) and only for DS-FL and SSFL, matching Table IV's ~0.96 MB order of magnitude.
