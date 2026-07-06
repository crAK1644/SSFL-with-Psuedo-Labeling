---
unit: fb1
feature: Payload contract byte accounting
priority: P1
---
# Payload byte calculators match hand-computed sizes

## Scenario
Using the payload contract's byte-size calculators with N_o = 8900 open samples, L = 11 classes, and the Table I CNN parameter count P: compute upload sizes for FL, FD, DS-FL, SSFL hard labels, and SSFL soft labels at 2 decimals.

## Expected
FL upload = 4 × P bytes (float32 weights).
FD upload = 4 × 11 × 11 = 484 bytes.
DS-FL upload = 4 × 8900 × 11 = 391,600 bytes.
SSFL hard-label upload = 8 × 8900 = 71,200 bytes (int64) — orders of magnitude below FL.
SSFL soft-label sizes vary with the configured decimal precision and exceed the hard-label size.
