---
unit: ss1
feature: SSFL client round (Eqs. 11–16)
priority: P0
---
# A client round returns filtered hard labels; degenerate clients survive

## Scenario
Run one SSFL client round for: (a) a normal client with ~200 private samples of 3 classes from the built cache; (b) a client holding a single class (Scenario 2 shape); (c) a client whose discriminator marks every open sample unfamiliar (forced by a tiny/adversarial setup).

## Expected
(a) returns an int64 vector of open-set length whose entries are −1 or valid class indices, containing at least one −1 (something was unfamiliar) and at least one valid label.
(b) completes all five steps (distill/train/discriminate/filter/return) without error.
(c) returns all −1; a subsequent vote over only such clients returns all −1; the round still completes and produces metrics.
