---
unit: fd1
feature: FD logit aggregation (Eqs. 2–4)
priority: P0
---
# Absent classes yield zero vectors; Eq. 4 excludes the client's own logits

## Scenario
Through the FD method's public functions: client A has samples of classes 0 and 1 only (L=3 classes total); compute its per-class average logit matrix. Then, with three clients' [3,3] logit matrices where hand-computable values are used, compute each client's new target per Eq. 4.

## Expected
Client A's row for class 2 is the zero vector.
For a class where all three clients contributed, client k's target equals the average of the other clients' vectors only — verified against the hand-computed value of (N^l × ȳ^{s,l} − ȳ^{k,l}) / (N^l − 1).
