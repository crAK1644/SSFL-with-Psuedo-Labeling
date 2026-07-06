---
unit: ss1
feature: Majority vote (Eq. 17) with tie-break and zero-vote rules
priority: P0
---
# The server vote reproduces the documented walkthrough exactly

## Scenario
Through the SSFL method's public vote function, with L=3 classes and 4 open samples, submit client label vectors [[2, −1, 0, 1], [2, −1, 1, 1], [0, −1, 1, 2]]. Also submit a two-sample case where sample 0 receives one vote each for classes 0 and 1 (a tie).

## Expected
The 4-sample vote returns [2, −1, 1, 1]: majority wins; sample 1 (all −1) has no global label.
The tie case returns class 0 (lowest class index), and returns it identically on repeated invocations.
−1 entries are never counted as votes for any class.
