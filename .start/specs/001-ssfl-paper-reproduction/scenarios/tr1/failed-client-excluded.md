---
unit: tr1
feature: Client-failure tolerance
priority: P1
---
# An erroring client is excluded and the round completes

## Scenario
At the strategy level (no Ray required), feed the SSFL strategy's aggregation a set of replies where one reply carries an error and the rest carry valid hard-label payloads.

## Expected
Aggregation completes using only the valid replies.
The resulting round record counts exactly 1 failed client.
The voted global labels equal what the valid replies alone would produce.
