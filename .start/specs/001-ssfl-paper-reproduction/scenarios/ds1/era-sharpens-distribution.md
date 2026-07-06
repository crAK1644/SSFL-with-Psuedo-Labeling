---
unit: ds1
feature: Entropy Reduction Aggregation (Eqs. 6–8)
priority: P0
---
# ERA reduces entropy of averaged logits

## Scenario
Through the DS-FL method's public aggregate function: average three clients' logit matrices for 10 open samples (L=11), then apply ERA with the configured temperature T < 1. Compute Shannon entropy of each sample's distribution before and after ERA.

## Expected
For every non-degenerate sample distribution, post-ERA entropy is strictly lower than pre-ERA entropy.
Each output row is a valid probability distribution (non-negative, sums to 1).
The most probable class per sample is unchanged by ERA.
