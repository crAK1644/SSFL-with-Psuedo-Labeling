---
unit: rn1
feature: Timing pilot
priority: P1
---
# Timing mode measures and projects

## Scenario
Run `uv run python -m ssfl.run --method ssfl --scenario 1 --rounds 3 --timing` on the built cache with a small client count.

## Expected
Output includes a measured seconds-per-round figure.
Output includes a projected wall-clock duration for every run in the campaign plan, scaled by each run's client count and round count.
Exit code 0; no 200-round run is started.
