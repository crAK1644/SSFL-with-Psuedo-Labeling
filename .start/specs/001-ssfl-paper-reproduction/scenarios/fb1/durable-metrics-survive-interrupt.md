---
unit: fb1
feature: Durable per-round metrics
priority: P0
---
# Round records survive a mid-run kill; final.json is atomic

## Scenario
Open a metrics store for a fresh run directory. Append 5 round records (flushing per append, as the API contracts). Simulate an abrupt process death (no close/finalize call). Reopen the results directory with the reader API. Separately, begin writing final metrics and interrupt before completion.

## Expected
The reader returns exactly 5 well-formed round records (round, test_acc, wall_s fields present).
No partial or corrupt trailing line is returned.
After the interrupted finalization, no `final.json` exists in the directory (a temp file may remain); a subsequent successful finalization produces a complete, parseable `final.json`.
