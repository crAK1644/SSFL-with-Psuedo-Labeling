---
unit: rn1
feature: Resumable campaign orchestration
priority: P0
---
# Re-invoking the campaign never repeats completed runs

## Scenario
Prepare a results tree where 3 of the campaign's planned runs have complete `final.json` files and one has a directory without `final.json` (simulated crash). Invoke the campaign script in dry-run/plan mode (or with a stubbed runner).

## Expected
The 3 complete runs are reported as done and are not re-executed.
The crashed run is scheduled for re-execution, with its stale directory moved aside (renamed with an aborted marker), not silently overwritten.
Progress output shows done/remaining counts consistent with the plan (~30 runs total).
