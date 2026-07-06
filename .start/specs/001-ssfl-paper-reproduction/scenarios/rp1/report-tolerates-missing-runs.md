---
unit: rp1
feature: Partial-results tolerance
priority: P1
---
# Missing runs are listed, not fatal

## Scenario
Delete half the runs (including one full method and one ablation series) from the synthetic results tree and run the report again.

## Expected
Exit code 0.
The report header explicitly lists every missing run by run-id.
Tables render available cells and mark absent cells distinctly (not zero, not the paper value).
Figures that depend entirely on missing runs are skipped with a note; the rest render.
