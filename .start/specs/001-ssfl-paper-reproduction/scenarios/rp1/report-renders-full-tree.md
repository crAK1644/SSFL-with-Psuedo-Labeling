---
unit: rp1
feature: Report generation
priority: P0
---
# The report regenerates all paper artifacts with deltas

## Scenario
Against a synthetic but complete results tree (all planned runs present with plausible metrics), run `uv run python -m ssfl.report`.

## Expected
The output directory contains: Table II, Table III, and Table IV in both Markdown and CSV; three confusion-matrix PNGs (Fig 3); ablation-curve PNGs per scenario (Fig 4); threshold-curve PNGs (Fig 5); label-strategy accuracy and cost PNGs (Fig 6).
Every Table II cell shows our value, the paper's value, and the delta (e.g., SSFL scenario 1 compared against 87.40% accuracy).
Unreached C@x entries in Table IV render as unreached, not as numbers.
Exit code 0, completing in under 1 minute.
