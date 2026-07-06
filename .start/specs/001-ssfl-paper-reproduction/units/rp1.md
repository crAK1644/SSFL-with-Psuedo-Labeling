---
id: rp1
title: Report generation (Tables II–IV, Figures 3–6)
type: feature
dependencies: [fb1, cm1]
---
# Report generation (Tables II–IV, Figures 3–6)

## Goal
One command that regenerates every result artifact of the paper from `results/`, with the paper's published values side-by-side for instant reproduction assessment.

## Requirements
- `python -m ssfl.report` scans `results/`, treats runs with `final.json` as complete, and produces into a report output directory:
  - **Table II**: accuracy / F1-score / precision for FL, FD, DS-FL, MLP, LSTM, SSFL × scenarios 1–3, each cell paired with the paper's value and the delta.
  - **Table III**: top-1 test accuracy at rounds 10/50/100/150/200 per method × scenario, ours vs. paper.
  - **Table IV**: C@D^o, C@50, C@75, C@Top-Acc and Top-Acc per method × scenario via cm1, unreached targets shown as such, ours vs. paper.
  - **Figure 3**: one confusion-matrix heatmap per scenario from SSFL's `cm.npy`, row-normalized with the paper's class ordering and labels.
  - **Figure 4**: SSFL ablation accuracy-vs-round curves (full, w/o voting, w/o discriminating, w/o both, simply filtering) per scenario.
  - **Figure 5**: confidence-threshold curves (0.7 / 0.8 / 0.9 / median) per scenario.
  - **Figure 6**: label-strategy accuracy curves and cumulative communication-cost curves (soft 8/6/4/2 decimals vs. hard) per scenario.
- Tables emitted as both Markdown and CSV; figures as PNG.
- Missing or incomplete runs are listed explicitly in the report header; the report renders everything available and never fails because runs are absent.
- Paper reference values live in one data file (not scattered through plotting code).
- Runs in under 1 minute against a full results tree.
- Unit tests cover: rendering against a synthetic results tree (complete + missing runs), delta computation, and the unreached-C@x display path.

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (Quality Requirements; results layout from fb1).
- No Flower imports; matplotlib + pandas over fb1's reader and cm1's cost functions only.
- Full test coverage for all requirements.

## Interfaces
- Consumes fb1's results reader, cm1's cost functions, and rn1's campaign plan (for the expected-runs list).
