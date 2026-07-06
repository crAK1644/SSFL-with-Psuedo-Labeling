---
id: cm1
title: Communication-cost accounting
type: feature
dependencies: [fb1]
---
# Communication-cost accounting

## Goal
Analytic per-round and cumulative communication cost for every method, and the C@x metrics derived from a run's accuracy curve.

## Requirements
- Per-round upload/download bytes computed analytically from fb1's payload contract (never by sniffing live traffic): FL = parameter bytes × clients × both directions; FD = [L,L] float32 per client; DS-FL = [N_o,L] float32 per client; SSFL hard = int64 [N_o] per client upload, global labels download; SSFL soft modes scale with decimal precision as the paper's Fig. 6 describes.
- One-time open-set distribution cost (C@D^o) reported separately for the methods that require it (DS-FL, SSFL), as in Table IV.
- Cumulative MB-vs-round curve for any completed run, computed from its config + round count.
- C@50, C@75, C@Top-Acc: cumulative cost at the first round the run's test accuracy reaches 50% / 75% / its own maximum, read from the run's per-round records (fb1 reader). Targets the run never reaches are reported as explicitly unreached, never fabricated.
- Unit tests cover: hand-computed byte counts per method (including a soft-label mode), C@x extraction from a synthetic accuracy curve, and the unreached-target case.

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (payload contract is the single source of truth).
- No Flower imports; consumes only fb1 APIs and results files.
- Full test coverage for all requirements.

## Interfaces
- Cost functions consumed by rp1 (Table IV, Fig. 6) and available to rn1 for per-run summaries.
