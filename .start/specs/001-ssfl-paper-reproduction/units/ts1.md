---
id: ts1
title: Aggregate smoke suite (pre-flight gate)
type: feature
dependencies: [dc1, mz1, fl1, fd1, ds1, ss1]
---
# Aggregate smoke suite (pre-flight gate)

## Goal
The fast pre-flight gate run before any hours-long experiment: end-to-end data invariants plus a 2-round micro-run of every method, all without Ray, finishing in under 5 minutes.

## Requirements
- A pytest suite (marker or directory selectable, e.g. `pytest -m smoke` or `pytest tests/smoke`) that runs strictly at the logic layer — no Flower simulation, no Ray.
- Data invariants against the real built cache: split disjointness per subset, values within [0,1], X shape [N,23,5], scenario client counts 27/89/89, devices 3/7 exposing exactly 6 classes, and the open-set view exposing no labels.
- Method micro-runs: for each of FL, FD, DS-FL, SSFL and at least one SSFL ablation variant, a 2-round round-driver run on a tiny subset (few clients, few samples) that completes and yields well-formed metrics (accuracy in [0,1], correct record fields).
- Semisupervised integrity check: an assertion that no code path consumed open-split ground-truth labels during any micro-run.
- Wall-clock budget asserted: the whole suite passes in under 5 minutes on the target laptop.
- Clear failure messages naming the violated invariant (these failures are what stop a researcher from burning hours on a broken run).

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (Quality Requirements; ADR-4 makes Ray-free micro-runs possible).
- This unit owns the aggregate gate; per-unit unit tests remain in their own units (no duplication — this suite tests cross-unit behavior and the real cache).
- Full test coverage is the deliverable itself.

## Interfaces
- Consumes dc1 loaders/builder output, mz1 models, and the four method round drivers.
- `uv run pytest -m smoke` (or equivalent) is the documented pre-campaign gate used by rn1's docs.
