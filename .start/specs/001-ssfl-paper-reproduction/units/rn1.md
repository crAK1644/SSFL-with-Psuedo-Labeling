---
id: rn1
title: Runner CLI, timing pilot, campaign orchestration
type: feature
dependencies: [tr1]
---
# Runner CLI, timing pilot, campaign orchestration

## Goal
The researcher-facing entry points: a single-run CLI with validation and progress, a timing mode that projects campaign duration, and a resumable script that executes the full ~30-run suite.

## Requirements
- `python -m ssfl.run` accepting: --method, --model, --scenario, --seed, --rounds, --device, --threshold, --no-voting, --no-discriminating, --simply-filtering, --label-mode, --num-parallel-clients; defaults are the paper's hyperparameters via fb1's RunConfig.
- Config validation happens before any launch (fb1); invalid input exits with the allowed values and a non-zero code; a missing dataset cache aborts with instructions to run the builder.
- On launch: prints the run-id, writes `config.json`, invokes `flwr run` with the resolved run-config; per round prints a one-line progress entry (round, accuracy, ETA); on completion prints the results path.
- Restart policy: if the run-id's directory exists without `final.json`, it is moved aside to `<run-id>.aborted-<timestamp>` and the run restarts from round 1 (no mid-run resume).
- `--timing` mode: runs a short configurable number of rounds, reports measured seconds/round, and prints a projected wall-clock for every run in the campaign plan on this machine.
- `python -m ssfl.campaign`: executes the full planned suite in a documented order (18 Table II runs, then ablation and label-strategy runs); skips any run whose `final.json` exists; prints done/remaining progress; safe to re-invoke any time.
- Unit tests cover: CLI validation failures, restart policy directory handling, campaign skip logic against a synthetic results tree, and timing projection arithmetic (mocked round timings).

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (Project Commands; Error Handling; ADR-7).
- The runner launches tr1's apps; it never trains anything itself.
- Full test coverage for all requirements.

## Interfaces
- Consumes tr1 (via `flwr run`) and fb1 (RunConfig, results layout).
- The campaign plan (run list) is a data structure rp1 can read to know which runs are expected.
