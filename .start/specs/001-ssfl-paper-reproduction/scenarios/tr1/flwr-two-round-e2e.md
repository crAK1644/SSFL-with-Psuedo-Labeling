---
unit: tr1
feature: Flower simulation end-to-end
priority: P0
---
# A 2-round Flower simulation produces a well-formed results directory

## Scenario
With the dataset cache built, launch the Flower simulation (through the project's documented launch path) for method `fl` and then for method `ssfl`, each with rounds=2, a small client count, and a fixed seed on this machine.

## Expected
Both runs exit with code 0.
Each `results/<run-id>/` contains: `config.json` matching the requested settings, `rounds.jsonl` with exactly 2 well-formed lines (round, test_acc in [0,1], wall_s), an atomic `final.json` with accuracy/F1/precision, and `cm.npy` shaped [11, 11].
The SSFL run's round records include diagnostics (unfamiliar counts, zero-vote count, vote agreement).
Per-client checkpoint files exist under the run's `ckpt/` directory, keyed by partition id.
