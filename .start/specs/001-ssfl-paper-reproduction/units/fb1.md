---
id: fb1
title: Framework base (config, payload contract, durable metrics, seeding)
type: feature
dependencies: []
---
# Framework base (config, payload contract, durable metrics, seeding)

## Goal
The shared spine every other unit builds on: run configuration with validation, the per-method payload contract, durable run-results storage, and the reproducibility/seeding discipline.

## Requirements
- `RunConfig`: method (fl|fd|dsfl|ssfl), model (cnn|mlp|lstm), scenario (1|2|3), seed, rounds (default 200), lr (1e-4), batch (80), local_epochs (5), threshold ("median"|0.7|0.8|0.9), ablation flags (no_voting, no_discriminating, simply_filtering), label_mode (hard|soft2|soft4|soft6|soft8), device ("auto"), num_parallel_clients.
- Validation rejects invalid combinations before anything launches (ablation flags/threshold/label_mode only valid with method=ssfl; unknown values listed with allowed options).
- Deterministic run-id derivation: `{method}-{model}-s{scenario}-seed{seed}[-flags]`; same config ⇒ same id.
- Payload contract module: single source of truth for each method's exchanged arrays — names, dtypes, shapes as specified in solution.md's "Internal API Changes" (FL weights; FD float32 [L,L]; DS-FL float32 [N_o,L]; SSFL int64 [N_o] with −1 sentinel; soft-label modes float32 [N_o,L] with decimal precision attribute). Includes per-payload byte-size calculators.
- Durable metrics store for `results/<run-id>/`: `config.json` written at start; one JSONL line per round appended and flushed immediately (round, test_acc, wall_s, optional diagnostics dict); `final.json` written atomically (temp file + rename); confusion matrix saved as `cm.npy`; a reader API returning rounds as records.
- Seeding discipline: one helper deriving all stochastic state from (run seed, client_id, round) so identical configs reproduce identical results.
- Unit tests cover validation rejections, run-id stability, byte-size calculations, append durability (simulated interrupt leaves prior lines readable), and atomicity of final.json.

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (ADR-2, ADR-7; Data Storage Changes).
- No Flower imports; stdlib + numpy only.
- Full test coverage for all requirements.

## Interfaces
- RunConfig, payload contract, metrics store, and seeding helper are consumed by every other unit; the payload contract is the sole definition cm1 uses for cost accounting and tr1 uses for Message assembly.
