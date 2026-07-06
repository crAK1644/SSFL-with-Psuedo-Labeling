---
id: ss1
title: SSFL method logic (proposed method + ablation flags)
type: feature
dependencies: [dc1, mz1, fb1]
---
# SSFL method logic (proposed method + ablation flags)

## Goal
Pure-Python implementation of the paper's proposed SSFL round (Eqs. 11–18, Algorithm 1) — classifier training, discriminator-based filtering, hard-label upload, server majority vote, distillation — with every ablation variant behind flags.

## Requirements
- Client round (one call, Algorithm 1 unrolled per solution.md's traced example): (1) distill the classifier on open-set samples whose previous-round global label ≠ −1 (skip in round 1), (2) train the classifier on private data for the configured local epochs (Eq. 11), (3) compute max-softmax confidence for all open samples (Eq. 12) and build the discriminator training set — open samples below the threshold labeled "unfamiliar", all private samples "familiar" (Eqs. 13–14), (4) train the discriminator, (5) predict open-set labels and replace those the discriminator marks unfamiliar with −1 (Eqs. 15–16); return the int64 hard-label vector.
- Threshold: per-client median of that round's confidence scores by default; fixed values 0.7/0.8/0.9 when configured.
- Server vote (Eq. 17): per-sample majority over client labels, −1 votes never counted, ties → lowest class index, zero-vote samples → −1 (excluded from that round's distillation). Must match solution.md's traced walkthrough exactly.
- Server model: trained each round on the open set with the voted labels (≠ −1); this model is what evaluation reports.
- Ablation flags, each altering only its own mechanism: `no_discriminating` (upload predictions for all open samples, no filtering), `no_voting` (server aggregates client predictions directly without majority vote), `simply_filtering` (threshold-only filtering, no discriminator model), fixed `threshold`, and soft-label modes (upload float32 [N_o, L] soft labels rounded to 8/6/4/2 decimals instead of hard labels; server averages instead of votes).
- Fixed judgment calls (ADR-8): distillation and discriminator training run 1 epoch per round; optimizers re-created each round; distillation loss is cross-entropy on hard labels.
- Per-round diagnostics returned for the metrics store: unfamiliar count per client, zero-vote sample count, vote agreement rate.
- Round driver runnable without any FL framework; deterministic under fb1's seeding.
- Unit tests cover: the vote against the solution.md walkthrough (including tie and zero-vote cases), an all-unfamiliar client round completing, a single-class client (Scenario 2 shape) completing all five steps, each ablation flag changing only its targeted behavior, and a 2-round micro-run.

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (ADR-4, ADR-8; Implementation Examples).
- Pure logic module: no Flower imports; data via dc1, models via mz1, config/seed via fb1.
- Full test coverage for all requirements.

## Interfaces
- Exposes client-round / vote / server-train / evaluate functions with payloads matching fb1's SSFL contract (int64 [N_o], −1 sentinel; soft modes float32 [N_o, L]); tr1 wires these to Flower Messages.
