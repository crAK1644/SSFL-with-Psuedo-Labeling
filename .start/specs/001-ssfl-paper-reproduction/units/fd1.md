---
id: fd1
title: FD method logic (federated distillation)
type: feature
dependencies: [dc1, mz1, fb1]
---
# FD method logic (federated distillation)

## Goal
Pure-Python Federated Distillation per the paper's Eqs. 2–4: clients exchange per-class average logit vectors instead of weights.

## Requirements
- Client step: train locally on private data (paper hyperparameters), then compute the local-average logit vector per class label (Eqs. 2–3); a class absent from the client's data yields a zero vector.
- Server step: for each client and class, compute the new local-average logit target per Eq. 4 — the global average excluding the client's own contribution: ȳ^{k,l} ← (N^l × ȳ^{s,l} − ȳ^{k,l}) / (N^l − 1).
- Client distillation step: continue local training using the received per-class logit targets combined with ground-truth labels, as the paper describes for FD.
- Round driver runnable without any FL framework (for smoke tests), plus the same evaluation helper contract as fl1. Since FD has no global model, evaluation follows the paper's convention: report the best-performing client model's test accuracy.
- Deterministic under fb1's seeding discipline.
- Unit tests cover: per-class averaging with an absent class (zero-vector rule), the Eq. 4 exclude-self computation against a hand-computed example, a 2-round micro-run completing on a tiny subset.

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (ADR-4; Implementation Gotchas: FD zero-vector rule).
- Pure logic module: no Flower imports; data via dc1, models via mz1, config/seed via fb1.
- Full test coverage for all requirements.

## Interfaces
- Exposes client-step / aggregate / distill / evaluate functions with payloads matching fb1's FD contract (float32 [L,L]); tr1 wires these to Flower Messages.
