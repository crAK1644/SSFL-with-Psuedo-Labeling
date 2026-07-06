---
id: fl1
title: FL method logic (FedAvg)
type: feature
dependencies: [dc1, mz1, fb1]
---
# FL method logic (FedAvg)

## Goal
Pure-Python FedAvg: client local training and sample-weighted parameter averaging per the paper's Eq. 1, framework-free.

## Requirements
- Client step: train the client's model on its private labeled data for the configured local epochs with the paper's hyperparameters (Adam lr 1e-4, batch 80, cross-entropy); return updated weights and sample count.
- Server step: aggregate client weights as the sample-count-weighted average (Eq. 1: w^s = Σ_k (N^k / N) w^k).
- A round driver usable without any FL framework: given client datasets and a model factory, execute one full round (all clients train → aggregate → return global weights) — this is what smoke tests exercise.
- Evaluation helper: score a given weight set on the test split (top-1 accuracy, and final-run macro-F1/precision/confusion matrix via fb1's metrics types).
- Deterministic under fb1's seeding discipline: identical inputs and seed ⇒ identical aggregated weights.
- Unit tests cover: weighted average correctness against a hand-computed 2-client example, a 2-round micro-run on a tiny subset showing decreasing training loss, determinism.

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (ADR-4: pure logic module, no Flower imports).
- Data access only through dc1 loaders; models only from mz1; config/seeding only from fb1.
- Full test coverage for all requirements.

## Interfaces
- Exposes client-step / aggregate / evaluate functions with ndarray inputs and outputs matching fb1's FL payload contract; tr1 wires these to Flower Messages.
