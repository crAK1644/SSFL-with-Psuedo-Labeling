# Specification: 001-ssfl-paper-reproduction

## Status

| Field | Value |
|-------|-------|
| **Created** | 2026-07-06 |
| **Current Phase** | Ready |
| **Last Updated** | 2026-07-07 |

## Documents

| Document | Status | Notes |
|----------|--------|-------|
| requirements.md | completed | Approved 2026-07-06; 27 ACs, ±3-pt fidelity target |
| solution.md | completed | Approved 2026-07-06; 8 ADRs confirmed |

**Status values**: `pending` | `in_progress` | `completed` | `skipped`

## Decomposition

| Field | Value |
|-------|-------|
| **Tier** | Factory |
| **Status** | completed |

**Tier values**: `Direct` (no artifacts) | `Incremental` (plan/) | `Factory` (manifest.md + units/ + scenarios/) | `None` (not yet chosen)

For Incremental tier, see `plan/README.md`.
For Factory tier, see `manifest.md`, `units/`, `scenarios/`.
For Direct tier, no decomposition artifacts are produced — implement-direct reads requirements.md and solution.md directly.

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-06 | Reproduce full paper (Table II–IV + ablations) | User chose full reproduction during brainstorm |
| 2026-07-06 | Flower (Ray simulation) + PyTorch CNN | User's explicit stack choice; payloads (logits/hard labels) carried as ndarray Parameters via custom Strategies |
| 2026-07-06 | Compute target deferred | Device-agnostic code; timing run (SSFL scenario 1) decides local Mac vs cloud GPU |
| 2026-07-06 | Pin flwr[simulation]==1.32.1, Message API only | Research: heavy 1.x API churn; legacy start_simulation deprecated; ArrayRecord is the sanctioned carrier for label/logit payloads |
| 2026-07-06 | Fresh uv env, Python 3.12 | Installed flwr 1.9.0 broken vs numpy 2.4.6; 3.12 safest for Ray wheels on macOS |
| 2026-07-06 | Preprocess CSVs once to cached arrays | 7.6 GB raw data; avoid re-parsing per run |
| 2026-07-06 | Local MPS feasible (~2.7–9 h/run, benchmarked) | M4 micro-benchmark: 3.8 ms/step MPS; Ray doesn't schedule MPS — select device in-code, num_gpus=0 |
| 2026-07-06 | Requirements approved (±3-pt tolerance, first-1000 sampling) | User approved without revisions; continue to Solution |
| 2026-07-06 | Solution approved; ADR-1…8 confirmed | Layered harness (pure logic + Flower transport), disk client state, CPU-parallel clients, restart-from-zero, paper-silent judgment calls pinned |
| 2026-07-06 | Decomposition tier: Factory | Classifier recommendation: Factory (10 features, ~9 components, parallel method workstreams). Accepted. |
| 2026-07-07 | Factory artifacts approved (12 units, 25 scenarios, manifest) | Units, scenarios/stubs, and manifest each user-approved; threshold 0.90, max 5 iterations; 4 execution groups |

## Context

Full reproduction of Zhao et al., "Semisupervised Federated-Learning-Based Intrusion Detection Method for IoT" (IEEE IoT-J 2023) on the N-BaIoT dataset already in `data/`. Approved brainstorm design: mini-N-BaIoT builder (1000 samples/device-category, 70/10/20 private/open/test, min-max norm, 115→23×5), Table I Conv1D CNN + MLP/LSTM baselines, four federated methods (FL, FD, DS-FL, SSFL) as custom Flower Strategies, SSFL round = distill(t−1) → train classifier → train discriminator (median-confidence threshold) → filter → upload hard labels → server vote + server-model training; run.py CLI with ablation flags, paper hyperparameters (Adam 1e-4, batch 80, 5 local epochs, 200 rounds), analytic communication accounting, report.py for Tables II–IV / Figs 3–5, pytest smoke suite. ~30 runs total.

---
*This file is managed by the specify-meta skill.*
