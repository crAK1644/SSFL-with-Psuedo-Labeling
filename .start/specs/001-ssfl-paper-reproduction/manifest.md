---
title: "SSFL Paper Reproduction — Semisupervised Federated-Learning IDS on N-BaIoT"
status: pending
threshold: 0.90
max_iterations: 5
---
# Decomposition Manifest

## Units
- [ ] dc1: Mini-N-BaIoT data core (build, partition, load) — no dependencies
- [ ] mz1: Model zoo (Table I CNN, MLP, LSTM) — no dependencies
- [ ] fb1: Framework base (config, payload contract, durable metrics, seeding) — no dependencies
- [ ] fl1: FL method logic (FedAvg) — after: dc1, mz1, fb1
- [ ] fd1: FD method logic (federated distillation) — after: dc1, mz1, fb1
- [ ] ds1: DS-FL method logic (distillation-based semisupervised FL) — after: dc1, mz1, fb1
- [ ] ss1: SSFL method logic (proposed method + ablation flags) — after: dc1, mz1, fb1
- [ ] cm1: Communication-cost accounting — after: fb1
- [ ] tr1: Flower transport (ClientApp, ServerApp, method strategies) — after: fb1, fl1, fd1, ds1, ss1
- [ ] ts1: Aggregate smoke suite (pre-flight gate) — after: dc1, mz1, fl1, fd1, ds1, ss1
- [ ] rp1: Report generation (Tables II–IV, Figures 3–6) — after: fb1, cm1
- [ ] rn1: Runner CLI, timing pilot, campaign orchestration — after: tr1

## Execution Order
Group 1 (parallel): dc1, mz1, fb1
Group 2 (parallel): fl1, fd1, ds1, ss1, cm1
Group 3 (parallel): tr1, ts1, rp1
Group 4 (sequential): rn1

## Scenario Counts
25 scenarios total (12 P0, 13 P1) across 12 units; every unit has ≥1 P0. E2E stubs: pytest, one `e2e-stubs.md` per unit.
