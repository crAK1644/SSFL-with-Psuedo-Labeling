---
id: ds1
title: DS-FL method logic (distillation-based semisupervised FL)
type: feature
dependencies: [dc1, mz1, fb1]
---
# DS-FL method logic (distillation-based semisupervised FL)

## Goal
Pure-Python DS-FL per the paper's Eqs. 5–10: clients exchange logits on the shared open set; the server averages them and applies Entropy Reduction Aggregation (ERA); clients and server distill on the result.

## Requirements
- Client step: train on private labeled data (paper hyperparameters), then predict logits for every open-set sample (Eq. 5).
- Server step: average client logit matrices (Eq. 6), then apply ERA — a temperature softmax with T < 1 that sharpens the distribution / reduces entropy (Eqs. 7–8). The temperature must be a config value; sharpening direction (T < 1) is load-bearing.
- Client distillation: train the local model on the open set against the global soft labels (Eq. 9). Server model: same distillation on a server-held model (Eq. 10); this server model is what evaluation reports each round.
- Round driver runnable without any FL framework, evaluation helper contract as fl1.
- Deterministic under fb1's seeding discipline.
- Unit tests cover: ERA sharpening (output entropy strictly lower than input entropy for a non-degenerate distribution), the averaging step, a 2-round micro-run on a tiny subset, and open-set label integrity (no ground-truth open labels consumed anywhere).

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (ADR-4; Implementation Gotchas: ERA direction T<1).
- Pure logic module: no Flower imports; data via dc1, models via mz1, config/seed via fb1.
- Full test coverage for all requirements.

## Interfaces
- Exposes client-step / aggregate / distill / evaluate functions with payloads matching fb1's DS-FL contract (float32 [N_o, L]); tr1 wires these to Flower Messages.
