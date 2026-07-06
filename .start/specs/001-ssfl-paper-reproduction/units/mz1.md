---
id: mz1
title: Model zoo (Table I CNN, MLP, LSTM)
type: feature
dependencies: []
---
# Model zoo (Table I CNN, MLP, LSTM)

## Goal
PyTorch models matching the paper: the Table I Conv1D CNN with classifier (11-out) and discriminator (2-out) heads, plus MLP and LSTM comparison models, all device-agnostic.

## Requirements
- CNN per Table I of the paper, input [B, 23, 5]: Conv1D layers 1–4 (64 filters, kernel 3, stride 1) → layers 5–6 (128, 3, 1) → layer 7 (128, 3, 2) → layer 8 (128, 3, 2) → flatten → fully-connected MLP head ending in a Linear(…, 128) then output layer; output size 11 (classifier) or 2 (discriminator), selected at construction.
- Intermediate activation shapes must match Table I exactly for batch 80: (80,64,5) after layers 1–4, (80,128,5) after 5–6, (80,128,3) after 7, (80,128,2) after 8, (80,128) at the FC layer.
- MLP and LSTM models for the same input/output signature (23×5 in, 11 classes out), sized comparably to the paper's comparison models.
- A device helper resolving "auto" → cuda if available, else mps, else cpu; every model trains one step successfully on each available device (LSTM may require MPS fallback — document `PYTORCH_ENABLE_MPS_FALLBACK=1`).
- Unit tests assert layer shapes, output dimensions, and a successful forward+backward+step on the resolved device.

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (Directory Map: `src/ssfl/models.py`).
- No Flower imports; torch only.
- Full test coverage for all requirements.

## Interfaces
- Model constructors and the device helper are consumed by fl1/fd1/ds1/ss1, tr1, and ts1.
