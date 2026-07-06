---
unit: mz1
feature: Device-agnostic model zoo
priority: P1
---
# Every model completes a training step on the resolved device

## Scenario
Resolve the device with the helper's "auto" mode on the current machine. For each of CNN (11-class), MLP, and LSTM: run one forward pass, cross-entropy loss, backward pass, and Adam optimizer step on a [80, 23, 5] batch.

## Expected
The helper resolves to "mps" on this machine (cuda absent, MPS present).
Each of the three models completes the step without raising, and produces a finite loss value.
Forcing device "cpu" also completes for all three models.
