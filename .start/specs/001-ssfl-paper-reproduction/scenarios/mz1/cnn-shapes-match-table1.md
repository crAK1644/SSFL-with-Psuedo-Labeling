---
unit: mz1
feature: Table I CNN architecture
priority: P0
---
# CNN activation shapes match Table I

## Scenario
Construct the CNN classifier and the CNN discriminator through the public model API. Feed a batch of 80 random samples shaped [80, 23, 5] through each, capturing intermediate activations.

## Expected
After convolution layers 1–4 the activation shape is (80, 64, 5).
After layers 5–6 it is (80, 128, 5); after layer 7 (80, 128, 3); after layer 8 (80, 128, 2).
The fully-connected layer produces (80, 128).
The classifier head outputs (80, 11); the discriminator head outputs (80, 2).
