---
unit: ss1
feature: Ablation variants (Figs. 4–6)
priority: P1
---
# Each ablation flag alters only its own mechanism

## Scenario
Run one identically-seeded SSFL round four times, varying one flag each: baseline, `no_discriminating`, `simply_filtering`, and soft-label mode `soft2`. Compare the uploaded payloads and the aggregation behavior.

## Expected
Baseline: int64 hard labels containing −1 entries.
`no_discriminating`: int64 hard labels with zero −1 entries (all open samples predicted), other steps unchanged.
`simply_filtering`: −1 entries decided purely by the confidence threshold; no discriminator model is trained (its checkpoint/state is absent).
`soft2`: payload is float32 [N_o, L] with every value equal to its 2-decimal rounding; aggregation averages instead of voting.
With `no_voting`: the server produces global labels without the majority-vote path while the client side is unchanged.
