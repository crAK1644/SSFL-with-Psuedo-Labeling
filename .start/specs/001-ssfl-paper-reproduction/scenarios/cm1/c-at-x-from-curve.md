---
unit: cm1
feature: C@x metrics from accuracy curves
priority: P1
---
# C@50/C@75/C@Top-Acc read the right rounds; unreached targets are honest

## Scenario
Feed the accounting API a synthetic run whose per-round accuracy curve is 30% (rounds 1–9), 55% (rounds 10–99), 70% (rounds 100–200), with a known per-round cost of 1 MB.

## Expected
C@50 = 10 MB (first round reaching ≥50% is round 10).
C@Top-Acc = 100 MB (first round reaching the run's own maximum, 70%).
C@75 is reported as explicitly unreached (no fabricated number), and the report value distinguishes "unreached" from 0.
