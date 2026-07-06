---
unit: rn1
feature: Runner CLI
priority: P0
---
# The CLI rejects bad input and launches valid runs

## Scenario
(a) Run `uv run python -m ssfl.run --method bogus`. (b) Run `uv run python -m ssfl.run --method fl --no-voting`. (c) With the cache absent, run a valid command. (d) With the cache built, run `uv run python -m ssfl.run --method fl --scenario 1 --seed 0 --rounds 2` with a small client count.

## Expected
(a) and (b) exit non-zero before any simulation starts, printing the allowed values.
(c) exits non-zero with a message telling the user to run the data builder.
(d) exits 0, prints the run-id at start and the results path at the end, and one progress line per round appears in the output.
