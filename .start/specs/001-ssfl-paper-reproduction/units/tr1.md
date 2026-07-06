---
id: tr1
title: Flower transport (ClientApp, ServerApp, method strategies)
type: feature
dependencies: [fb1, fl1, fd1, ds1, ss1]
---
# Flower transport (ClientApp, ServerApp, method strategies)

## Goal
The thin Flower 1.32.1 shell that runs any method's pure logic as a federated simulation: ClientApp dispatch, ServerApp with one Strategy per method, per-client persistence, and central evaluation each round.

## Requirements
- Flower pinned exactly at `flwr[simulation]==1.32.1`; only `flwr.app` / `flwr.clientapp` / `flwr.serverapp` imports (Message API); simulation configured for 27 or 89 supernodes per scenario.
- ClientApp `@app.train()` handler: resolves the method from run-config, loads the client's partition via its `partition-id`, loads/saves per-client model and optimizer state as disk checkpoints under `results/<run-id>/ckpt/` (Context.state carries scalars only), executes the method's client step, returns the method's payload as ArrayRecord.
- ServerApp `@app.main()`: builds the method's Strategy, runs `strategy.start(num_rounds, evaluate_fn=…)`; `evaluate_fn` scores the method's reportable model on the test set every round and feeds fb1's durable metrics store (round record appended and flushed before the next round starts).
- One Strategy subclass per method over a shared base: broadcasts the method's server→client payload, collects replies, delegates aggregation to the method's pure aggregate/vote function, trains the server model where the method defines one (DS-FL, SSFL).
- Client replies carrying errors are excluded from aggregation; the round record counts failed clients; the round completes.
- Round/flag values travel in ConfigRecord; arrays only in ArrayRecord, exactly per fb1's payload contract.
- After the final round: final metrics and confusion matrix written via fb1 (atomic); process exits non-zero on unrecoverable failure, zero on completion.
- An integration test (marked slow/optional, may use Ray) runs FL and SSFL for 2 rounds × small client count end-to-end through `flwr run` and asserts well-formed results directories.

## Constraints
- Conventions per `.start/specs/001-ssfl-paper-reproduction/solution.md` (ADR-1, ADR-2, ADR-3; Implementation Gotchas: run-config scalars only, Ray workers share no globals).
- All science stays in the method logic units — this unit contains no training mathematics beyond delegation.
- Device selection inside client/server code (Ray does not schedule MPS): clients default CPU, server evaluation may use MPS; `num_gpus` only meaningful on CUDA.
- Full test coverage for the non-Ray-dependent parts (dispatch, checkpoint round-trip, error exclusion).

## Interfaces
- Consumes: fb1 (config, payloads, metrics), method logic functions (fl1, fd1, ds1, ss1), dc1 loaders, mz1 models (transitively via method logic).
- Provides: the runnable Flower apps that rn1 launches via `flwr run`.
