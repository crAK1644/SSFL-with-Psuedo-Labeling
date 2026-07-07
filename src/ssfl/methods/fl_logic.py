"""Pure FedAvg logic (paper Eq. 1) — framework-free (ADR-4).

Client local training, sample-count-weighted parameter averaging, a round
driver and an evaluation helper. No Flower/Ray imports anywhere: the Flower
transport layer wires these functions to Messages later.

Weight payloads follow the FL contract in :mod:`ssfl.methods.payloads`:
a list of float32 ndarrays (state-dict order) plus a sample count.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn

from ssfl.config import derive_seed
from ssfl.methods._shared import classification_metrics
from ssfl.models import resolve_device

WeightList = list[np.ndarray]
ModelFn = Callable[[], nn.Module]

#: Paper hyperparameters (CON-2): Adam lr 1e-4, batch 80, 5 local epochs.
LR = 1e-4
BATCH = 80
LOCAL_EPOCHS = 5

__all__ = [
    "LR",
    "BATCH",
    "LOCAL_EPOCHS",
    "get_weights",
    "set_weights",
    "init_weights",
    "client_step",
    "aggregate",
    "run_round",
    "predict",
    "evaluate",
    "classification_metrics",
    "evaluate_full",
]

_EVAL_BATCH = 512


# ---------------------------------------------------------------------------
# Weight payloads: list of float32 ndarrays in state-dict order (payloads.py)
# ---------------------------------------------------------------------------


def get_weights(model: nn.Module) -> WeightList:
    """Extract model weights as float32 ndarrays (transport payload form)."""
    return [
        t.detach().cpu().numpy().astype(np.float32, copy=True)
        for t in model.state_dict().values()
    ]


def set_weights(model: nn.Module, weights: Sequence[np.ndarray]) -> None:
    """Load a weight payload back into a model (strict, shape-checked)."""
    keys = list(model.state_dict().keys())
    if len(weights) != len(keys):
        raise ValueError(
            f"model has {len(keys)} state tensors but payload has {len(weights)}"
        )
    state = {k: torch.as_tensor(np.asarray(w)) for k, w in zip(keys, weights, strict=True)}
    model.load_state_dict(state, strict=True)


def init_weights(model_fn: ModelFn, *, seed: int) -> WeightList:
    """Deterministically initialized global weights for round 0."""
    torch.manual_seed(seed)
    return get_weights(model_fn())


# ---------------------------------------------------------------------------
# Client step: local supervised training on private labeled data
# ---------------------------------------------------------------------------


def client_step(
    model_fn: ModelFn,
    weights: Sequence[np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = LOCAL_EPOCHS,
    lr: float = LR,
    batch: int = BATCH,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[WeightList, int, float]:
    """Train the global weights on one client's private data.

    Adam + cross-entropy per the paper; `seed` (derived via
    ssfl.config.derive_seed by the round driver) fixes both the model-buffer
    RNG and the shuffle order, so identical inputs reproduce identical
    outputs bit-for-bit on CPU.

    Returns ``(updated_weights, num_examples, mean_train_loss)``.
    """
    n = len(X)
    if n == 0:
        raise ValueError("client_step requires a non-empty dataset")

    dev = resolve_device(device)
    torch.manual_seed(seed)
    model = model_fn().to(dev)
    set_weights(model, weights)

    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long, device=dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    shuffle_rng = np.random.default_rng(seed)

    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        order = shuffle_rng.permutation(n)
        for start in range(0, n, batch):
            idx = torch.as_tensor(order[start : start + batch], device=dev)
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return get_weights(model), n, float(np.mean(losses))


def aggregate(
    weights_list: Sequence[WeightList], num_examples: Sequence[int]
) -> WeightList:
    """Sample-count-weighted average of client weights (Eq. 1).

    ``w^s = sum_k (N^k / N) * w^k`` with ``N = sum_k N^k``. Accumulates in
    float64 and returns float32 arrays per the payload contract.
    """
    if len(weights_list) != len(num_examples):
        raise ValueError(
            f"got {len(weights_list)} weight sets but {len(num_examples)} sample counts"
        )
    if not weights_list:
        raise ValueError("no client results to aggregate")
    total = sum(int(n) for n in num_examples)
    if total <= 0:
        raise ValueError(f"total sample count must be positive, got {total}")

    acc = [np.zeros(w.shape, dtype=np.float64) for w in weights_list[0]]
    for weights, n in zip(weights_list, num_examples, strict=True):
        frac = int(n) / total
        for slot, w in zip(acc, weights, strict=True):
            slot += frac * np.asarray(w, dtype=np.float64)
    return [slot.astype(np.float32) for slot in acc]


# ---------------------------------------------------------------------------
# Round driver: one full FedAvg round without any FL framework (ADR-4)
# ---------------------------------------------------------------------------


def run_round(
    model_fn: ModelFn,
    global_weights: Sequence[np.ndarray],
    client_data: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    round_num: int,
    run_seed: int,
    epochs: int = LOCAL_EPOCHS,
    lr: float = LR,
    batch: int = BATCH,
    device: str = "cpu",
) -> tuple[WeightList, dict]:
    """Execute one full FedAvg round: all clients train, then Eq. 1 aggregate.

    ``client_data[k]`` is client k's private ``(X, y)``; each client trains
    from the same ``global_weights`` with the seed
    ``derive_seed(run_seed, client_id=k, round_num=round_num)`` (the
    framework's seeding discipline), so identical inputs and run_seed yield
    bit-identical aggregated weights on CPU.

    Returns ``(new_global_weights, info)`` where ``info`` holds per-client
    ``client_losses`` / ``num_examples`` and the sample-weighted mean
    ``train_loss`` of the round.
    """
    if not client_data:
        raise ValueError("run_round requires at least one client dataset")

    weights_list: list[WeightList] = []
    num_examples: list[int] = []
    client_losses: list[float] = []
    for client_id, (X, y) in enumerate(client_data):
        w, n, loss = client_step(
            model_fn,
            global_weights,
            X,
            y,
            epochs=epochs,
            lr=lr,
            batch=batch,
            seed=derive_seed(run_seed, client_id=client_id, round_num=round_num),
            device=device,
        )
        weights_list.append(w)
        num_examples.append(n)
        client_losses.append(loss)

    info = {
        "round": round_num,
        "num_examples": num_examples,
        "client_losses": client_losses,
        "train_loss": float(np.average(client_losses, weights=num_examples)),
    }
    return aggregate(weights_list, num_examples), info


# ---------------------------------------------------------------------------
# Evaluation: top-1 accuracy per round, full metrics for the final run
# ---------------------------------------------------------------------------


def predict(
    model_fn: ModelFn,
    weights: Sequence[np.ndarray],
    X: np.ndarray,
    *,
    batch: int = _EVAL_BATCH,
    device: str = "cpu",
) -> np.ndarray:
    """Argmax class predictions (int64) for a weight payload on X."""
    dev = resolve_device(device)
    model = model_fn().to(dev)
    set_weights(model, weights)
    model.eval()
    if len(X) == 0:
        return np.zeros(0, dtype=np.int64)
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = Xt[start : start + batch]
            preds.append(model(xb).argmax(dim=1).cpu().numpy().astype(np.int64))
    return np.concatenate(preds)


def evaluate(
    model_fn: ModelFn,
    weights: Sequence[np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch: int = _EVAL_BATCH,
    device: str = "cpu",
) -> float:
    """Top-1 accuracy of a weight payload on (X, y)."""
    preds = predict(model_fn, weights, X, batch=batch, device=device)
    return float(np.mean(preds == np.asarray(y)))


def evaluate_full(
    model_fn: ModelFn,
    weights: Sequence[np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    batch: int = _EVAL_BATCH,
    device: str = "cpu",
) -> dict:
    """Final-run metrics of a weight payload on the test split.

    Everything except ``confusion_matrix`` is JSON-ready for
    MetricsStore.write_final; the matrix goes to save_confusion_matrix.
    """
    preds = predict(model_fn, weights, X, batch=batch, device=device)
    return classification_metrics(y, preds, num_classes=num_classes)
