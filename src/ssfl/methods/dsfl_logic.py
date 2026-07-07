"""DS-FL: Distillation-Based Semisupervised Federated Learning (Eqs. 5-10).

Pure-Python method logic (ADR-4: no Flower imports) for Itahara et al.'s
DS-FL as described by Zhao et al. (2023):

1. Each client trains on its private labeled data, then predicts logits for
   every shared open-set sample (Eq. 5) — :func:`client_step`.
2. The server averages the client logit matrices elementwise (Eq. 6,
   :func:`average_logits`) and applies Entropy Reduction Aggregation — a
   temperature softmax with T < 1 that SHARPENS the distribution (Eqs. 7-8,
   :func:`era`). :func:`aggregate` does both.
3. Clients and a server-held model distill on the resulting global soft
   labels (Eqs. 9-10) — :func:`distill`. The server model is what evaluation
   reports each round.

ERA direction (Implementation Gotchas): Eq. 8's T < 1 *reduces* entropy;
T > 1 (the usual distillation softening) would invert the mechanism, so
:func:`era` rejects it outright.

Payloads follow ssfl.methods.payloads' DS-FL contract: float32 [N_o, L]
both directions. Open-set data is consumed strictly label-free via
``ssfl.data.load_open`` (CON-6).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from ssfl.config import derive_seed
from ssfl.methods._shared import classification_metrics
from ssfl.models import NUM_CLASSES, resolve_device

#: Eq. 8 temperature. Must satisfy 0 < T < 1 so aggregation sharpens.
DEFAULT_ERA_TEMPERATURE = 0.1

#: Distillation trains 1 epoch/round (ADR-8: local_epochs applies to the
#: supervised step only).
DISTILL_EPOCHS = 1

__all__ = [
    "DEFAULT_ERA_TEMPERATURE",
    "DISTILL_EPOCHS",
    "softmax",
    "average_logits",
    "era",
    "aggregate",
    "client_step",
    "distill",
    "run_round",
    "predict_logits",
    "evaluate",
    "final_metrics",
    "classification_metrics",
]


# ---------------------------------------------------------------------------
# Pure math: softmax, Eq. 6 averaging, Eqs. 7-8 ERA
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable plain softmax (Eq. 7 with T = 1)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def average_logits(client_logits: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Eq. 6: elementwise mean of the K client logit matrices.

    ``client_logits``: K arrays of shape [N_o, L] (or one [K, N_o, L] array).
    Returns float32 [N_o, L].
    """
    if isinstance(client_logits, np.ndarray) and client_logits.ndim == 3:
        stack = client_logits
    else:
        mats = list(client_logits)
        if not mats:
            raise ValueError("average_logits needs at least one client logit matrix")
        shapes = {np.shape(m) for m in mats}
        if len(shapes) != 1:
            raise ValueError(
                f"client logit matrices must share one shape, got {sorted(shapes)}"
            )
        stack = np.stack([np.asarray(m, dtype=np.float32) for m in mats])
    if stack.shape[0] == 0:
        raise ValueError("average_logits needs at least one client logit matrix")
    return stack.mean(axis=0, dtype=np.float64).astype(np.float32)


def era(avg_logits: np.ndarray, temperature: float = DEFAULT_ERA_TEMPERATURE) -> np.ndarray:
    """Eqs. 7-8: Entropy Reduction Aggregation — sharpening temperature softmax.

    ``S(p_hat | T)_l = exp(p_hat_l / T) / sum_l' exp(p_hat_l' / T)`` with
    T < 1, so the output distribution has strictly lower entropy than the
    plain softmax of the same logits. The direction is load-bearing:
    temperatures >= 1 soften instead of sharpen and are rejected.
    """
    if not 0.0 < temperature < 1.0:
        raise ValueError(
            f"ERA requires 0 < temperature < 1 (Eq. 8 sharpens; T >= 1 would "
            f"invert the mechanism and raise entropy), got {temperature!r}"
        )
    out = softmax(np.asarray(avg_logits, dtype=np.float64) / temperature)
    return out.astype(np.float32)


def aggregate(
    client_logits: Sequence[np.ndarray] | np.ndarray,
    temperature: float = DEFAULT_ERA_TEMPERATURE,
) -> np.ndarray:
    """Server step: Eq. 6 average, then Eq. 8 ERA.

    Returns the global soft labels, float32 [N_o, L] — the DS-FL
    server->client payload.
    """
    return era(average_logits(client_logits), temperature=temperature)


# ---------------------------------------------------------------------------
# Torch plumbing (transport-friendly: ndarrays in, ndarrays out)
# ---------------------------------------------------------------------------


def _as_tensor(a: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(dtype=dtype, device=device)


def _batches(n: int, batch: int, rng: np.random.Generator | None):
    """Yield index batches; shuffled when an rng is given, sequential otherwise."""
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch):
        yield order[start : start + batch]


def _train(
    model: nn.Module,
    X: np.ndarray,
    targets: np.ndarray,
    *,
    soft: bool,
    lr: float,
    batch: int,
    epochs: int,
    seed: int,
    device: str | torch.device,
) -> None:
    """Shared Adam training loop: hard cross-entropy or soft-label distillation.

    The optimizer is created fresh per call (ADR-8: re-created each round).
    All shuffling/init randomness derives from ``seed`` alone.
    """
    dev = resolve_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(dev).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = _as_tensor(X, torch.float32, dev)
    tt = _as_tensor(targets, torch.float32 if soft else torch.int64, dev)
    hard_ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for idx in _batches(len(Xt), batch, rng):
            bidx = torch.from_numpy(idx).to(dev)
            logits = model(Xt[bidx])
            if soft:
                logp = torch.log_softmax(logits, dim=1)
                loss = -(tt[bidx] * logp).sum(dim=1).mean()
            else:
                loss = hard_ce(logits, tt[bidx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def predict_logits(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch: int = 256,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Model logits for every row of X, float32 [N, L]."""
    dev = resolve_device(device)
    model.to(dev).eval()
    if len(X) == 0:
        return np.zeros((0, getattr(model, "num_classes", NUM_CLASSES)), dtype=np.float32)
    Xt = _as_tensor(X, torch.float32, dev)
    outs = []
    with torch.no_grad():
        for idx in _batches(len(Xt), batch, None):
            outs.append(model(Xt[torch.from_numpy(idx).to(dev)]).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


# ---------------------------------------------------------------------------
# Eq. 5: client step — Eqs. 9-10: distillation
# ---------------------------------------------------------------------------


def client_step(
    model: nn.Module,
    X_private: np.ndarray,
    y_private: np.ndarray,
    open_X: np.ndarray,
    *,
    lr: float = 1e-4,
    batch: int = 80,
    local_epochs: int = 5,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """One DS-FL client step: supervised training + open-set logits (Eq. 5).

    Trains ``model`` in place on the private labeled data (Adam, cross
    entropy), then returns its logits on the open set — the client->server
    payload, float32 [N_o, L]. Note ``open_X`` is X only: no open labels
    exist on the client (CON-6).
    """
    if len(X_private) == 0:
        raise ValueError("client_step requires a non-empty private dataset")
    _train(
        model, X_private, y_private, soft=False,
        lr=lr, batch=batch, epochs=local_epochs, seed=seed, device=device,
    )
    return predict_logits(model, open_X, batch=batch, device=device)


def distill(
    model: nn.Module,
    open_X: np.ndarray,
    global_soft_labels: np.ndarray,
    *,
    lr: float = 1e-4,
    batch: int = 80,
    epochs: int = DISTILL_EPOCHS,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> None:
    """Eq. 9 (client) / Eq. 10 (server model): distill on the ERA soft labels.

    Trains ``model`` in place against ``global_soft_labels`` (float32
    [N_o, L], rows summing to 1) with soft-target cross-entropy. Consumes no
    ground-truth labels for the open samples — only the aggregated logits.
    """
    _train(
        model, open_X, global_soft_labels, soft=True,
        lr=lr, batch=batch, epochs=epochs, seed=seed, device=device,
    )


# ---------------------------------------------------------------------------
# Round driver: one full DS-FL round, no FL framework (ADR-4)
# ---------------------------------------------------------------------------


def run_round(
    client_models: Sequence[nn.Module],
    server_model: nn.Module,
    clients_data: Sequence[tuple[np.ndarray, np.ndarray]],
    open_X: np.ndarray,
    *,
    round_num: int,
    run_seed: int,
    lr: float = 1e-4,
    batch: int = 80,
    local_epochs: int = 5,
    era_temperature: float = DEFAULT_ERA_TEMPERATURE,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """One DS-FL round, driven without any FL framework.

    Composes the primitives exactly as the transport layer will:

    1. every client trains on its private data and uploads open-set logits
       (Eq. 5, :func:`client_step`);
    2. the server averages and sharpens them (Eqs. 6-8, :func:`aggregate`);
    3. every client and the server-held model distill on the global soft
       labels (Eqs. 9-10, :func:`distill`). ``server_model`` — updated in
       place — is what evaluation reports each round.

    Seeding discipline: every stochastic step keys off
    ``ssfl.config.derive_seed(run_seed, client_id, round_num)``; the server
    model uses ``client_id = len(client_models)``, which no real client id
    (0..K-1) can collide with. Each client's distillation step (step 3) uses
    that seed plus 1, so it draws an independent shuffle from its own
    client_step (step 1) instead of replaying the same permutation. Identical
    inputs reproduce identical outputs bit-for-bit on CPU.

    Returns the global soft labels, float32 [N_o, L].
    """
    if len(client_models) != len(clients_data):
        raise ValueError(
            f"got {len(client_models)} client models but "
            f"{len(clients_data)} client datasets"
        )

    def seed_for(client_id: int) -> int:
        return derive_seed(run_seed, client_id=client_id, round_num=round_num)

    logits = [
        client_step(
            model, X, y, open_X,
            lr=lr, batch=batch, local_epochs=local_epochs,
            seed=seed_for(k), device=device,
        )
        for k, (model, (X, y)) in enumerate(zip(client_models, clients_data))
    ]
    global_soft_labels = aggregate(logits, temperature=era_temperature)

    for k, model in enumerate(client_models):
        distill(model, open_X, global_soft_labels,
                lr=lr, batch=batch, seed=seed_for(k) + 1, device=device)
    distill(server_model, open_X, global_soft_labels,
            lr=lr, batch=batch, seed=seed_for(len(client_models)) + 1, device=device)
    return global_soft_labels


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch: int = 256,
    device: str | torch.device = "cpu",
) -> float:
    """Top-1 accuracy of ``model`` on (X, y)."""
    preds = predict_logits(model, X, batch=batch, device=device).argmax(axis=1)
    return float((preds == np.asarray(y)).mean())


def final_metrics(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    batch: int = 256,
    device: str | torch.device = "cpu",
) -> dict:
    """Final-report metrics: accuracy, macro-F1/precision, confusion matrix."""
    preds = predict_logits(model, X, batch=batch, device=device).argmax(axis=1)
    return classification_metrics(np.asarray(y), preds, num_classes=num_classes)
