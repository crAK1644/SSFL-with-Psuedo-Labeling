"""Federated Distillation (FD) — pure client/server logic, Eqs. 2-4.

Jeong et al.'s FD as described by Zhao et al. (2023): instead of weights,
clients exchange per-class average logit vectors.

- Eqs. 2-3 (client upload): after local training, the client averages its
  model's logit vectors over its own samples of each label l. A class absent
  from the client's data yields a **zero vector** (payload contract /
  Implementation Gotchas: FD zero-vector rule).
- Eq. 4 (server aggregate): each client k receives, for every class l, the
  average over the other contributing clients:
  ``(N^l * ybar_s_l - ybar_k_l) / (N^l - 1)`` where N^l counts clients whose
  upload row for l is non-zero.

No Flower/Ray imports (ADR-4): everything here takes/returns plain ndarrays
plus torch modules, so the transport shell only routes arrays.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ssfl.config import derive_seed, make_rng
from ssfl.methods._shared import classification_metrics
from ssfl.models import NUM_CLASSES, resolve_device

__all__ = [
    "per_class_avg_logits",
    "fd_aggregate",
    "fd_client_step",
    "fd_distill_step",
    "evaluate_model",
    "classification_metrics",
    "evaluate_model_full",
    "run_fd",
]

#: Paper hyperparameter defaults (CON-2): Adam lr 1e-4, batch 80, 5 epochs.
DEFAULT_LR = 1e-4
DEFAULT_BATCH = 80
DEFAULT_LOCAL_EPOCHS = 5
#: Weight of the distillation term; 1 epoch/round for distillation per ADR-8.
DEFAULT_GAMMA = 1.0
DEFAULT_DISTILL_EPOCHS = 1

_EVAL_BATCH = 512


def _resolve(device: str | torch.device) -> torch.device:
    """Resolve a device argument (supports ``"auto"`` via ssfl.models)."""
    return resolve_device(device) if isinstance(device, str) else torch.device(device)


def _batches(n: int, batch: int, rng: np.random.Generator):
    """Deterministic shuffled mini-batch index arrays."""
    perm = rng.permutation(n)
    for start in range(0, n, batch):
        yield perm[start : start + batch]


def _train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn,
    *,
    lr: float,
    batch: int,
    epochs: int,
    device: str | torch.device,
    rng: np.random.Generator,
) -> None:
    """Local Adam/mini-batch loop. ``loss_fn(logits, yb, idx) -> scalar``.

    The optimizer is created fresh per call (ADR-8: optimizers re-created
    each round).
    """
    device = _resolve(device)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(np.ascontiguousarray(X), device=device)
    yt = torch.as_tensor(np.ascontiguousarray(y), device=device)
    for _ in range(epochs):
        for idx in _batches(len(X), batch, rng):
            bidx = torch.from_numpy(idx).to(device)
            loss = loss_fn(model(Xt[bidx]), yt[bidx], idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
    opt.zero_grad(set_to_none=True)


@torch.no_grad()
def _forward_logits(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch: int,
    device: str | torch.device,
) -> np.ndarray:
    """Batched eval-mode forward pass; float32 [N, L] logits on CPU."""
    device = _resolve(device)
    model.to(device).eval()
    if len(X) == 0:
        return np.zeros((0, getattr(model, "num_classes", NUM_CLASSES)), np.float32)
    Xt = torch.as_tensor(np.ascontiguousarray(X), device=device)
    out = []
    for start in range(0, len(X), batch):
        out.append(model(Xt[start : start + batch]).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def per_class_avg_logits(
    logits: np.ndarray, labels: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Eqs. 2-3: local-average logit vector per class label.

    ``logits`` is float [N, L], ``labels`` int [N]. Returns float32 [L, L]
    whose row l is the mean logit vector over samples with label l, or a zero
    vector when the class is absent (FD zero-vector rule).
    """
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels)
    out = np.zeros((num_classes, num_classes), dtype=np.float32)
    for label in np.unique(labels):
        out[label] = logits[labels == label].mean(axis=0)
    return out


def fd_aggregate(local_class_logits: np.ndarray) -> np.ndarray:
    """Eq. 4: per-client distillation targets, excluding own contributions.

    ``local_class_logits`` is float32 [K, L, L] — the stacked client uploads
    (row l of client k is its Eq. 3 average, zero when class l is absent).

    Returns float32 [K, L, L]: for client k and class l,
    ``(N^l * ybar_s_l - ybar_k_l) / (N^l - 1)`` over the N^l clients whose
    row l is non-zero. Edge cases (paper-silent, pinned here):

    - client k did not contribute to class l -> the plain average over the
      N^l contributors (nothing of its own to exclude);
    - N^l == 1 and client k is the sole contributor -> zero vector (no other
      client to learn from; the distill step skips zero target rows);
    - N^l == 0 -> zero vector for everyone.
    """
    uploads = np.asarray(local_class_logits, dtype=np.float32)
    if uploads.ndim != 3 or uploads.shape[1] != uploads.shape[2]:
        raise ValueError(
            f"expected stacked uploads of shape [K, L, L], got {uploads.shape}"
        )
    contributed = np.any(uploads != 0, axis=2)          # [K, L]
    n_contrib = contributed.sum(axis=0)                 # [L] = N^l
    total = uploads.sum(axis=0)                         # [L, L] = N^l * ybar_s_l

    targets = np.zeros_like(uploads)
    for l in range(uploads.shape[1]):
        n = int(n_contrib[l])
        if n == 0:
            continue
        for k in range(uploads.shape[0]):
            if contributed[k, l]:
                if n > 1:
                    targets[k, l] = (total[l] - uploads[k, l]) / (n - 1)
            else:
                targets[k, l] = total[l] / n
    return targets


# ---------------------------------------------------------------------------
# Client steps: local supervised training + upload, then distillation
# ---------------------------------------------------------------------------


def fd_client_step(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int = NUM_CLASSES,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    seed: int = 0,
    client_id: int = 0,
    round_num: int = 0,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Local training + Eqs. 2-3 upload for one client and round.

    Trains ``model`` in place (Adam + cross-entropy, paper hyperparameters),
    then returns the per-class average logit matrix (float32 [L, L], zero row
    for absent classes) — the ``local_class_logits`` payload in
    :mod:`ssfl.methods.payloads`. The shuffle stream is keyed on
    ``(seed, client_id, round_num)`` via ssfl.config, so identical inputs
    reproduce identical uploads bit-for-bit on CPU.
    """
    if len(X) == 0:
        raise ValueError("fd_client_step requires a non-empty dataset")
    rng = make_rng(seed, client_id=client_id, round_num=round_num)
    _train(
        model, X, y,
        lambda logits, yb, idx: F.cross_entropy(logits, yb),
        lr=lr, batch=batch, epochs=local_epochs, device=device, rng=rng,
    )
    logits = _forward_logits(model, X, batch=batch, device=device)
    return per_class_avg_logits(logits, y, num_classes)


def fd_distill_step(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    targets: np.ndarray,
    *,
    gamma: float = DEFAULT_GAMMA,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    epochs: int = DEFAULT_DISTILL_EPOCHS,
    seed: int = 0,
    client_id: int = 0,
    round_num: int = 0,
    device: str | torch.device = "cpu",
) -> None:
    """Continue local training with the Eq. 4 per-class logit targets.

    Per Jeong et al.'s FD, each sample keeps its ground-truth CE term and adds
    a distillation regularizer pulling the model's logits toward the global
    per-class target of its own label:

        ``loss = CE(logits, y) + gamma * MSE(logits, targets[y])``

    (MSE on raw logits — the targets are average *logit* vectors, not
    probabilities). Samples whose label has an all-zero target row (absent
    class / sole contributor, see :func:`fd_aggregate`) are excluded from the
    distillation term, so with no usable targets this reduces exactly to the
    plain supervised step.
    """
    targets = np.asarray(targets, dtype=np.float32)
    has_target = np.any(targets != 0, axis=1)  # [L]
    device_t = _resolve(device)
    targets_t = torch.as_tensor(targets, device=device_t)
    usable = torch.as_tensor(has_target, device=device_t)

    def loss_fn(logits: torch.Tensor, yb: torch.Tensor, idx) -> torch.Tensor:
        loss = F.cross_entropy(logits, yb)
        if gamma:
            mask = usable[yb]
            if bool(mask.any()):
                loss = loss + gamma * F.mse_loss(logits[mask], targets_t[yb[mask]])
        return loss

    rng = make_rng(seed, client_id=client_id, round_num=round_num)
    _train(
        model, X, y, loss_fn,
        lr=lr, batch=batch, epochs=epochs, device=device_t, rng=rng,
    )


# ---------------------------------------------------------------------------
# Evaluation: top-1 accuracy per round, full metrics for the final run
# (same helper contract as the FL unit, but on models — FD has no weights)
# ---------------------------------------------------------------------------


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch: int = _EVAL_BATCH,
    device: str | torch.device = "cpu",
) -> float:
    """Top-1 accuracy of ``model`` on (X, y). No gradients are created."""
    logits = _forward_logits(model, X, batch=batch, device=device)
    return float(np.mean(logits.argmax(axis=1) == np.asarray(y)))


def evaluate_model_full(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int = NUM_CLASSES,
    batch: int = _EVAL_BATCH,
    device: str | torch.device = "cpu",
) -> dict:
    """Final-run metrics (accuracy, macro-F1/precision, confusion matrix)."""
    logits = _forward_logits(model, X, batch=batch, device=device)
    return classification_metrics(y, logits.argmax(axis=1), num_classes=num_classes)


# ---------------------------------------------------------------------------
# Round driver: framework-free FD for smoke tests (ADR-4)
# ---------------------------------------------------------------------------


def run_fd(
    model_fn,
    client_data,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    rounds: int,
    num_classes: int = NUM_CLASSES,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    distill_epochs: int = DEFAULT_DISTILL_EPOCHS,
    gamma: float = DEFAULT_GAMMA,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """Run FD for ``rounds`` rounds without any FL framework.

    ``client_data`` is a sequence of per-client ``(X, y)`` tuples; each client
    keeps a persistent local model (FD exchanges logit statistics, never
    weights). Round r: every client trains locally and uploads its Eqs. 2-3
    matrix, the server computes Eq. 4 exclude-self targets, and each client
    runs the distillation step on them.

    FD has no global model, so evaluation follows the paper's convention: the
    per-round reported accuracy is the **best-performing client's** test
    accuracy. Returns::

        {
          "round_accuracy":  [best-client top-1 accuracy per round],
          "client_accuracy": [[per-client top-1 accuracy] per round],
          "best_client":     client index with the highest final-round accuracy,
          "final":           evaluate_model_full(...) of that client's model,
        }

    Deterministic on CPU: model inits and all shuffle streams are keyed on
    ``(seed, client_id, round_num)`` via ssfl.config.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if not len(client_data):
        raise ValueError("run_fd requires at least one client")

    models = []
    for k in range(len(client_data)):
        torch.manual_seed(derive_seed(seed, client_id=k, round_num=0))
        models.append(model_fn())

    round_accuracy: list[float] = []
    client_accuracy: list[list[float]] = []
    for r in range(1, rounds + 1):
        uploads = np.stack(
            [
                fd_client_step(
                    models[k], X, y,
                    num_classes=num_classes, lr=lr, batch=batch,
                    local_epochs=local_epochs, seed=seed,
                    client_id=k, round_num=r, device=device,
                )
                for k, (X, y) in enumerate(client_data)
            ]
        )
        targets = fd_aggregate(uploads)
        for k, (X, y) in enumerate(client_data):
            fd_distill_step(
                models[k], X, y, targets[k],
                gamma=gamma, lr=lr, batch=batch, epochs=distill_epochs,
                seed=seed + 1, client_id=k, round_num=r, device=device,
            )
        accs = [evaluate_model(m, X_test, y_test, device=device) for m in models]
        client_accuracy.append(accs)
        round_accuracy.append(max(accs))

    best_client = int(np.argmax(client_accuracy[-1]))
    return {
        "round_accuracy": round_accuracy,
        "client_accuracy": client_accuracy,
        "best_client": best_client,
        "final": evaluate_model_full(
            models[best_client], X_test, y_test,
            num_classes=num_classes, device=device,
        ),
    }
