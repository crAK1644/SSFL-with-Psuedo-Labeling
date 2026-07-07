"""SSFL — pure client/server logic for the paper's proposed method
(Zhao et al. 2023, Eqs. 11-18, Algorithm 1) plus every ablation variant.

Round shape (Algorithm 1 unrolled into one client call, solution.md example):

1. distill the classifier on open-set samples whose previous-round global
   label is not ``UNLABELED`` (cross-entropy on hard labels; skipped round 1),
2. train the classifier on private data (Eq. 11: Adam, CE),
3. max-softmax confidence on the open set (Eq. 12),
4. train the 2-class discriminator: low-confidence open samples are
   "unfamiliar" (class 1), all private samples "familiar" (class 0)
   (Eqs. 13-14),
5. predict open-set labels and overwrite discriminator-rejected samples with
   ``UNLABELED`` (Eqs. 15-16); upload the int64 hard-label vector [N_o].

Server side: majority vote (Eq. 17, ties -> lowest class index, zero votes ->
``UNLABELED``), then 1 epoch of CE training of the server model on the voted
open-set labels; evaluation always reports the *server* model.

Ablation flags (each touches only its own mechanism):
- ``no_discriminating``  — skip steps 4-5; upload predictions for all samples.
- ``simply_filtering``   — unfamiliar = confidence below threshold; no
  discriminator model is trained.
- ``no_voting``          — server aggregates by the mean of the clients'
  one-hot/soft predictions per sample (argmax as global label).
- fixed ``threshold``    — 0.7/0.8/0.9 instead of the per-client median.
- ``label_mode=softX``   — upload float32 [N_o, L] softmax vectors rounded to
  X decimals (unfamiliar rows zeroed); the server averages and argmaxes.

Judgment calls pinned by ADR-8: distillation and discriminator training run
1 epoch per round; optimizers are re-created each round; distillation loss is
cross-entropy on hard labels.

No Flower/Ray imports (ADR-4): plain ndarrays in/out, torch modules passed
explicitly so a transport layer can persist their state between rounds.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ssfl.config import derive_seed
from ssfl.methods._shared import classification_metrics
from ssfl.methods.payloads import LABEL_MODES, UNLABELED
from ssfl.models import NUM_CLASSES, resolve_device

__all__ = [
    "vote",
    "aggregate",
    "softmax",
    "predict_logits",
    "client_round",
    "server_step",
    "run_round",
    "evaluate",
    "final_metrics",
    "classification_metrics",
]

#: Paper hyperparameter defaults (CON-2): Adam lr 1e-4, batch 80, 5 epochs.
DEFAULT_LR = 1e-4
DEFAULT_BATCH = 80
DEFAULT_LOCAL_EPOCHS = 5
#: ADR-8: distillation, discriminator and server training run 1 epoch/round;
#: optimizers are re-created each call.
DISTILL_EPOCHS = 1
DISC_EPOCHS = 1
SERVER_EPOCHS = 1

_EVAL_BATCH = 256


def vote(client_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Eq. 17 majority vote over client hard labels.

    ``client_labels``: int64 [K, N_o], entries in {-1, 0..L-1}; -1 = unfamiliar
    (never counted). Returns int64 [N_o] global labels; ties break to the
    lowest class index (np.argmax convention, ADR-8); zero-vote samples -> -1.
    """
    client_labels = np.asarray(client_labels, dtype=np.int64)
    if client_labels.ndim != 2:
        raise ValueError(
            f"client_labels must be 2-D [K, N_o], got shape {client_labels.shape}"
        )
    n_open = client_labels.shape[1]
    counts = np.zeros((n_open, num_classes), dtype=np.int64)
    for k_row in client_labels:
        valid = k_row >= 0
        np.add.at(counts, (np.nonzero(valid)[0], k_row[valid]), 1)
    winners = counts.argmax(axis=1)
    winners[counts.sum(axis=1) == 0] = UNLABELED
    return winners


def _hard_views(payloads: np.ndarray, label_mode: str) -> np.ndarray:
    """Per-client hard label views, int64 [K, N_o] with -1 for unfamiliar.

    Hard payloads pass through; soft payloads map all-zero rows (unfamiliar)
    to -1 and other rows to their argmax.
    """
    if label_mode == "hard":
        return payloads.astype(np.int64)
    views = payloads.argmax(axis=2).astype(np.int64)
    views[(payloads != 0).sum(axis=2) == 0] = UNLABELED
    return views


def aggregate(
    payloads: "list[np.ndarray] | np.ndarray",
    num_classes: int,
    *,
    no_voting: bool = False,
    label_mode: str = "hard",
) -> tuple[np.ndarray, dict]:
    """Aggregate client uploads into global hard labels + diagnostics.

    - default (hard labels): majority vote (Eq. 17);
    - ``no_voting``: mean of the clients' one-hot vectors, argmax — for hard
      labels this is provably identical to the count-based vote (uniform
      positive scaling never changes an argmax, and both break ties to the
      lowest class index), so it's computed via :func:`vote` directly rather
      than materializing a ``[K, N_o, num_classes]`` one-hot tensor;
    - ``label_mode=softX``: mean of the soft vectors, argmax.
    In every mode, samples that received no vote (all -1 / all-zero rows)
    come back as ``UNLABELED``.

    Diagnostics: ``zero_vote`` (samples with no vote), ``vote_agreement``
    (fraction of valid client votes matching the global label; 0.0 when there
    are no valid votes).
    """
    if len(payloads) == 0:
        raise ValueError("aggregate needs at least one client payload")
    if no_voting and label_mode != "hard":
        raise ValueError(
            "no_voting has no distinct effect when label_mode != 'hard' "
            f"(soft-label aggregation is already a mean); got label_mode={label_mode!r}"
        )
    stacked = np.stack([np.asarray(p) for p in payloads])
    hard = _hard_views(stacked, label_mode)

    if label_mode != "hard":
        mean = stacked.astype(np.float64).mean(axis=0)
        global_labels = mean.argmax(axis=1).astype(np.int64)
        global_labels[(stacked != 0).sum(axis=(0, 2)) == 0] = UNLABELED
    else:
        # no_voting's one-hot-mean-then-argmax and the plain vote count
        # produce identical results for hard labels (see docstring), so both
        # paths share this call rather than materializing a one-hot tensor.
        global_labels = vote(hard, num_classes)

    valid = hard >= 0
    n_valid = int(valid.sum())
    agreeing = int((valid & (hard == global_labels[np.newaxis, :])).sum())
    diag = {
        "zero_vote": int((global_labels == UNLABELED).sum()),
        "vote_agreement": agreeing / n_valid if n_valid else 0.0,
    }
    return global_labels, diag


# ---------------------------------------------------------------------------
# Torch plumbing (transport-friendly: ndarrays in, ndarrays out)
# ---------------------------------------------------------------------------


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
    *,
    epochs: int,
    lr: float,
    batch: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> None:
    """Adam + cross-entropy mini-batch loop; optimizer re-created per call
    (ADR-8). The shuffle stream is keyed on ``seed`` alone: callers derive a
    distinct seed per sub-step (distillation / private training /
    discriminator) from a single client-round seed so each still gets a
    reproducible, independent shuffle.
    """
    if len(X) == 0:
        return
    dev = _resolve(device)
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.int64, device=dev)
    for _ in range(epochs):
        for idx in _batches(len(Xt), batch, rng):
            bidx = torch.from_numpy(idx).to(dev)
            loss = F.cross_entropy(model(Xt[bidx]), yt[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    opt.zero_grad(set_to_none=True)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch: int = _EVAL_BATCH,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Batched eval-mode forward pass; float32 [N, C] logits on CPU.

    ``C`` is whatever the model's head produces (``num_classes`` for the
    classifier/server model, 2 for the discriminator).
    """
    dev = _resolve(device)
    model.to(dev).eval()
    if len(X) == 0:
        return np.zeros(
            (0, getattr(model, "num_classes", NUM_CLASSES)), dtype=np.float32
        )
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
    out = []
    for start in range(0, len(Xt), batch):
        out.append(model(Xt[start : start + batch]).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable plain softmax (Eq. 12 confidence, T = 1)."""
    logits = np.asarray(logits)
    z = logits.astype(np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def _resolve_threshold(threshold: str | float, conf: np.ndarray) -> float:
    """Eq. 13 threshold: per-client median confidence, or a fixed value."""
    if threshold == "median":
        return float(np.median(conf))
    return float(threshold)


# ---------------------------------------------------------------------------
# Client round: Algorithm 1 unrolled (Eqs. 11-16) + ablation flags
# ---------------------------------------------------------------------------


def client_round(
    classifier: nn.Module,
    discriminator: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    open_X: np.ndarray,
    global_labels: np.ndarray | None,
    *,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    seed: int = 0,
    device: str | torch.device = "cpu",
    threshold: str | float = "median",
    no_discriminating: bool = False,
    simply_filtering: bool = False,
    label_mode: str = "hard",
) -> tuple[np.ndarray, dict]:
    """One SSFL client round (module docstring, Algorithm 1 unrolled).

    Mutates ``classifier`` (and, unless ``no_discriminating``/
    ``simply_filtering``, ``discriminator``) in place; returns the
    client->server upload plus a diagnostics dict with ``unfamiliar``
    (count of filtered-out open samples) and ``threshold`` (the confidence
    cutoff actually used).

    Steps, in order: (1) distill on previous-round global labels (skipped
    when ``global_labels`` is ``None`` or entirely ``UNLABELED``);
    (2) train on private data (Eq. 11); (3) max-softmax confidence on the
    open set (Eq. 12); (4) train the discriminator (Eqs. 13-14, skipped by
    ``no_discriminating``/``simply_filtering``); (5) predict + filter
    (Eqs. 15-16). ``no_discriminating`` uploads every prediction;
    ``simply_filtering`` filters by the confidence threshold directly, with
    no discriminator model. ``label_mode`` only changes the upload's
    representation (hard label vs. rounded softmax vector); it never
    changes which samples are filtered.
    """
    if label_mode not in LABEL_MODES:
        raise ValueError(
            f"unknown label_mode {label_mode!r}; allowed: {', '.join(LABEL_MODES)}"
        )
    if len(X) == 0:
        raise ValueError("client_round requires a non-empty private dataset")
    n_open = len(open_X)
    if global_labels is not None:
        global_labels = np.asarray(global_labels)
        if global_labels.shape != (n_open,):
            raise ValueError(
                f"global_labels must have shape ({n_open},), got {global_labels.shape}"
            )
        mask = global_labels != UNLABELED
        if mask.any():
            _train(
                classifier,
                open_X[mask],
                global_labels[mask].astype(np.int64),
                epochs=DISTILL_EPOCHS,
                lr=lr,
                batch=batch,
                seed=seed,
                device=device,
            )

    _train(
        classifier,
        X,
        y,
        epochs=local_epochs,
        lr=lr,
        batch=batch,
        seed=seed + 1,
        device=device,
    )

    probs = softmax(predict_logits(classifier, open_X, batch=batch, device=device))
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1).astype(np.int64)
    threshold_value = _resolve_threshold(threshold, conf)

    if no_discriminating:
        unfamiliar = np.zeros(n_open, dtype=bool)
    elif simply_filtering:
        unfamiliar = conf < threshold_value
    else:
        disc_unfamiliar = conf < threshold_value
        disc_X = np.concatenate([X, open_X[disc_unfamiliar]], axis=0)
        disc_y = np.concatenate(
            [
                np.zeros(len(X), dtype=np.int64),
                np.ones(int(disc_unfamiliar.sum()), dtype=np.int64),
            ]
        )
        _train(
            discriminator,
            disc_X,
            disc_y,
            epochs=DISC_EPOCHS,
            lr=lr,
            batch=batch,
            seed=seed + 2,
            device=device,
        )
        verdict = predict_logits(discriminator, open_X, batch=batch, device=device)
        unfamiliar = verdict.argmax(axis=1) == 1

    if label_mode == "hard":
        payload = np.where(unfamiliar, UNLABELED, preds).astype(np.int64)
    else:
        decimals = int(label_mode[len("soft") :])
        payload = np.round(probs, decimals).astype(np.float32)
        payload[unfamiliar] = 0.0

    diag = {"unfamiliar": int(unfamiliar.sum()), "threshold": threshold_value}
    return payload, diag


# ---------------------------------------------------------------------------
# Server step: 1 epoch of CE training on the voted open-set labels
# ---------------------------------------------------------------------------


def server_step(
    model: nn.Module,
    open_X: np.ndarray,
    global_labels: np.ndarray,
    *,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> int:
    """Train ``model`` in place on samples with a valid (!= UNLABELED) vote.

    Returns the number of samples trained on; 0 (and no-op) when every
    sample is ``UNLABELED``.
    """
    global_labels = np.asarray(global_labels)
    mask = global_labels != UNLABELED
    n = int(mask.sum())
    if n == 0:
        return 0
    _train(
        model,
        open_X[mask],
        global_labels[mask].astype(np.int64),
        epochs=SERVER_EPOCHS,
        lr=lr,
        batch=batch,
        seed=seed,
        device=device,
    )
    return n


# ---------------------------------------------------------------------------
# Round driver: one full SSFL round without any FL framework (ADR-4)
# ---------------------------------------------------------------------------


def run_round(
    client_states,
    server_model: nn.Module,
    clients_data,
    open_X: np.ndarray,
    global_labels: np.ndarray | None,
    *,
    round_num: int,
    run_seed: int,
    num_classes: int = NUM_CLASSES,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    threshold: str | float = "median",
    no_voting: bool = False,
    no_discriminating: bool = False,
    simply_filtering: bool = False,
    label_mode: str = "hard",
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, dict]:
    """One full SSFL round, driven without any FL framework.

    ``client_states`` is a sequence of per-client ``(classifier,
    discriminator)`` module pairs; ``clients_data`` the matching ``(X, y)``
    private datasets. Every client runs :func:`client_round`, the server
    aggregates (:func:`aggregate`) and trains on the result
    (:func:`server_step`, in place on ``server_model`` — the model
    evaluation reports each round).

    Seeding discipline: client k's round seed is
    ``ssfl.config.derive_seed(run_seed, client_id=k, round_num=round_num)``;
    the server-side training step uses ``client_id=len(client_states)``,
    which no real client id (0..K-1) can collide with. Identical inputs
    reproduce identical outputs bit-for-bit on CPU.

    Returns ``(global_labels, diag)`` where ``diag`` has ``round``,
    ``unfamiliar_per_client`` (list, one entry per client),
    ``zero_vote``, ``vote_agreement`` (from :func:`aggregate`) and
    ``server_trained_on`` (from :func:`server_step`).
    """
    if len(client_states) != len(clients_data):
        raise ValueError(
            f"got {len(client_states)} client states but "
            f"{len(clients_data)} client datasets"
        )

    payloads = []
    unfamiliar_per_client = []
    for k, ((classifier, discriminator), (X, y)) in enumerate(
        zip(client_states, clients_data)
    ):
        seed = derive_seed(run_seed, client_id=k, round_num=round_num)
        payload, client_diag = client_round(
            classifier,
            discriminator,
            X,
            y,
            open_X,
            global_labels,
            lr=lr,
            batch=batch,
            local_epochs=local_epochs,
            seed=seed,
            device=device,
            threshold=threshold,
            no_discriminating=no_discriminating,
            simply_filtering=simply_filtering,
            label_mode=label_mode,
        )
        payloads.append(payload)
        unfamiliar_per_client.append(client_diag["unfamiliar"])

    new_globals, agg_diag = aggregate(
        payloads,
        num_classes,
        no_voting=no_voting,
        label_mode=label_mode,
    )

    server_seed = derive_seed(
        run_seed, client_id=len(client_states), round_num=round_num
    )
    trained_on = server_step(
        server_model,
        open_X,
        new_globals,
        lr=lr,
        batch=batch,
        seed=server_seed,
        device=device,
    )

    diag = {
        "round": round_num,
        "unfamiliar_per_client": unfamiliar_per_client,
        "zero_vote": agg_diag["zero_vote"],
        "vote_agreement": agg_diag["vote_agreement"],
        "server_trained_on": trained_on,
    }
    return new_globals, diag


# ---------------------------------------------------------------------------
# Evaluation: per-round top-1 accuracy + final macro metrics
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch: int = _EVAL_BATCH,
    device: str | torch.device = "cpu",
) -> float:
    """Top-1 accuracy of ``model`` on (X, y). No gradients are created."""
    preds = predict_logits(model, X, batch=batch, device=device).argmax(axis=1)
    return float(np.mean(preds == np.asarray(y)))


def final_metrics(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int = NUM_CLASSES,
    batch: int = _EVAL_BATCH,
    device: str | torch.device = "cpu",
) -> dict:
    """Final-run metrics (accuracy, macro-F1/precision, confusion matrix)."""
    preds = predict_logits(model, X, batch=batch, device=device).argmax(axis=1)
    return classification_metrics(y, preds, num_classes=num_classes)
