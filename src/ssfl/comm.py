"""Analytic communication-cost accounting (PRD F6; Table IV, Fig. 6).

Every byte count is computed analytically from the payload contract in
:mod:`ssfl.methods.payloads` — never by sniffing live traffic. Per round:

* FL       — float32 parameter vector, per client, both directions
* FD       — float32 ``[L, L]`` per-class logits, per client, both directions
* DS-FL    — float32 ``[N_o, L]`` open-set logits, per client, both directions
* SSFL     — int64 ``[N_o]`` hard labels, per client, both directions
* SSFL soft ablation (``label_mode=softX``) — uploads are soft labels rounded
  to *d* decimals; downloads stay hard int64 labels.

Soft-label encoding assumption (kept in lockstep with
:class:`ssfl.methods.payloads.ArraySpec`): each soft-label value lies in
``[0, 1]`` and, rounded to *d* decimals, takes one of ``10**d + 1`` levels,
i.e. ``ceil(log2(10**d + 1))`` bits per value, bit-packed with the total
rounded up to whole bytes. That is why Fig. 6's byte counts scale with the
decimal precision instead of costing full float32 per value.

The one-time open-set distribution cost (Table IV's C@D^o) — shipping the
shared unlabeled open set to every client — applies only to DS-FL and SSFL
and is reported separately from the per-round curve, as in the paper.

MB throughout means 10**6 bytes.

Consumes only framework-base APIs (ssfl.config, ssfl.methods.payloads,
ssfl.metrics) plus results files; :func:`model_param_count` lazily imports
:mod:`ssfl.models` only when a parameter count must be derived from a model
name (e.g. summarising an FL run).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ssfl.config import RunConfig
from ssfl.methods.payloads import payload_nbytes
from ssfl.metrics import CONFIG_FILE, read_rounds

#: Clients per partition scenario (solution.md acceptance criteria: scenario
#: 1/2/3 -> 27/89/89 client index sets; see ssfl.data.partition).
SCENARIO_NUM_CLIENTS: dict[int, int] = {1: 27, 2: 89, 3: 89}

#: Paper-default open-set size: 10% of 89 subsets x 1000 samples.
DEFAULT_N_OPEN = 8_900

#: Global class count L (11 canonical N-BaIoT classes).
DEFAULT_NUM_CLASSES = 11

#: One open-set sample = the 115-feature N-BaIoT vector (23 x 5 reshape),
#: shipped as float32 (ssfl.data.build: N_FEATURES).
OPEN_SET_SAMPLE_BYTES = 115 * 4

#: Methods whose clients need the shared open set distributed once (C@D^o).
METHODS_WITH_OPEN_SET = frozenset({"dsfl", "ssfl"})

_MB = 1e6


def scenario_num_clients(scenario: int) -> int:
    """Client count for a partition scenario (1 -> 27, 2 -> 89, 3 -> 89)."""
    try:
        return SCENARIO_NUM_CLIENTS[scenario]
    except KeyError:
        raise ValueError(
            f"unknown scenario {scenario!r}; allowed: "
            f"{', '.join(map(str, SCENARIO_NUM_CLIENTS))}"
        ) from None


def model_param_count(model: str) -> int:
    """Parameter count of a classifier by model name (e.g. cnn -> 248,395).

    Lazily imports :mod:`ssfl.models` (and therefore torch) so that the rest
    of this module stays numpy-only; used when an FL run's config names a
    model but no explicit ``param_count`` is given.
    """
    from ssfl.models import build_model  # deferred: torch only when needed

    return sum(p.numel() for p in build_model(model).parameters())


@dataclass(frozen=True)
class RoundCost:
    """Analytic communication cost of one round of one method."""

    method: str
    label_mode: str
    n_clients: int
    upload_bytes_per_client: int
    download_bytes_per_client: int

    @property
    def upload_bytes(self) -> int:
        """Client -> server bytes per round across all clients."""
        return self.upload_bytes_per_client * self.n_clients

    @property
    def download_bytes(self) -> int:
        """Server -> client bytes per round across all clients."""
        return self.download_bytes_per_client * self.n_clients

    @property
    def total_bytes(self) -> int:
        return self.upload_bytes + self.download_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / _MB


def round_cost(
    method: str,
    *,
    n_clients: int,
    n_open: int = DEFAULT_N_OPEN,
    num_classes: int = DEFAULT_NUM_CLASSES,
    param_count: int | None = None,
    label_mode: str = "hard",
) -> RoundCost:
    """Per-round cost of `method`, from the payload contract.

    ``param_count`` is required for FL (payloads.py raises if the dimension
    is missing); ``label_mode`` only changes SSFL uploads (soft ablation).
    """
    if not isinstance(n_clients, int) or n_clients <= 0:
        raise ValueError(f"n_clients must be a positive integer, got {n_clients!r}")
    dims: dict[str, int | None] = {
        "n_open": n_open,
        "num_classes": num_classes,
        "param_count": param_count,
    }
    kwargs = {name: value for name, value in dims.items() if value is not None}
    return RoundCost(
        method=method,
        label_mode=label_mode,
        n_clients=n_clients,
        upload_bytes_per_client=payload_nbytes(
            method, "client_to_server", label_mode=label_mode, **kwargs
        ),
        download_bytes_per_client=payload_nbytes(
            method, "server_to_client", label_mode=label_mode, **kwargs
        ),
    )


def round_cost_for_config(
    config: RunConfig,
    *,
    n_clients: int | None = None,
    n_open: int = DEFAULT_N_OPEN,
    num_classes: int = DEFAULT_NUM_CLASSES,
    param_count: int | None = None,
) -> RoundCost:
    """Per-round cost implied by a run's config (method, scenario, label mode).

    ``n_clients`` defaults to the scenario's client count and ``param_count``
    (FL only) to the config's model's parameter count.
    """
    if n_clients is None:
        n_clients = scenario_num_clients(config.scenario)
    if param_count is None and config.method == "fl":
        param_count = model_param_count(config.model)
    return round_cost(
        config.method,
        n_clients=n_clients,
        n_open=n_open,
        num_classes=num_classes,
        param_count=param_count,
        label_mode=config.label_mode,
    )


def open_set_cost_bytes(
    method: str,
    *,
    n_clients: int,
    n_open: int = DEFAULT_N_OPEN,
    sample_bytes: int = OPEN_SET_SAMPLE_BYTES,
) -> int:
    """One-time cost of distributing the open set D^o to every client (C@D^o).

    Applies only to the open-set methods (DS-FL, SSFL); 0 for FL and FD.
    Reported separately from the per-round curve, as in Table IV.
    """
    if method not in ("fl", "fd", *METHODS_WITH_OPEN_SET):
        raise ValueError(f"unknown method {method!r}")
    if method not in METHODS_WITH_OPEN_SET:
        return 0
    return n_open * sample_bytes * n_clients


def cumulative_mb_curve(cost: RoundCost, rounds: int) -> np.ndarray:
    """Cumulative MB after each of ``rounds`` rounds (shape ``[rounds]``).

    Per-round cost is constant, so the curve is linear; the one-time open-set
    cost is *not* included (see :func:`open_set_cost_bytes`).
    """
    return np.arange(1, rounds + 1, dtype=np.float64) * cost.total_mb


@dataclass(frozen=True)
class CostAt:
    """C@x result: cost at the first round an accuracy target is reached.

    ``reached=False`` means the run never hit the target; the remaining
    fields are then ``None`` — unreached targets are never fabricated.
    """

    target_acc: float | None
    reached: bool
    round: int | None = None
    test_acc: float | None = None
    cumulative_mb: float | None = None


def cost_at_accuracy(
    records: list[Mapping[str, Any]],
    *,
    per_round_mb: float,
    target_acc: float,
) -> CostAt:
    """C@x: cumulative MB at the first round with ``test_acc >= target_acc``.

    ``records`` are per-round records as read by ``ssfl.metrics``
    (``read_rounds``), in append order. Cumulative cost counts *completed
    records up to and including the hit* (robust to any round-numbering
    convention); the reported ``round`` is the record's own round field.
    """
    for i, record in enumerate(records):
        try:
            acc = record["test_acc"]
        except KeyError:
            raise ValueError(f"record {i} is missing 'test_acc': {record!r}") from None
        if acc >= target_acc:
            try:
                round_num = record["round"]
            except KeyError:
                raise ValueError(f"record {i} is missing 'round': {record!r}") from None
            return CostAt(
                target_acc=target_acc,
                reached=True,
                round=round_num,
                test_acc=acc,
                cumulative_mb=(i + 1) * per_round_mb,
            )
    return CostAt(target_acc=target_acc, reached=False)


def cost_at_top_acc(
    records: list[Mapping[str, Any]], *, per_round_mb: float
) -> CostAt:
    """C@Top-Acc: cost at the first round reaching the run's own best accuracy."""
    if not records:
        return CostAt(target_acc=None, reached=False)
    try:
        top = max(record["test_acc"] for record in records)
    except KeyError:
        raise ValueError("a record is missing 'test_acc'") from None
    return cost_at_accuracy(records, per_round_mb=per_round_mb, target_acc=top)


def cost_at_targets(
    records: list[Mapping[str, Any]],
    *,
    per_round_mb: float,
    targets: tuple[float, ...] = (0.50, 0.75),
) -> dict[str, CostAt]:
    """The Table IV C@x bundle: ``{"C@50", "C@75", "C@Top-Acc"}`` by default."""
    out = {
        f"C@{round(target * 100)}": cost_at_accuracy(
            records, per_round_mb=per_round_mb, target_acc=target
        )
        for target in targets
    }
    out["C@Top-Acc"] = cost_at_top_acc(records, per_round_mb=per_round_mb)
    return out


def run_comm_summary(
    results_root: str | Path,
    run_id: str,
    *,
    n_open: int = DEFAULT_N_OPEN,
    num_classes: int = DEFAULT_NUM_CLASSES,
    param_count: int | None = None,
) -> dict[str, Any]:
    """Communication summary for a completed run in ``results/<run-id>/``.

    Reads the run's ``config.json`` and per-round records and returns::

        method, label_mode, n_clients, rounds_completed,
        per_round_mb, open_set_mb          # C@D^o, separate (Table IV)
        cumulative_mb                      # np.ndarray, one entry per round
        C@50, C@75, C@Top-Acc              # CostAt results

    Consumed by the report unit (Table IV, Fig. 6) and by the runner for
    per-run summaries.
    """
    results_root = Path(results_root)
    run_dir = results_root / run_id
    if not run_dir.resolve().is_relative_to(results_root.resolve()):
        raise ValueError(f"run_id {run_id!r} escapes results_root {results_root!s}")
    config_path = run_dir / CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(
            f"no {CONFIG_FILE} in {run_dir} — is this a completed run directory?"
        )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        config = RunConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"malformed {CONFIG_FILE} in {run_dir}: {exc}") from exc

    n_clients = scenario_num_clients(config.scenario)
    cost = round_cost_for_config(
        config, n_clients=n_clients, n_open=n_open,
        num_classes=num_classes, param_count=param_count,
    )
    records = read_rounds(run_dir)

    summary: dict[str, Any] = {
        "method": config.method,
        "label_mode": config.label_mode,
        "n_clients": n_clients,
        "rounds_completed": len(records),
        "per_round_mb": cost.total_mb,
        "open_set_mb": open_set_cost_bytes(config.method, n_clients=n_clients, n_open=n_open)
        / _MB,
        "cumulative_mb": cumulative_mb_curve(cost, rounds=len(records)),
    }
    summary.update(cost_at_targets(records, per_round_mb=cost.total_mb))
    return summary
