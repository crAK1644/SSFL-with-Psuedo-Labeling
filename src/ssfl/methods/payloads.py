"""Payload contract: single source of truth for every method's exchanged arrays.

Names, dtypes and (symbolic) shapes follow solution.md "Internal API Changes"
(ADR-2). Method strategies, client logic, Message assembly and comm-cost
accounting must all consume these definitions — never re-declare shapes.

Symbolic dims:
    param_count  — number of model parameters (FL weights, flattened)
    num_classes  — L, global class count (11)
    n_open       — N_o, size of the shared open set
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Sentinel in SSFL hard-label arrays: -1 = unfamiliar (client) / unlabeled
#: this round (server). Never counted in the vote.
UNLABELED = -1

METHODS = ("fl", "fd", "dsfl", "ssfl")
DIRECTIONS = ("server_to_client", "client_to_server")
LABEL_MODES = ("hard", "soft2", "soft4", "soft6", "soft8")


@dataclass(frozen=True)
class ArraySpec:
    """One exchanged array: name, dtype, symbolic shape, optional precision.

    ``decimals`` is set only for SSFL soft-label ablation uploads: values are
    rounded to that many decimals and the cost model charges only the bits
    needed to represent them.
    """

    name: str
    dtype: np.dtype
    shape: tuple[str, ...]
    decimals: int | None = None

    def nbytes(self, dims: dict[str, int]) -> int:
        """Byte size given concrete dims (keyed by the symbolic dim names)."""
        try:
            n_values = math.prod(dims[d] for d in self.shape)
        except KeyError as missing:
            raise ValueError(
                f"payload {self.name!r} needs dimension {missing.args[0]!r} "
                f"(shape {self.shape})"
            ) from None
        if self.decimals is None:
            return n_values * np.dtype(self.dtype).itemsize
        # Rounded to `decimals` decimals: each value takes one of 10^d + 1
        # levels, i.e. ceil(log2(10^d + 1)) bits.
        bits_per_value = math.ceil(math.log2(10**self.decimals + 1))
        return math.ceil(n_values * bits_per_value / 8)


_HARD_LABELS = ArraySpec("hard_labels", np.int64, ("n_open",))

#: The contract: method -> direction -> ArraySpec (hard/default modes).
PAYLOAD_CONTRACT: dict[str, dict[str, ArraySpec]] = {
    "fl": {
        # Table I CNN state; modeled as the flattened float32 parameter vector.
        # (client->server additionally carries num_examples in a MetricRecord,
        # negligible and not part of the array contract.)
        "server_to_client": ArraySpec("weights", np.float32, ("param_count",)),
        "client_to_server": ArraySpec("weights", np.float32, ("param_count",)),
    },
    "fd": {
        # Per-class average logits; zero row when a class is absent (Eq. 3).
        "server_to_client": ArraySpec(
            "global_class_logits", np.float32, ("num_classes", "num_classes")
        ),
        "client_to_server": ArraySpec(
            "local_class_logits", np.float32, ("num_classes", "num_classes")
        ),
    },
    "dsfl": {
        # Open-set logits; global side is post-ERA soft labels (Eq. 8).
        "server_to_client": ArraySpec(
            "global_soft_labels", np.float32, ("n_open", "num_classes")
        ),
        "client_to_server": ArraySpec(
            "local_logits", np.float32, ("n_open", "num_classes")
        ),
    },
    "ssfl": {
        # Hard labels with the -1 sentinel (see UNLABELED).
        "server_to_client": _HARD_LABELS,
        "client_to_server": _HARD_LABELS,
    },
}

#: SSFL soft-label ablation uploads (label_mode=softX): rounded soft labels.
_SSFL_SOFT_UPLOADS: dict[str, ArraySpec] = {
    f"soft{d}": ArraySpec(
        "soft_labels", np.float32, ("n_open", "num_classes"), decimals=d
    )
    for d in (2, 4, 6, 8)
}


def payload_spec(method: str, direction: str, *, label_mode: str = "hard") -> ArraySpec:
    """The ArraySpec exchanged by `method` in `direction`.

    `label_mode` only affects SSFL client->server uploads (the soft-label
    ablation); downloads stay hard int64 labels.
    """
    if method not in PAYLOAD_CONTRACT:
        raise ValueError(f"unknown method {method!r}; allowed: {', '.join(METHODS)}")
    if direction not in DIRECTIONS:
        raise ValueError(
            f"unknown direction {direction!r}; allowed: {', '.join(DIRECTIONS)}"
        )
    if label_mode not in LABEL_MODES:
        raise ValueError(
            f"unknown label_mode {label_mode!r}; allowed: {', '.join(LABEL_MODES)}"
        )
    if method == "ssfl" and direction == "client_to_server" and label_mode != "hard":
        return _SSFL_SOFT_UPLOADS[label_mode]
    return PAYLOAD_CONTRACT[method][direction]


def payload_nbytes(
    method: str,
    direction: str,
    *,
    n_open: int | None = None,
    num_classes: int | None = None,
    param_count: int | None = None,
    label_mode: str = "hard",
) -> int:
    """Analytic byte size of one payload in one direction (comm accounting)."""
    spec = payload_spec(method, direction, label_mode=label_mode)
    dims = {
        name: value
        for name, value in (
            ("n_open", n_open),
            ("num_classes", num_classes),
            ("param_count", param_count),
        )
        if value is not None
    }
    return spec.nbytes(dims)
