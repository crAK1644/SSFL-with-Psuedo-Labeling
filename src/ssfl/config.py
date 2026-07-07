"""Run configuration: validated RunConfig, deterministic run-id, seeding discipline.

Conventions per .start/specs/001-ssfl-paper-reproduction/solution.md
(RunConfig entity, run-id derivation, reproducibility pattern).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

METHODS = ("fl", "fd", "dsfl", "ssfl")
MODELS = ("cnn", "mlp", "lstm")
SCENARIOS = (1, 2, 3)
THRESHOLDS = ("median", 0.7, 0.8, 0.9)
LABEL_MODES = ("hard", "soft2", "soft4", "soft6", "soft8")

_DEFAULT_THRESHOLD = "median"
_DEFAULT_LABEL_MODE = "hard"


def _fmt(options) -> str:
    return ", ".join(repr(o) for o in options)


@dataclass(frozen=True)
class RunConfig:
    """Immutable, validated configuration for one experiment run."""

    method: str
    model: str
    scenario: int
    seed: int
    rounds: int = 200
    lr: float = 1e-4
    batch: int = 80
    local_epochs: int = 5
    threshold: str | float = _DEFAULT_THRESHOLD
    no_voting: bool = False
    no_discriminating: bool = False
    simply_filtering: bool = False
    label_mode: str = _DEFAULT_LABEL_MODE
    device: str = "auto"
    num_parallel_clients: int = 8

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject invalid values/combinations before anything launches."""
        if self.method not in METHODS:
            raise ValueError(
                f"unknown method {self.method!r}; allowed: {_fmt(METHODS)}"
            )
        if self.model not in MODELS:
            raise ValueError(
                f"unknown model {self.model!r}; allowed: {_fmt(MODELS)}"
            )
        if self.scenario not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {self.scenario!r}; allowed: {_fmt(SCENARIOS)}"
            )
        if self.threshold not in THRESHOLDS:
            raise ValueError(
                f"unknown threshold {self.threshold!r}; allowed: {_fmt(THRESHOLDS)}"
            )
        if self.label_mode not in LABEL_MODES:
            raise ValueError(
                f"unknown label_mode {self.label_mode!r}; allowed: {_fmt(LABEL_MODES)}"
            )

        if self.method != "ssfl":
            ablation_flags = [
                name
                for name in ("no_voting", "no_discriminating", "simply_filtering")
                if getattr(self, name)
            ]
            if ablation_flags:
                raise ValueError(
                    f"ablation flag(s) {', '.join(ablation_flags)} are only valid "
                    f"with method='ssfl' (got method={self.method!r})"
                )
            if self.threshold != _DEFAULT_THRESHOLD:
                raise ValueError(
                    f"threshold={self.threshold!r} is only valid with method='ssfl' "
                    f"(got method={self.method!r}); non-ssfl runs must use "
                    f"threshold={_DEFAULT_THRESHOLD!r}"
                )
            if self.label_mode != _DEFAULT_LABEL_MODE:
                raise ValueError(
                    f"label_mode={self.label_mode!r} is only valid with method='ssfl' "
                    f"(got method={self.method!r}); non-ssfl runs must use "
                    f"label_mode={_DEFAULT_LABEL_MODE!r}"
                )

        for name in ("rounds", "batch", "local_epochs", "num_parallel_clients"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not self.lr > 0:
            raise ValueError(f"lr must be positive, got {self.lr!r}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")

    def run_id(self) -> str:
        """Deterministic id: {method}-{model}-s{scenario}-seed{seed}[-flags].

        Non-default SSFL knobs are appended in a fixed order so that the same
        config always maps to the same id and distinct configs never collide.
        """
        parts = [f"{self.method}-{self.model}-s{self.scenario}-seed{self.seed}"]
        for name in ("no_voting", "no_discriminating", "simply_filtering"):
            if getattr(self, name):
                parts.append(name)
        if self.threshold != _DEFAULT_THRESHOLD:
            parts.append(f"thr{self.threshold}")
        if self.label_mode != _DEFAULT_LABEL_MODE:
            parts.append(self.label_mode)
        return "-".join(parts)


def derive_seed(run_seed: int, *, client_id: int = 0, round_num: int = 0) -> int:
    """Derive a 32-bit seed from (run seed, client_id, round).

    Single source of stochastic state (solution.md, Reproducibility pattern):
    every random site keys off this so identical configs reproduce identical
    results, with no collisions between the three axes.
    """
    key = f"ssfl|seed={run_seed}|client={client_id}|round={round_num}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big")


def make_rng(run_seed: int, *, client_id: int = 0, round_num: int = 0) -> np.random.Generator:
    """A numpy Generator whose stream is fully determined by (seed, client, round)."""
    return np.random.default_rng(derive_seed(run_seed, client_id=client_id, round_num=round_num))
