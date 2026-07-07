"""Durable run-results store for results/<run-id>/ (ADR-7).

Layout (solution.md, Data Storage Changes):
    config.json   — full resolved RunConfig, written at run start
    rounds.jsonl  — one line per round, appended + flushed immediately
    final.json    — final metrics, written atomically (temp file + rename)
    cm.npy        — confusion matrix on the test set

A crash loses at most the in-flight round: every completed round's line is
already on disk, and final.json is either absent or complete — never partial.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_FILE = "config.json"
ROUNDS_FILE = "rounds.jsonl"
FINAL_FILE = "final.json"
CM_FILE = "cm.npy"


def read_rounds(run_dir: str | Path) -> list[dict[str, Any]]:
    """All completed round records, in append order.

    A truncated trailing line (interrupted append) is skipped, so records
    written before an interrupt remain readable.
    """
    path = Path(run_dir) / ROUNDS_FILE
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # partial line from an interrupted append
    return records


class MetricsStore:
    """Durable metrics writer/reader for one run directory."""

    def __init__(self, results_root: str | Path, run_id: str) -> None:
        self.run_dir = Path(results_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: dict[str, Any]) -> None:
        """Persist the resolved run config at run start."""
        (self.run_dir / CONFIG_FILE).write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def append_round(
        self,
        *,
        round: int,
        test_acc: float,
        wall_s: float,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Append one round record and force it to disk before returning."""
        record: dict[str, Any] = {"round": round, "test_acc": test_acc, "wall_s": wall_s}
        if diagnostics is not None:
            record["diagnostics"] = diagnostics
        line = json.dumps(record, sort_keys=False)
        with open(self.run_dir / ROUNDS_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_rounds(self) -> list[dict[str, Any]]:
        return read_rounds(self.run_dir)

    def write_final(self, final: dict[str, Any]) -> None:
        """Write final.json atomically: temp file in the same dir + rename."""
        target = self.run_dir / FINAL_FILE
        tmp = target.with_name(FINAL_FILE + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(final, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    def save_confusion_matrix(self, cm: np.ndarray) -> None:
        np.save(self.run_dir / CM_FILE, cm)
