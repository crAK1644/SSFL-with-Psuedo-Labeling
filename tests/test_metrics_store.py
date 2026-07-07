"""Tests for ssfl.metrics: durable per-round metrics under results/<run-id>/."""

import json
import os

import numpy as np
import pytest

from ssfl.metrics import MetricsStore, read_rounds


@pytest.fixture
def store(tmp_path):
    return MetricsStore(tmp_path, "ssfl-cnn-s1-seed0")


class TestLayoutAndConfig:
    def test_creates_run_directory(self, tmp_path, store):
        assert (tmp_path / "ssfl-cnn-s1-seed0").is_dir()

    def test_write_config_at_start(self, tmp_path, store):
        cfg = {"method": "ssfl", "model": "cnn", "scenario": 1, "seed": 0}
        store.write_config(cfg)
        on_disk = json.loads((tmp_path / "ssfl-cnn-s1-seed0" / "config.json").read_text())
        assert on_disk == cfg


class TestRoundAppends:
    def test_one_jsonl_line_per_round(self, tmp_path, store):
        store.append_round(round=1, test_acc=0.42, wall_s=1.5)
        store.append_round(round=2, test_acc=0.55, wall_s=1.4, diagnostics={"failed_clients": 1})
        lines = (tmp_path / "ssfl-cnn-s1-seed0" / "rounds.jsonl").read_text().splitlines()
        assert len(lines) == 2
        first, second = (json.loads(line) for line in lines)
        assert first == {"round": 1, "test_acc": 0.42, "wall_s": 1.5}
        assert second["round"] == 2
        assert second["diagnostics"] == {"failed_clients": 1}

    def test_append_is_immediately_durable(self, tmp_path, store):
        # Each line must be readable by an independent handle right away,
        # without close/flush on the store side.
        path = tmp_path / "ssfl-cnn-s1-seed0" / "rounds.jsonl"
        for r in range(1, 4):
            store.append_round(round=r, test_acc=0.1 * r, wall_s=1.0)
            lines = path.read_text().splitlines()
            assert len(lines) == r
            assert json.loads(lines[-1])["round"] == r

    def test_interrupt_leaves_prior_lines_readable(self, tmp_path, store):
        # Simulate a crash mid-append: a truncated partial trailing line.
        store.append_round(round=1, test_acc=0.4, wall_s=1.0)
        store.append_round(round=2, test_acc=0.5, wall_s=1.0)
        path = tmp_path / "ssfl-cnn-s1-seed0" / "rounds.jsonl"
        with open(path, "a") as f:
            f.write('{"round": 3, "test_ac')  # interrupted mid-write, no newline
        records = read_rounds(tmp_path / "ssfl-cnn-s1-seed0")
        assert [r["round"] for r in records] == [1, 2]


class TestReader:
    def test_reader_returns_records_in_order(self, tmp_path, store):
        for r in range(1, 6):
            store.append_round(round=r, test_acc=r / 10, wall_s=2.0)
        records = read_rounds(tmp_path / "ssfl-cnn-s1-seed0")
        assert [r["round"] for r in records] == [1, 2, 3, 4, 5]
        assert records[2]["test_acc"] == pytest.approx(0.3)

    def test_store_read_rounds_matches_module_reader(self, tmp_path, store):
        store.append_round(round=1, test_acc=0.9, wall_s=3.0)
        assert store.read_rounds() == read_rounds(tmp_path / "ssfl-cnn-s1-seed0")

    def test_reader_on_missing_file_returns_empty(self, tmp_path, store):
        assert read_rounds(tmp_path / "ssfl-cnn-s1-seed0") == []


class TestFinalAtomicity:
    def test_write_final_and_no_temp_leftover(self, tmp_path, store):
        final = {"accuracy": 0.91, "macro_f1": 0.88}
        store.write_final(final)
        run_dir = tmp_path / "ssfl-cnn-s1-seed0"
        assert json.loads((run_dir / "final.json").read_text()) == final
        leftovers = [p for p in run_dir.iterdir() if p.name not in ("final.json",)]
        assert leftovers == []

    def test_crash_before_rename_leaves_no_partial_final(self, tmp_path, store, monkeypatch):
        def boom(src, dst):
            raise OSError("simulated crash before rename")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            store.write_final({"accuracy": 0.5})
        assert not (tmp_path / "ssfl-cnn-s1-seed0" / "final.json").exists()


class TestConfusionMatrix:
    def test_cm_saved_as_npy(self, tmp_path, store):
        cm = np.arange(121, dtype=np.int64).reshape(11, 11)
        store.save_confusion_matrix(cm)
        loaded = np.load(tmp_path / "ssfl-cnn-s1-seed0" / "cm.npy")
        assert np.array_equal(loaded, cm)
