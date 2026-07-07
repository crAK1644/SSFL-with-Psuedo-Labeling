"""Tests for ssfl.comm — analytic communication-cost accounting.

Every byte count asserted here is hand-computed from the payload contract
(src/ssfl/methods/payloads.py); the C@x cases use synthetic accuracy curves.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ssfl import comm
from ssfl.config import RunConfig
from ssfl.methods.payloads import payload_nbytes
from ssfl.metrics import MetricsStore


# ---------------------------------------------------------------------------
# Per-round byte counts (hand-computed per method)
# ---------------------------------------------------------------------------


class TestRoundCost:
    def test_fl_hand_computed(self):
        # 1000 float32 params, 3 clients: 4000 B per client per direction.
        cost = comm.round_cost("fl", n_clients=3, param_count=1000)
        assert cost.download_bytes_per_client == 4000
        assert cost.upload_bytes_per_client == 4000
        assert cost.download_bytes == 12_000
        assert cost.upload_bytes == 12_000
        assert cost.total_bytes == 24_000
        assert cost.total_mb == pytest.approx(0.024)

    def test_fd_hand_computed(self):
        # [L, L] float32 per client each way: 11*11*4 = 484 B.
        cost = comm.round_cost("fd", n_clients=5, num_classes=11)
        assert cost.upload_bytes_per_client == 484
        assert cost.download_bytes_per_client == 484
        assert cost.total_bytes == 484 * 5 * 2

    def test_dsfl_hand_computed(self):
        # [N_o, L] float32 per client each way: 10*11*4 = 440 B.
        cost = comm.round_cost("dsfl", n_clients=4, n_open=10, num_classes=11)
        assert cost.upload_bytes_per_client == 440
        assert cost.download_bytes_per_client == 440
        assert cost.total_bytes == 440 * 4 * 2

    def test_ssfl_hard_hand_computed(self):
        # int64 [N_o] per client each way: 10*8 = 80 B.
        cost = comm.round_cost("ssfl", n_clients=7, n_open=10)
        assert cost.upload_bytes_per_client == 80
        assert cost.download_bytes_per_client == 80
        assert cost.total_bytes == 80 * 7 * 2

    def test_ssfl_soft2_hand_computed(self):
        # Upload: soft labels [N_o, L] rounded to 2 decimals -> 10^2+1 levels
        # -> ceil(log2(101)) = 7 bits/value; 10*11 values = 770 bits = 97 B.
        # Download stays hard int64 [N_o] = 80 B.
        cost = comm.round_cost(
            "ssfl", n_clients=2, n_open=10, num_classes=11, label_mode="soft2"
        )
        assert cost.upload_bytes_per_client == math.ceil(10 * 11 * 7 / 8) == 97
        assert cost.download_bytes_per_client == 80
        assert cost.total_bytes == 2 * (97 + 80)

    @pytest.mark.parametrize("label_mode", ["soft2", "soft4", "soft6", "soft8"])
    def test_soft_modes_match_payload_contract(self, label_mode):
        # comm must never re-derive byte math: it must equal payloads.py.
        cost = comm.round_cost(
            "ssfl", n_clients=1, n_open=123, num_classes=11, label_mode=label_mode
        )
        assert cost.upload_bytes_per_client == payload_nbytes(
            "ssfl", "client_to_server", n_open=123, num_classes=11,
            label_mode=label_mode,
        )
        assert cost.download_bytes_per_client == payload_nbytes(
            "ssfl", "server_to_client", n_open=123,
        )

    def test_soft_precision_scales_bytes(self):
        # Fig. 6: fewer decimals -> fewer bytes; d=8 costs more than d=2.
        sizes = [
            comm.round_cost(
                "ssfl", n_clients=1, n_open=100, num_classes=11, label_mode=m
            ).upload_bytes_per_client
            for m in ("soft2", "soft4", "soft6", "soft8")
        ]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            comm.round_cost("fedprox", n_clients=1)

    def test_fl_requires_param_count(self):
        with pytest.raises(ValueError):
            comm.round_cost("fl", n_clients=1)

    def test_nonpositive_clients_rejected(self):
        with pytest.raises(ValueError):
            comm.round_cost("fd", n_clients=0, num_classes=11)


class TestScenarioClients:
    def test_paper_counts(self):
        assert comm.scenario_num_clients(1) == 27
        assert comm.scenario_num_clients(2) == 89
        assert comm.scenario_num_clients(3) == 89

    def test_unknown_scenario_rejected(self):
        with pytest.raises(ValueError):
            comm.scenario_num_clients(4)


class TestRoundCostForConfig:
    def test_ssfl_soft_config(self):
        cfg = RunConfig(method="ssfl", model="cnn", scenario=2, seed=0,
                        label_mode="soft2")
        cost = comm.round_cost_for_config(cfg, n_open=10, num_classes=11)
        assert cost.n_clients == 89
        assert cost.upload_bytes_per_client == 97   # see soft2 case above
        assert cost.download_bytes_per_client == 80

    def test_fl_config_derives_param_count_from_model(self):
        cfg = RunConfig(method="fl", model="cnn", scenario=1, seed=0)
        cost = comm.round_cost_for_config(cfg)
        # CNN parameter count from Table I: 248,395 float32 values.
        assert cost.upload_bytes_per_client == 248_395 * 4
        assert cost.n_clients == 27

    def test_explicit_param_count_wins(self):
        cfg = RunConfig(method="fl", model="cnn", scenario=1, seed=0)
        cost = comm.round_cost_for_config(cfg, param_count=10)
        assert cost.upload_bytes_per_client == 40


class TestModelParamCount:
    def test_cnn_matches_table_i(self):
        assert comm.model_param_count("cnn") == 248_395


# ---------------------------------------------------------------------------
# One-time open-set distribution cost (Table IV, C@D^o)
# ---------------------------------------------------------------------------


class TestOpenSetCost:
    def test_methods_without_open_set_cost_zero(self):
        assert comm.open_set_cost_bytes("fl", n_clients=27) == 0
        assert comm.open_set_cost_bytes("fd", n_clients=27) == 0

    def test_dsfl_and_ssfl_hand_computed(self):
        # N_o samples x 115 float32 features per sample, sent to every client:
        # 10 * 115 * 4 * 3 = 13,800 B.
        for method in ("dsfl", "ssfl"):
            assert comm.open_set_cost_bytes(method, n_clients=3, n_open=10) == 13_800

    def test_default_n_open_is_paper_value(self):
        # 10% of 89 subsets x 1000 samples = 8,900.
        got = comm.open_set_cost_bytes("ssfl", n_clients=1)
        assert got == 8_900 * 115 * 4

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            comm.open_set_cost_bytes("nope", n_clients=1)


# ---------------------------------------------------------------------------
# Cumulative MB-vs-round curve
# ---------------------------------------------------------------------------


class TestCumulativeCurve:
    def test_linear_in_rounds(self):
        cost = comm.round_cost("ssfl", n_clients=1, n_open=125_000)
        # 125000 * 8 B up + same down = 2 MB/round.
        assert cost.total_mb == pytest.approx(2.0)
        curve = comm.cumulative_mb_curve(cost, rounds=4)
        np.testing.assert_allclose(curve, [2.0, 4.0, 6.0, 8.0])

    def test_length_and_monotonicity(self):
        cost = comm.round_cost("fd", n_clients=10, num_classes=11)
        curve = comm.cumulative_mb_curve(cost, rounds=200)
        assert len(curve) == 200
        assert np.all(np.diff(curve) > 0)

    def test_zero_rounds(self):
        cost = comm.round_cost("fd", n_clients=1, num_classes=11)
        assert len(comm.cumulative_mb_curve(cost, rounds=0)) == 0


# ---------------------------------------------------------------------------
# C@x extraction from accuracy curves
# ---------------------------------------------------------------------------


def _records(accs):
    return [
        {"round": i + 1, "test_acc": a, "wall_s": 1.0} for i, a in enumerate(accs)
    ]


class TestCostAtTargets:
    ACCS = [0.30, 0.55, 0.50, 0.80, 0.70]

    def test_c_at_50(self):
        hit = comm.cost_at_accuracy(_records(self.ACCS), per_round_mb=2.0,
                                    target_acc=0.50)
        assert hit.reached
        assert hit.round == 2
        assert hit.test_acc == pytest.approx(0.55)
        assert hit.cumulative_mb == pytest.approx(4.0)

    def test_c_at_75(self):
        hit = comm.cost_at_accuracy(_records(self.ACCS), per_round_mb=2.0,
                                    target_acc=0.75)
        assert hit.reached
        assert hit.round == 4
        assert hit.cumulative_mb == pytest.approx(8.0)

    def test_c_at_top_acc(self):
        hit = comm.cost_at_top_acc(_records(self.ACCS), per_round_mb=2.0)
        assert hit.reached
        assert hit.round == 4
        assert hit.test_acc == pytest.approx(0.80)
        assert hit.cumulative_mb == pytest.approx(8.0)

    def test_top_acc_uses_first_occurrence(self):
        accs = [0.2, 0.9, 0.4, 0.9]
        hit = comm.cost_at_top_acc(_records(accs), per_round_mb=1.0)
        assert hit.round == 2
        assert hit.cumulative_mb == pytest.approx(2.0)

    def test_unreached_target_is_explicit(self):
        hit = comm.cost_at_accuracy(_records(self.ACCS), per_round_mb=2.0,
                                    target_acc=0.90)
        assert not hit.reached
        assert hit.round is None
        assert hit.test_acc is None
        assert hit.cumulative_mb is None

    def test_empty_records_all_unreached(self):
        assert not comm.cost_at_accuracy([], per_round_mb=1.0, target_acc=0.5).reached
        assert not comm.cost_at_top_acc([], per_round_mb=1.0).reached

    def test_cost_at_targets_bundle(self):
        out = comm.cost_at_targets(_records(self.ACCS), per_round_mb=2.0)
        assert set(out) == {"C@50", "C@75", "C@Top-Acc"}
        assert out["C@50"].cumulative_mb == pytest.approx(4.0)
        assert out["C@75"].cumulative_mb == pytest.approx(8.0)
        assert out["C@Top-Acc"].test_acc == pytest.approx(0.80)

    def test_cumulative_uses_record_count_not_round_field(self):
        # Robust to 0-indexed round numbering: cost counts completed rounds.
        recs = [{"round": 0, "test_acc": 0.6, "wall_s": 1.0}]
        hit = comm.cost_at_accuracy(recs, per_round_mb=3.0, target_acc=0.5)
        assert hit.cumulative_mb == pytest.approx(3.0)

    def test_record_missing_test_acc_raises_clear_error(self):
        with pytest.raises(ValueError, match="test_acc"):
            comm.cost_at_accuracy([{"round": 1}], per_round_mb=1.0, target_acc=0.5)
        with pytest.raises(ValueError, match="test_acc"):
            comm.cost_at_top_acc([{"round": 1}], per_round_mb=1.0)

    def test_record_missing_round_raises_clear_error(self):
        with pytest.raises(ValueError, match="round"):
            comm.cost_at_accuracy([{"test_acc": 0.9}], per_round_mb=1.0, target_acc=0.5)


# ---------------------------------------------------------------------------
# Per-run summary from results files
# ---------------------------------------------------------------------------


class TestRunSummary:
    def _write_run(self, results_root, accs):
        cfg = RunConfig(method="ssfl", model="cnn", scenario=2, seed=0)
        store = MetricsStore(results_root, cfg.run_id())
        store.write_config(cfg.__dict__ | {})
        for i, acc in enumerate(accs):
            store.append_round(round=i + 1, test_acc=acc, wall_s=0.5)
        return cfg.run_id()

    def test_summary_fields(self, tmp_path):
        run_id = self._write_run(tmp_path, [0.30, 0.60, 0.80])
        summary = comm.run_comm_summary(tmp_path, run_id, n_open=10)

        # SSFL hard, N_o=10, 89 clients: (80+80)*89 B/round.
        per_round_mb = (80 + 80) * 89 / 1e6
        assert summary["method"] == "ssfl"
        assert summary["n_clients"] == 89
        assert summary["rounds_completed"] == 3
        assert summary["per_round_mb"] == pytest.approx(per_round_mb)
        assert summary["open_set_mb"] == pytest.approx(10 * 115 * 4 * 89 / 1e6)
        np.testing.assert_allclose(
            summary["cumulative_mb"],
            per_round_mb * np.arange(1, 4),
        )
        assert summary["C@50"].reached
        assert summary["C@50"].cumulative_mb == pytest.approx(2 * per_round_mb)
        assert summary["C@75"].cumulative_mb == pytest.approx(3 * per_round_mb)
        assert summary["C@Top-Acc"].test_acc == pytest.approx(0.80)

    def test_summary_unreached(self, tmp_path):
        run_id = self._write_run(tmp_path, [0.10, 0.20])
        summary = comm.run_comm_summary(tmp_path, run_id, n_open=10)
        assert not summary["C@50"].reached
        assert not summary["C@75"].reached
        assert summary["C@Top-Acc"].reached  # a run always reaches its own max

    def test_missing_config_rejected(self, tmp_path):
        (tmp_path / "empty-run").mkdir()
        with pytest.raises(FileNotFoundError):
            comm.run_comm_summary(tmp_path, "empty-run")

    def test_no_open_set_cost_for_fl(self, tmp_path):
        cfg = RunConfig(method="fd", model="cnn", scenario=1, seed=1)
        store = MetricsStore(tmp_path, cfg.run_id())
        store.write_config(dict(cfg.__dict__))
        store.append_round(round=1, test_acc=0.4, wall_s=0.1)
        summary = comm.run_comm_summary(tmp_path, cfg.run_id())
        assert summary["open_set_mb"] == 0.0
        assert summary["n_clients"] == 27
        assert summary["per_round_mb"] == pytest.approx(484 * 27 * 2 / 1e6)

    def test_run_id_escaping_results_root_rejected(self, tmp_path):
        run_id = self._write_run(tmp_path, [0.5])
        with pytest.raises(ValueError, match="escapes"):
            comm.run_comm_summary(tmp_path / "nested", f"../{run_id}")

    def test_malformed_config_raises_clear_error(self, tmp_path):
        run_dir = tmp_path / "bad-run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps({"method": "ssfl"}))
        with pytest.raises(ValueError, match="malformed config.json"):
            comm.run_comm_summary(tmp_path, "bad-run")
