"""Tests for ssfl.config: RunConfig defaults, validation, run-id, seeding."""

import pytest

from ssfl.config import RunConfig


def make(**overrides):
    base = dict(method="ssfl", model="cnn", scenario=1, seed=0)
    base.update(overrides)
    return RunConfig(**base)


class TestDefaults:
    def test_paper_defaults(self):
        cfg = make()
        assert cfg.rounds == 200
        assert cfg.lr == pytest.approx(1e-4)
        assert cfg.batch == 80
        assert cfg.local_epochs == 5
        assert cfg.threshold == "median"
        assert cfg.no_voting is False
        assert cfg.no_discriminating is False
        assert cfg.simply_filtering is False
        assert cfg.label_mode == "hard"
        assert cfg.device == "auto"

    def test_all_valid_methods_models_scenarios_accepted(self):
        for method in ("fl", "fd", "dsfl", "ssfl"):
            for model in ("cnn", "mlp", "lstm"):
                for scenario in (1, 2, 3):
                    make(method=method, model=model, scenario=scenario)

    def test_valid_thresholds_and_label_modes_accepted(self):
        for threshold in ("median", 0.7, 0.8, 0.9):
            make(threshold=threshold)
        for label_mode in ("hard", "soft2", "soft4", "soft6", "soft8"):
            make(label_mode=label_mode)


class TestValidationRejections:
    def test_unknown_method_rejected_with_allowed_options(self):
        with pytest.raises(ValueError) as exc:
            make(method="fedavg")
        msg = str(exc.value)
        for allowed in ("fl", "fd", "dsfl", "ssfl"):
            assert allowed in msg

    def test_unknown_model_rejected_with_allowed_options(self):
        with pytest.raises(ValueError) as exc:
            make(model="resnet")
        msg = str(exc.value)
        for allowed in ("cnn", "mlp", "lstm"):
            assert allowed in msg

    def test_unknown_scenario_rejected_with_allowed_options(self):
        with pytest.raises(ValueError) as exc:
            make(scenario=4)
        msg = str(exc.value)
        for allowed in ("1", "2", "3"):
            assert allowed in msg

    def test_unknown_threshold_rejected_with_allowed_options(self):
        with pytest.raises(ValueError) as exc:
            make(threshold=0.75)
        msg = str(exc.value)
        assert "median" in msg
        assert "0.7" in msg and "0.8" in msg and "0.9" in msg

    def test_unknown_label_mode_rejected_with_allowed_options(self):
        with pytest.raises(ValueError) as exc:
            make(label_mode="soft3")
        msg = str(exc.value)
        for allowed in ("hard", "soft2", "soft4", "soft6", "soft8"):
            assert allowed in msg

    @pytest.mark.parametrize("flag", ["no_voting", "no_discriminating", "simply_filtering"])
    @pytest.mark.parametrize("method", ["fl", "fd", "dsfl"])
    def test_ablation_flags_require_ssfl(self, flag, method):
        with pytest.raises(ValueError) as exc:
            make(method=method, **{flag: True})
        assert "ssfl" in str(exc.value)

    @pytest.mark.parametrize("method", ["fl", "fd", "dsfl"])
    def test_non_default_threshold_requires_ssfl(self, method):
        with pytest.raises(ValueError) as exc:
            make(method=method, threshold=0.7)
        assert "ssfl" in str(exc.value)

    @pytest.mark.parametrize("method", ["fl", "fd", "dsfl"])
    def test_non_default_label_mode_requires_ssfl(self, method):
        with pytest.raises(ValueError) as exc:
            make(method=method, label_mode="soft4")
        assert "ssfl" in str(exc.value)

    def test_default_threshold_and_label_mode_fine_for_non_ssfl(self):
        make(method="fl", threshold="median", label_mode="hard")

    def test_no_voting_with_soft_label_mode_rejected(self):
        """no_voting would be silently indistinguishable from soft-mode
        aggregation (both are already a mean) — reject at config time."""
        with pytest.raises(ValueError) as exc:
            make(no_voting=True, label_mode="soft4")
        assert "no_voting" in str(exc.value)

    def test_nonpositive_rounds_rejected(self):
        with pytest.raises(ValueError):
            make(rounds=0)

    def test_nonpositive_batch_rejected(self):
        with pytest.raises(ValueError):
            make(batch=-1)

    def test_nonpositive_lr_rejected(self):
        with pytest.raises(ValueError):
            make(lr=0.0)

    def test_nonpositive_local_epochs_rejected(self):
        with pytest.raises(ValueError):
            make(local_epochs=0)


class TestRunId:
    def test_base_format(self):
        cfg = make(method="fl", model="cnn", scenario=1, seed=0)
        assert cfg.run_id() == "fl-cnn-s1-seed0"

    def test_varies_with_each_axis(self):
        base = make(method="ssfl", model="mlp", scenario=2, seed=3)
        assert base.run_id() == "ssfl-mlp-s2-seed3"
        assert make(method="dsfl", model="mlp", scenario=2, seed=3).run_id() != base.run_id()
        assert make(method="ssfl", model="lstm", scenario=2, seed=3).run_id() != base.run_id()
        assert make(method="ssfl", model="mlp", scenario=3, seed=3).run_id() != base.run_id()
        assert make(method="ssfl", model="mlp", scenario=2, seed=4).run_id() != base.run_id()

    def test_same_config_same_id(self):
        a = make(no_discriminating=True, threshold=0.8, label_mode="soft4")
        b = make(no_discriminating=True, threshold=0.8, label_mode="soft4")
        assert a.run_id() == b.run_id()

    def test_default_flags_add_no_suffix(self):
        assert make().run_id() == "ssfl-cnn-s1-seed0"

    def test_flags_appear_in_id(self):
        assert "no_voting" in make(no_voting=True).run_id()
        assert "no_discriminating" in make(no_discriminating=True).run_id()
        assert "simply_filtering" in make(simply_filtering=True).run_id()
        assert "thr0.7" in make(threshold=0.7).run_id()
        assert "soft4" in make(label_mode="soft4").run_id()

    def test_distinct_flag_combos_distinct_ids(self):
        variants = [
            make(),
            make(no_voting=True),
            make(no_discriminating=True),
            make(simply_filtering=True),
            make(threshold=0.7),
            make(threshold=0.8),
            make(threshold=0.9),
            make(label_mode="soft2"),
            make(label_mode="soft8"),
            make(no_voting=True, threshold=0.7),
        ]
        ids = [v.run_id() for v in variants]
        assert len(set(ids)) == len(ids)


class TestSeeding:
    def test_derive_seed_deterministic(self):
        from ssfl.config import derive_seed

        assert derive_seed(0, client_id=3, round_num=7) == derive_seed(0, client_id=3, round_num=7)

    def test_derive_seed_varies_with_each_component(self):
        from ssfl.config import derive_seed

        base = derive_seed(0, client_id=0, round_num=0)
        assert derive_seed(1, client_id=0, round_num=0) != base
        assert derive_seed(0, client_id=1, round_num=0) != base
        assert derive_seed(0, client_id=0, round_num=1) != base

    def test_derive_seed_no_axis_confusion(self):
        # (client_id, round) must not be interchangeable.
        from ssfl.config import derive_seed

        assert derive_seed(0, client_id=1, round_num=2) != derive_seed(0, client_id=2, round_num=1)

    def test_derive_seed_range(self):
        from ssfl.config import derive_seed

        for s in (0, 1, 42, 2**31):
            v = derive_seed(s, client_id=88, round_num=200)
            assert isinstance(v, int)
            assert 0 <= v < 2**32

    def test_make_rng_reproducible_streams(self):
        import numpy as np

        from ssfl.config import make_rng

        a = make_rng(5, client_id=2, round_num=9).random(4)
        b = make_rng(5, client_id=2, round_num=9).random(4)
        c = make_rng(5, client_id=2, round_num=10).random(4)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)
