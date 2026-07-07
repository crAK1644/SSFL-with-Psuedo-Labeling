"""Tests for DS-FL logic (ssfl.methods.dsfl_logic): Eqs. 5-10 + ERA.

Covers, per the spec:
- the elementwise averaging step (Eq. 6),
- ERA sharpening (Eqs. 7-8): entropy strictly below the plain-softmax
  average's entropy for non-degenerate inputs, argmax preserved, rows sum
  to 1, and the load-bearing T<1 direction guarded,
- client step / distillation / evaluation against the payload contract,
- a deterministic 2-round micro-run on a tiny cache subset,
- open-set label integrity (no open labels consumed anywhere).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import torch
from torch import nn

from ssfl.methods import dsfl_logic
from ssfl.methods.dsfl_logic import (
    DEFAULT_ERA_TEMPERATURE,
    aggregate,
    average_logits,
    classification_metrics,
    client_step,
    distill,
    era,
    evaluate,
    final_metrics,
    predict_logits,
    run_round,
    softmax,
)

NUM_CLASSES = 11


def _tiny_model(seed: int = 0) -> nn.Module:
    """A small deterministic classifier over the [B, 23, 5] input."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(23 * 5, NUM_CLASSES))


def _tiny_data(rng, n: int = 24):
    X = rng.normal(size=(n, 23, 5)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=n).astype(np.int64)
    return X, y


class _ConstantLogits(nn.Module):
    """Always predicts `cls` regardless of input."""

    def __init__(self, cls: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        logits = torch.zeros(num_classes)
        logits[cls] = 10.0
        self.register_buffer("logits", logits)

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy (natural log)."""
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=-1)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# softmax helper
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_rows_sum_to_one(self, rng):
        z = rng.normal(size=(10, 11)).astype(np.float32)
        p = softmax(z)
        np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-5)
        assert (p > 0).all()

    def test_matches_definition(self, rng):
        z = rng.normal(size=(4, 5))
        expected = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(softmax(z), expected, rtol=1e-6)

    def test_stable_for_large_logits(self):
        z = np.array([[1000.0, 1000.0, 999.0]])
        p = softmax(z)
        assert np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-6)

    def test_accepts_plain_list_and_returns_float32(self):
        p = softmax([[1.0, 2.0, 3.0]])
        assert p.dtype == np.float32


# ---------------------------------------------------------------------------
# Eq. 6: elementwise average of client logit matrices
# ---------------------------------------------------------------------------


class TestAverageLogits:
    def test_elementwise_mean(self, rng):
        mats = [rng.normal(size=(7, 11)).astype(np.float32) for _ in range(3)]
        avg = average_logits(mats)
        np.testing.assert_allclose(
            avg, np.mean(np.stack(mats), axis=0), rtol=1e-5
        )

    def test_returns_float32_payload_shape(self, rng):
        mats = [rng.normal(size=(5, 11)).astype(np.float32) for _ in range(4)]
        avg = average_logits(mats)
        assert avg.dtype == np.float32
        assert avg.shape == (5, 11)

    def test_single_client_is_identity(self, rng):
        m = rng.normal(size=(3, 11)).astype(np.float32)
        np.testing.assert_allclose(average_logits([m]), m, rtol=1e-6)

    def test_rejects_empty_and_mismatched(self, rng):
        with pytest.raises(ValueError):
            average_logits([])
        with pytest.raises(ValueError):
            average_logits(
                [np.zeros((3, 11), np.float32), np.zeros((4, 11), np.float32)]
            )
        with pytest.raises(ValueError):
            average_logits(np.zeros((0, 3, 11), np.float32))  # empty stacked array

    def test_accepts_a_single_stacked_ndarray(self, rng):
        """docstring: 'K arrays ... (or one [K, N_o, L] array)'."""
        mats = [rng.normal(size=(5, 11)).astype(np.float32) for _ in range(3)]
        np.testing.assert_allclose(
            average_logits(np.stack(mats)), average_logits(mats), rtol=1e-6
        )


# ---------------------------------------------------------------------------
# Eqs. 7-8: ERA — temperature softmax with T < 1 SHARPENS
# ---------------------------------------------------------------------------


class TestEra:
    def test_rows_sum_to_one(self, rng):
        avg = rng.normal(size=(20, 11)).astype(np.float32)
        out = era(avg, temperature=0.1)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-5)
        assert (out >= 0).all()

    def test_entropy_strictly_below_plain_softmax(self, rng):
        """The load-bearing property: ERA output entropy < plain-softmax entropy."""
        avg = rng.normal(size=(50, 11)).astype(np.float32)
        sharpened = era(avg, temperature=0.1)
        plain = softmax(avg)
        assert (_entropy(sharpened) < _entropy(plain)).all()

    def test_argmax_preserved(self, rng):
        avg = rng.normal(size=(50, 11)).astype(np.float32)
        out = era(avg, temperature=0.1)
        np.testing.assert_array_equal(
            out.argmax(axis=-1), avg.argmax(axis=-1)
        )

    def test_temperature_configurable_and_monotone(self, rng):
        avg = rng.normal(size=(30, 11)).astype(np.float32)
        sharper = era(avg, temperature=0.05)
        softer = era(avg, temperature=0.5)
        assert (_entropy(sharper) < _entropy(softer)).all()

    def test_default_temperature_sharpens(self):
        assert 0.0 < DEFAULT_ERA_TEMPERATURE < 1.0

    @pytest.mark.parametrize("bad_t", [1.0, 2.0, 20.0, 0.0, -0.5])
    def test_rejects_non_sharpening_temperature(self, bad_t, rng):
        """T >= 1 inverts the mechanism (softens); T <= 0 is undefined."""
        avg = rng.normal(size=(3, 11)).astype(np.float32)
        with pytest.raises(ValueError):
            era(avg, temperature=bad_t)

    def test_output_is_float32(self, rng):
        avg = rng.normal(size=(3, 11)).astype(np.float32)
        assert era(avg, temperature=0.1).dtype == np.float32


class TestAggregate:
    def test_is_average_then_era(self, rng):
        mats = [rng.normal(size=(9, 11)).astype(np.float32) for _ in range(5)]
        out = aggregate(mats, temperature=0.2)
        np.testing.assert_allclose(
            out, era(average_logits(mats), temperature=0.2), rtol=1e-5
        )

    def test_matches_dsfl_download_contract(self, rng):
        from ssfl.methods.payloads import payload_spec

        spec = payload_spec("dsfl", "server_to_client")
        mats = [rng.normal(size=(6, 11)).astype(np.float32) for _ in range(2)]
        out = aggregate(mats)
        assert out.dtype == np.dtype(spec.dtype)
        assert out.shape == (6, 11)  # (n_open, num_classes)


# ---------------------------------------------------------------------------
# evaluation helper: accuracy / macro-F1 / macro-precision / confusion matrix
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_hand_computed_example(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        m = classification_metrics(y_true, y_pred, num_classes=3)
        expected_cm = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        np.testing.assert_array_equal(m["confusion_matrix"], expected_cm)
        assert m["accuracy"] == pytest.approx(4 / 6)
        # per-class precision: 0 -> 1/2, 1 -> 2/3, 2 -> 1/1
        assert m["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
        # per-class recall: 1/2, 2/2, 1/2 ; F1: 1/2, 4/5, 2/3
        assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)

    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 3])
        m = classification_metrics(y, y, num_classes=4)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["macro_precision"] == 1.0
        np.testing.assert_array_equal(m["confusion_matrix"], np.eye(4, dtype=np.int64))

    def test_absent_class_yields_zero_not_nan(self):
        """Classes never seen nor predicted must not produce NaNs (macro over L)."""
        y_true = np.array([0, 0])
        y_pred = np.array([0, 0])
        m = classification_metrics(y_true, y_pred, num_classes=3)
        assert np.isfinite(m["macro_f1"])
        assert np.isfinite(m["macro_precision"])
        assert m["confusion_matrix"].shape == (3, 3)


class TestPredictLogits:
    def test_empty_input_returns_empty_array_not_crash(self):
        model = _tiny_model()
        out = predict_logits(model, np.zeros((0, 23, 5), dtype=np.float32))
        assert out.shape == (0, NUM_CLASSES)
        assert out.dtype == np.float32

    def test_supports_auto_device(self):
        """device='auto' (RunConfig's own default) must not crash."""
        model = _tiny_model()
        out = predict_logits(model, np.zeros((3, 23, 5), dtype=np.float32), device="auto")
        assert out.shape == (3, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Eq. 5: client step — private supervised training + open-set logits
# ---------------------------------------------------------------------------


class TestClientStep:
    def test_returns_upload_contract_payload(self, rng):
        from ssfl.methods.payloads import payload_spec

        spec = payload_spec("dsfl", "client_to_server")
        X, y = _tiny_data(rng)
        open_X = rng.normal(size=(13, 23, 5)).astype(np.float32)
        logits = client_step(
            _tiny_model(), X, y, open_X,
            lr=1e-3, batch=8, local_epochs=1, seed=0, device="cpu",
        )
        assert logits.dtype == np.dtype(spec.dtype)  # float32
        assert logits.shape == (13, NUM_CLASSES)  # (n_open, num_classes)

    def test_rejects_empty_private_dataset(self, rng):
        open_X = rng.normal(size=(5, 23, 5)).astype(np.float32)
        with pytest.raises(ValueError, match="non-empty"):
            client_step(
                _tiny_model(),
                np.zeros((0, 23, 5), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                open_X,
                seed=0, device="cpu",
            )

    def test_defaults_are_paper_hyperparameters(self):
        """CON-2: Adam lr 1e-4, batch 80 (client step and distillation)."""
        for fn in (client_step, distill, run_round):
            params = inspect.signature(fn).parameters
            assert params["lr"].default == pytest.approx(1e-4), fn
            assert params["batch"].default == 80, fn
        assert inspect.signature(client_step).parameters["local_epochs"].default == 5

    def test_trains_the_model(self, rng):
        """Supervised training must actually update parameters."""
        X, y = _tiny_data(rng)
        open_X = rng.normal(size=(5, 23, 5)).astype(np.float32)
        model = _tiny_model()
        before = [p.detach().clone() for p in model.parameters()]
        client_step(model, X, y, open_X,
                    lr=1e-2, batch=8, local_epochs=2, seed=0, device="cpu")
        after = list(model.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_reduces_private_loss(self, rng):
        X, y = _tiny_data(rng, n=64)
        open_X = rng.normal(size=(5, 23, 5)).astype(np.float32)
        model = _tiny_model()
        ce = nn.CrossEntropyLoss()

        def loss():
            with torch.no_grad():
                return ce(model(torch.from_numpy(X)), torch.from_numpy(y)).item()

        before = loss()
        client_step(model, X, y, open_X,
                    lr=1e-2, batch=8, local_epochs=20, seed=0, device="cpu")
        assert loss() < before

    def test_deterministic_given_seed(self, rng):
        X, y = _tiny_data(rng)
        open_X = rng.normal(size=(7, 23, 5)).astype(np.float32)
        out = [
            client_step(_tiny_model(), X, y, open_X,
                        lr=1e-3, batch=8, local_epochs=2, seed=5, device="cpu")
            for _ in range(2)
        ]
        np.testing.assert_array_equal(out[0], out[1])

    def test_logits_match_model_predictions_after_training(self, rng):
        """Eq. 5: the upload is the trained model's logits on the open set."""
        X, y = _tiny_data(rng)
        open_X = rng.normal(size=(9, 23, 5)).astype(np.float32)
        model = _tiny_model()
        logits = client_step(model, X, y, open_X,
                             lr=1e-3, batch=8, local_epochs=1, seed=0, device="cpu")
        np.testing.assert_allclose(
            logits, predict_logits(model, open_X, batch=4, device="cpu"),
            rtol=1e-5, atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Eqs. 9-10: distillation on global soft labels
# ---------------------------------------------------------------------------


class TestDistill:
    def test_moves_predictions_toward_targets(self, rng):
        open_X = rng.normal(size=(40, 23, 5)).astype(np.float32)
        target_cls = rng.integers(0, NUM_CLASSES, size=40)
        targets = np.full((40, NUM_CLASSES), 1e-4, dtype=np.float32)
        targets[np.arange(40), target_cls] = 1.0
        targets /= targets.sum(axis=1, keepdims=True)

        model = _tiny_model()

        def soft_ce():
            with torch.no_grad():
                logp = torch.log_softmax(model(torch.from_numpy(open_X)), dim=1)
            return -(torch.from_numpy(targets) * logp).sum(dim=1).mean().item()

        before = soft_ce()
        distill(model, open_X, targets,
                lr=1e-2, batch=8, epochs=10, seed=0, device="cpu")
        assert soft_ce() < before

    def test_deterministic_given_seed(self, rng):
        open_X = rng.normal(size=(16, 23, 5)).astype(np.float32)
        targets = era(rng.normal(size=(16, NUM_CLASSES)).astype(np.float32), 0.1)
        states = []
        for _ in range(2):
            model = _tiny_model()
            distill(model, open_X, targets,
                    lr=1e-3, batch=8, epochs=2, seed=3, device="cpu")
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        for k in states[0]:
            assert torch.equal(states[0][k], states[1][k])

    def test_takes_no_labels_for_open_samples(self):
        """Distillation consumes soft labels from aggregation only — the
        signature has no ground-truth label parameter."""
        params = set(inspect.signature(distill).parameters)
        assert "y" not in params and "y_open" not in params and "labels" not in params


# ---------------------------------------------------------------------------
# evaluation: top-1 accuracy + final metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Round driver: one full DS-FL round without any FL framework (ADR-4)
# ---------------------------------------------------------------------------


class TestRunRound:
    def _setup(self, rng, n_clients=2, n_open=10):
        clients_data = [_tiny_data(rng, n=16) for _ in range(n_clients)]
        open_X = rng.normal(size=(n_open, 23, 5)).astype(np.float32)
        client_models = [_tiny_model(seed=k) for k in range(n_clients)]
        server_model = _tiny_model(seed=99)
        return client_models, server_model, clients_data, open_X

    def test_returns_contract_soft_labels(self, rng):
        from ssfl.methods.payloads import payload_spec

        client_models, server_model, clients_data, open_X = self._setup(rng)
        soft = run_round(
            client_models, server_model, clients_data, open_X,
            round_num=1, run_seed=42,
            lr=1e-3, batch=8, local_epochs=1, device="cpu",
        )
        spec = payload_spec("dsfl", "server_to_client")
        assert soft.dtype == np.dtype(spec.dtype)
        assert soft.shape == (len(open_X), NUM_CLASSES)
        np.testing.assert_allclose(soft.sum(axis=-1), 1.0, rtol=1e-4)

    def test_trains_every_model_in_place(self, rng):
        client_models, server_model, clients_data, open_X = self._setup(rng)
        before = [
            [p.detach().clone() for p in m.parameters()]
            for m in (*client_models, server_model)
        ]
        run_round(
            client_models, server_model, clients_data, open_X,
            round_num=0, run_seed=0, lr=1e-2, batch=8, local_epochs=1, device="cpu",
        )
        for snap, model in zip(before, (*client_models, server_model)):
            assert any(
                not torch.equal(b, p.detach())
                for b, p in zip(snap, model.parameters())
            )

    def test_composes_client_step_aggregate_distill_with_derived_seeds(self, rng):
        """The driver must be exactly Eq. 5 -> Eqs. 6-8 -> Eqs. 9-10 with
        seeds from ssfl.config.derive_seed (fb1 seeding discipline)."""
        from ssfl.config import derive_seed

        client_models, server_model, clients_data, open_X = self._setup(rng)
        manual_clients = [_tiny_model(seed=k) for k in range(len(client_models))]
        manual_server = _tiny_model(seed=99)
        kw = dict(lr=1e-3, batch=8, device="cpu")
        round_num, run_seed = 2, 7

        soft = run_round(
            client_models, server_model, clients_data, open_X,
            round_num=round_num, run_seed=run_seed, local_epochs=1, **kw,
        )

        logits = [
            client_step(
                m, X, y, open_X, local_epochs=1,
                seed=derive_seed(run_seed, client_id=k, round_num=round_num), **kw,
            )
            for k, (m, (X, y)) in enumerate(zip(manual_clients, clients_data))
        ]
        expected_soft = aggregate(logits, temperature=DEFAULT_ERA_TEMPERATURE)
        np.testing.assert_array_equal(soft, expected_soft)

        for k, m in enumerate(manual_clients):
            distill(m, open_X, expected_soft,
                    seed=derive_seed(run_seed, client_id=k, round_num=round_num) + 1, **kw)
        distill(manual_server, open_X, expected_soft,
                seed=derive_seed(run_seed, client_id=len(manual_clients),
                                 round_num=round_num) + 1, **kw)

        for got, want in zip((*client_models, server_model),
                             (*manual_clients, manual_server)):
            for pg, pw in zip(got.parameters(), want.parameters()):
                assert torch.equal(pg.detach(), pw.detach())

    def test_deterministic_across_invocations(self, rng):
        states = []
        for _ in range(2):
            client_models, server_model, clients_data, open_X = self._setup(
                np.random.default_rng(11)
            )
            soft = run_round(
                client_models, server_model, clients_data, open_X,
                round_num=1, run_seed=5, lr=1e-3, batch=8, local_epochs=1,
                device="cpu",
            )
            states.append((soft, {k: v.clone() for k, v in server_model.state_dict().items()}))
        np.testing.assert_array_equal(states[0][0], states[1][0])
        for k in states[0][1]:
            assert torch.equal(states[0][1][k], states[1][1][k])

    def test_era_temperature_is_configurable(self, rng):
        outs = []
        for temp in (0.05, 0.5):
            client_models, server_model, clients_data, open_X = self._setup(
                np.random.default_rng(3)
            )
            outs.append(run_round(
                client_models, server_model, clients_data, open_X,
                round_num=0, run_seed=1, era_temperature=temp,
                lr=1e-3, batch=8, local_epochs=1, device="cpu",
            ))
        assert (_entropy(outs[0]) < _entropy(outs[1])).all()

    def test_rejects_mismatched_models_and_data(self, rng):
        client_models, server_model, clients_data, open_X = self._setup(rng)
        with pytest.raises(ValueError):
            run_round(
                client_models[:1], server_model, clients_data, open_X,
                round_num=0, run_seed=0,
            )


# ---------------------------------------------------------------------------
# 2-round micro-run on a tiny real-cache subset (server model is what reports)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cache_subsets():
    """Tiny slices of the real cache: 2 clients, open set, test set."""
    from ssfl.data import load_client, load_open, load_test

    clients = []
    for cid in (0, 1):
        X, y = load_client(1, cid)
        clients.append((
            np.asarray(X[:48], dtype=np.float32),
            np.asarray(y[:48], dtype=np.int64),
        ))
    open_X = np.asarray(load_open()[:40], dtype=np.float32)
    Xt, yt = load_test()
    return clients, open_X, (
        np.asarray(Xt[:128], dtype=np.float32),
        np.asarray(yt[:128], dtype=np.int64),
    )


class TestTwoRoundMicroRun:
    def _run(self, cache_subsets, run_seed=42):
        from ssfl.models import build_model

        clients_data, open_X, (X_test, y_test) = cache_subsets
        torch.manual_seed(run_seed)
        client_models = [build_model("mlp") for _ in clients_data]
        server_model = build_model("mlp")

        accs = []
        for round_num in (1, 2):
            run_round(
                client_models, server_model, clients_data, open_X,
                round_num=round_num, run_seed=run_seed,
                lr=1e-4, batch=16, local_epochs=1, device="cpu",
            )
            accs.append(evaluate(server_model, X_test, y_test, device="cpu"))
        return accs, server_model, (X_test, y_test)

    def test_two_rounds_report_server_accuracy_and_final_metrics(self, cache_subsets):
        accs, server_model, (X_test, y_test) = self._run(cache_subsets)
        assert len(accs) == 2
        assert all(0.0 <= a <= 1.0 for a in accs)
        m = final_metrics(server_model, X_test, y_test,
                          num_classes=NUM_CLASSES, device="cpu")
        assert set(m) >= {"accuracy", "macro_f1", "macro_precision", "confusion_matrix"}
        assert m["accuracy"] == pytest.approx(accs[-1])
        assert m["confusion_matrix"].sum() == len(y_test)

    def test_micro_run_is_reproducible(self, cache_subsets):
        accs1, model1, _ = self._run(cache_subsets, run_seed=7)
        accs2, model2, _ = self._run(cache_subsets, run_seed=7)
        assert accs1 == accs2
        for k, v in model1.state_dict().items():
            assert torch.equal(v, model2.state_dict()[k])


# ---------------------------------------------------------------------------
# Open-set label integrity: no ground-truth open labels consumed anywhere
# ---------------------------------------------------------------------------


class TestOpenSetLabelIntegrity:
    def test_open_loader_is_x_only(self):
        """ssfl.data.load_open returns a single array — no label view exists
        for the open split, and DS-FL must keep it that way."""
        from ssfl.data import load_open

        out = load_open()
        assert isinstance(out, np.ndarray)  # X only, not an (X, y) tuple

    def test_no_open_label_parameters_in_the_api(self):
        """Every function touching the open set takes X only."""
        for fn in (client_step, distill, run_round):
            params = set(inspect.signature(fn).parameters)
            assert not params & {"open_y", "y_open", "open_labels", "labels"}, fn

    def test_logic_module_never_touches_data_loaders(self):
        """Pure logic (ADR-4): data is passed in; the module never imports
        ssfl.data, so open ground truth is unreachable by construction."""
        import ast

        tree = ast.parse(inspect.getsource(dsfl_logic))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert not any(m == "ssfl.data" or m.startswith("ssfl.data.") for m in imported)

    def test_distill_targets_come_only_from_aggregation(self, rng):
        """Distillation on the open set is driven purely by the uploaded
        logits: permuting hypothetical open ground truth changes nothing
        because none exists in the flow."""
        open_X = rng.normal(size=(12, 23, 5)).astype(np.float32)
        mats = [rng.normal(size=(12, NUM_CLASSES)).astype(np.float32) for _ in range(2)]
        soft = aggregate(mats)
        m1, m2 = _tiny_model(1), _tiny_model(1)
        distill(m1, open_X, soft, lr=1e-3, batch=4, epochs=1, seed=0, device="cpu")
        distill(m2, open_X, soft, lr=1e-3, batch=4, epochs=1, seed=0, device="cpu")
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1.detach(), p2.detach())


class TestEvaluate:
    def test_constant_predictor_accuracy(self, rng):
        X, y = _tiny_data(rng, n=50)
        acc = evaluate(_ConstantLogits(3), X, y, batch=16, device="cpu")
        assert acc == pytest.approx(float((y == 3).mean()))

    def test_final_metrics_reports_full_set(self, rng):
        X, y = _tiny_data(rng, n=30)
        m = final_metrics(_ConstantLogits(0), X, y,
                          num_classes=NUM_CLASSES, batch=16, device="cpu")
        assert m["accuracy"] == pytest.approx(float((y == 0).mean()))
        assert set(m) >= {"accuracy", "macro_f1", "macro_precision", "confusion_matrix"}
        cm = m["confusion_matrix"]
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
        assert cm.sum() == 30
        assert cm[:, 1:].sum() == 0  # everything predicted as class 0
