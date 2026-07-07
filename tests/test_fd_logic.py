"""Tests for the pure-Python Federated Distillation logic (ssfl.methods.fd_logic).

Covers the paper's Eqs. 2-4 (Zhao et al. 2023 / Jeong et al. FD):
per-class average logits with the zero-vector rule, the Eq. 4 exclude-self
aggregation, the distillation step, evaluation, and a 2-round micro-run.
No Flower/Ray anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Eqs. 2-3: per-class average logit vectors (zero row for absent classes)
# ---------------------------------------------------------------------------


class TestPerClassAvgLogits:
    def test_hand_computed_averages(self):
        from ssfl.methods.fd_logic import per_class_avg_logits

        logits = np.array(
            [
                [1.0, 2.0, 3.0],   # label 0
                [3.0, 4.0, 5.0],   # label 0
                [10.0, 0.0, -2.0], # label 2
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 2], dtype=np.int64)
        out = per_class_avg_logits(logits, labels, num_classes=3)
        np.testing.assert_allclose(out[0], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(out[2], [10.0, 0.0, -2.0])

    def test_absent_class_yields_zero_vector(self):
        from ssfl.methods.fd_logic import per_class_avg_logits

        logits = np.array([[1.0, -1.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        out = per_class_avg_logits(logits, labels, num_classes=2)
        np.testing.assert_array_equal(out[0], np.zeros(2, dtype=np.float32))
        assert out[1] @ out[1] > 0

    def test_matches_payload_contract_shape_and_dtype(self):
        from ssfl.methods.fd_logic import per_class_avg_logits
        from ssfl.methods.payloads import payload_spec

        spec = payload_spec("fd", "client_to_server")
        L = 11
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(7, L)).astype(np.float32)
        labels = rng.integers(0, L, size=7).astype(np.int64)
        out = per_class_avg_logits(logits, labels, num_classes=L)
        assert out.dtype == spec.dtype
        assert out.shape == (L, L)

    def test_no_samples_at_all_gives_all_zero(self):
        from ssfl.methods.fd_logic import per_class_avg_logits

        out = per_class_avg_logits(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            num_classes=4,
        )
        np.testing.assert_array_equal(out, np.zeros((4, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Eq. 4: server aggregation, excluding each client's own contribution
# ---------------------------------------------------------------------------


class TestFdAggregate:
    def test_hand_computed_exclude_self(self):
        """K=3 clients, L=3 classes; zero rows mark absent classes."""
        from ssfl.methods.fd_logic import fd_aggregate

        uploads = np.array(
            [
                # client 0: has classes 0 and 2
                [[2, 4, 6], [0, 0, 0], [1, 1, 1]],
                # client 1: has classes 0 and 1
                [[4, 8, 10], [3, 3, 3], [0, 0, 0]],
                # client 2: has classes 1 and 2
                [[0, 0, 0], [5, 7, 9], [3, 5, 7]],
            ],
            dtype=np.float32,
        )
        targets = fd_aggregate(uploads)
        assert targets.shape == (3, 3, 3) and targets.dtype == np.float32

        # class 0: contributors {c0, c1}, N=2, sum=[6,12,16]
        np.testing.assert_allclose(targets[0, 0], [4, 8, 10])  # excludes own
        np.testing.assert_allclose(targets[1, 0], [2, 4, 6])
        np.testing.assert_allclose(targets[2, 0], [3, 6, 8])   # non-contributor: plain avg
        # class 1: contributors {c1, c2}, N=2, sum=[8,10,12]
        np.testing.assert_allclose(targets[0, 1], [4, 5, 6])
        np.testing.assert_allclose(targets[1, 1], [5, 7, 9])
        np.testing.assert_allclose(targets[2, 1], [3, 3, 3])
        # class 2: contributors {c0, c2}, N=2, sum=[4,6,8]
        np.testing.assert_allclose(targets[0, 2], [3, 5, 7])
        np.testing.assert_allclose(targets[1, 2], [2, 3, 4])
        np.testing.assert_allclose(targets[2, 2], [1, 1, 1])

    def test_single_contributor_gets_zero_target(self):
        """N^l == 1: exclude-self leaves nothing -> zero vector for the
        contributor; the plain average for everyone else."""
        from ssfl.methods.fd_logic import fd_aggregate

        uploads = np.zeros((2, 2, 2), dtype=np.float32)
        uploads[0, 1] = [7.0, -3.0]  # only client 0 has class 1
        targets = fd_aggregate(uploads)
        np.testing.assert_array_equal(targets[0, 1], [0.0, 0.0])
        np.testing.assert_allclose(targets[1, 1], [7.0, -3.0])

    def test_no_contributor_gives_zero_for_all(self):
        from ssfl.methods.fd_logic import fd_aggregate

        uploads = np.zeros((3, 2, 2), dtype=np.float32)
        uploads[:, 0] = [[1, 2], [3, 4], [5, 6]]  # class 1 has no contributor
        targets = fd_aggregate(uploads)
        np.testing.assert_array_equal(
            targets[:, 1], np.zeros((3, 2), dtype=np.float32)
        )

    def test_input_not_mutated(self):
        from ssfl.methods.fd_logic import fd_aggregate

        rng = np.random.default_rng(1)
        uploads = rng.normal(size=(4, 5, 5)).astype(np.float32)
        before = uploads.copy()
        fd_aggregate(uploads)
        np.testing.assert_array_equal(uploads, before)

    def test_rejects_non_square_or_wrong_rank_uploads(self):
        from ssfl.methods.fd_logic import fd_aggregate

        with pytest.raises(ValueError):
            fd_aggregate(np.zeros((3, 5), dtype=np.float32))  # 2-D, not [K, L, L]
        with pytest.raises(ValueError):
            fd_aggregate(np.zeros((3, 5, 6), dtype=np.float32))  # non-square L


# ---------------------------------------------------------------------------
# Client-side steps (local train + upload, distillation) and evaluation
# ---------------------------------------------------------------------------

L = 11  # global class count


def _tiny_data(n=24, classes=(0, 3), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 23, 5)).astype(np.float32)
    y = rng.choice(classes, size=n).astype(np.int64)
    return X, y


def _fresh_model(init_seed=0):
    import torch

    from ssfl.models import build_model

    torch.manual_seed(init_seed)
    return build_model("mlp", num_classes=L)


class TestFdClientStep:
    def test_returns_contract_payload_with_zero_rows_for_absent_classes(self):
        from ssfl.methods.fd_logic import fd_client_step
        from ssfl.methods.payloads import payload_spec

        X, y = _tiny_data(classes=(0, 3))
        out = fd_client_step(
            _fresh_model(), X, y,
            num_classes=L, lr=1e-4, batch=8, local_epochs=1, seed=0,
        )
        spec = payload_spec("fd", "client_to_server")
        assert out.dtype == spec.dtype and out.shape == (L, L)
        present = set(np.unique(y).tolist())
        for label in range(L):
            row_is_zero = not np.any(out[label])
            assert row_is_zero == (label not in present)

    def test_trains_the_model(self):
        import torch

        from ssfl.methods.fd_logic import fd_client_step

        model = _fresh_model()
        before = [p.detach().clone() for p in model.parameters()]
        X, y = _tiny_data()
        fd_client_step(model, X, y, num_classes=L, batch=8, local_epochs=1, seed=0)
        changed = any(
            not torch.equal(b, p.detach())
            for b, p in zip(before, model.parameters())
        )
        assert changed, "local training must update the model"

    def test_deterministic_under_config_seeding(self):
        from ssfl.methods.fd_logic import fd_client_step

        X, y = _tiny_data()
        kwargs = dict(num_classes=L, batch=8, local_epochs=2,
                      seed=42, client_id=3, round_num=1)
        out1 = fd_client_step(_fresh_model(7), X, y, **kwargs)
        out2 = fd_client_step(_fresh_model(7), X, y, **kwargs)
        np.testing.assert_array_equal(out1, out2)

    def test_single_class_client_is_fine(self):
        from ssfl.methods.fd_logic import fd_client_step

        X, y = _tiny_data(classes=(5,))
        out = fd_client_step(
            _fresh_model(), X, y, num_classes=L, batch=8, local_epochs=1, seed=0
        )
        assert np.any(out[5])
        zero_rows = [l for l in range(L) if not np.any(out[l])]
        assert zero_rows == [l for l in range(L) if l != 5]

    def test_rejects_empty_dataset(self):
        from ssfl.methods.fd_logic import fd_client_step

        X = np.zeros((0, 23, 5), dtype=np.float32)
        y = np.zeros(0, dtype=np.int64)
        with pytest.raises(ValueError, match="non-empty"):
            fd_client_step(_fresh_model(), X, y, num_classes=L, batch=8, local_epochs=1, seed=0)


class TestFdDistillStep:
    def test_pulls_outputs_toward_targets(self):
        import torch

        from ssfl.methods.fd_logic import fd_distill_step

        model = _fresh_model()
        X, y = _tiny_data(n=16, classes=(2,))
        targets = np.zeros((L, L), dtype=np.float32)
        targets[2] = np.linspace(-2, 2, L, dtype=np.float32)

        def dist_to_target():
            with torch.no_grad():
                out = model(torch.as_tensor(X)).numpy()
            return float(np.mean((out - targets[2]) ** 2))

        before = dist_to_target()
        fd_distill_step(
            model, X, y, targets,
            gamma=10.0, lr=1e-2, batch=8, epochs=5, seed=0,
        )
        assert dist_to_target() < before

    def test_zero_target_rows_are_skipped(self):
        """All-zero targets (no class has a target) + any gamma must train
        exactly like the pure-CE path — the distill term is masked out."""
        import torch

        from ssfl.methods.fd_logic import fd_distill_step

        X, y = _tiny_data(n=16)
        zero_targets = np.zeros((L, L), dtype=np.float32)
        m1, m2 = _fresh_model(3), _fresh_model(3)
        common = dict(lr=1e-3, batch=8, epochs=1, seed=9)
        fd_distill_step(m1, X, y, zero_targets, gamma=5.0, **common)
        fd_distill_step(m2, X, y, zero_targets, gamma=0.0, **common)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1.detach(), p2.detach())

    def test_uses_ground_truth_labels_too(self):
        """The distill step keeps the CE term: with gamma=0 it is plain
        supervised training and must reduce CE loss on the batch."""
        import torch
        import torch.nn.functional as F

        from ssfl.methods.fd_logic import fd_distill_step

        model = _fresh_model()
        X, y = _tiny_data(n=32)
        targets = np.zeros((L, L), dtype=np.float32)

        def ce():
            with torch.no_grad():
                return float(
                    F.cross_entropy(model(torch.as_tensor(X)), torch.as_tensor(y))
                )

        before = ce()
        fd_distill_step(model, X, y, targets, gamma=0.0,
                        lr=1e-2, batch=8, epochs=5, seed=0)
        assert ce() < before


class TestEvaluate:
    def test_accuracy_matches_manual_argmax(self):
        import torch

        from ssfl.methods.fd_logic import evaluate_model

        model = _fresh_model()
        X, y = _tiny_data(n=40, classes=tuple(range(L)))
        with torch.no_grad():
            preds = model(torch.as_tensor(X)).argmax(dim=1).numpy()
        expected = float((preds == y).mean())
        assert evaluate_model(model, X, y, batch=16) == pytest.approx(expected)

    def test_does_not_flip_training_mode_side_effects(self):
        from ssfl.methods.fd_logic import evaluate_model

        model = _fresh_model()
        X, y = _tiny_data(n=8)
        model.train()
        evaluate_model(model, X, y)
        # evaluation must not leave gradients behind
        assert all(p.grad is None for p in model.parameters())


class TestFinalMetrics:
    """Same evaluation-helper contract as the FL unit (final macro metrics)."""

    def test_hand_computed_metrics(self):
        from ssfl.methods.fd_logic import classification_metrics

        # 3 classes: true [0,0,1,2], pred [0,1,1,2]
        m = classification_metrics(
            np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]), num_classes=3
        )
        assert m["accuracy"] == pytest.approx(3 / 4)
        # precision: c0=1/1, c1=1/2, c2=1/1 ; recall: c0=1/2, c1=1, c2=1
        assert m["macro_precision"] == pytest.approx((1 + 0.5 + 1) / 3)
        f0 = 2 * 1 * 0.5 / 1.5
        f1 = 2 * 0.5 * 1 / 1.5
        assert m["macro_f1"] == pytest.approx((f0 + f1 + 1.0) / 3)
        cm = m["confusion_matrix"]
        assert cm.dtype == np.int64
        expected_cm = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(cm, expected_cm)

    def test_zero_support_classes_score_zero(self):
        from ssfl.methods.fd_logic import classification_metrics

        m = classification_metrics(
            np.array([0, 0]), np.array([0, 0]), num_classes=3
        )
        assert m["per_class_precision"][1:] == [0.0, 0.0]
        assert m["per_class_f1"][1:] == [0.0, 0.0]
        assert m["accuracy"] == 1.0

    def test_evaluate_model_full_consistent_with_accuracy(self):
        from ssfl.methods.fd_logic import evaluate_model, evaluate_model_full

        model = _fresh_model()
        X, y = _tiny_data(n=30, classes=tuple(range(L)))
        full = evaluate_model_full(model, X, y, num_classes=L)
        assert full["accuracy"] == pytest.approx(evaluate_model(model, X, y))
        assert full["confusion_matrix"].shape == (L, L)
        assert int(full["confusion_matrix"].sum()) == len(y)


# ---------------------------------------------------------------------------
# Round driver: framework-free 2-round micro-run (smoke)
# ---------------------------------------------------------------------------


def _micro_setup(seed=0):
    """2 clients with disjoint-ish classes + a small test split."""
    c0 = _tiny_data(n=16, classes=(0, 3), seed=seed)
    c1 = _tiny_data(n=16, classes=(3, 5), seed=seed + 1)
    X_test, y_test = _tiny_data(n=20, classes=(0, 3, 5), seed=seed + 2)
    return [c0, c1], X_test, y_test


def _run_micro(**overrides):
    from ssfl.methods.fd_logic import run_fd

    clients, X_test, y_test = _micro_setup()
    kwargs = dict(
        rounds=2, num_classes=L, lr=1e-3, batch=8,
        local_epochs=1, distill_epochs=1, gamma=1.0, seed=42,
    )
    kwargs.update(overrides)
    return run_fd(_model_fn, clients, X_test, y_test, **kwargs)


def _model_fn():
    from ssfl.models import build_model

    return build_model("mlp", num_classes=L)


class TestRunFd:
    def test_two_round_micro_run_completes_with_history(self):
        hist = _run_micro()
        assert len(hist["round_accuracy"]) == 2
        assert len(hist["client_accuracy"]) == 2
        assert all(len(row) == 2 for row in hist["client_accuracy"])
        assert all(0.0 <= a <= 1.0 for a in hist["round_accuracy"])

    def test_reports_best_client_per_paper_convention(self):
        """FD has no global model: the round metric is the best client's
        accuracy, and the final metrics come from that client's model."""
        hist = _run_micro()
        for best, per_client in zip(hist["round_accuracy"], hist["client_accuracy"]):
            assert best == max(per_client)
        best_client = hist["best_client"]
        assert best_client in (0, 1)
        assert hist["final"]["accuracy"] == pytest.approx(
            hist["client_accuracy"][-1][best_client]
        )
        assert hist["final"]["confusion_matrix"].shape == (L, L)

    def test_deterministic_under_same_seed(self):
        h1, h2 = _run_micro(), _run_micro()
        assert h1["round_accuracy"] == h2["round_accuracy"]
        assert h1["client_accuracy"] == h2["client_accuracy"]
        assert h1["best_client"] == h2["best_client"]
        np.testing.assert_array_equal(
            h1["final"]["confusion_matrix"], h2["final"]["confusion_matrix"]
        )

    def test_rejects_bad_arguments(self):
        from ssfl.methods.fd_logic import run_fd

        clients, X_test, y_test = _micro_setup()
        with pytest.raises(ValueError):
            run_fd(_model_fn, clients, X_test, y_test, rounds=0, num_classes=L)
        with pytest.raises(ValueError):
            run_fd(_model_fn, [], X_test, y_test, rounds=1, num_classes=L)
