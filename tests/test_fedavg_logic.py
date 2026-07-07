"""Tests for the pure FedAvg logic (Eq. 1): client step, aggregation, round
driver, evaluation, determinism. No Flower/Ray anywhere (ADR-4)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ssfl.methods import fl_logic
from ssfl.models import build_model

# ---------------------------------------------------------------------------
# Fixtures: tiny real-cache subsets (loaders only) and a cheap model factory
# ---------------------------------------------------------------------------

TINY = 64  # samples per client for micro-runs


def model_fn():
    return build_model("mlp", num_classes=11)


@pytest.fixture(scope="session")
def tiny_clients():
    """Two clients' private data, truncated to TINY samples each."""
    from ssfl.data import load_client

    clients = []
    for cid in (0, 1):
        X, y = load_client(1, cid)
        clients.append(
            (np.asarray(X[:TINY], dtype=np.float32), np.asarray(y[:TINY], dtype=np.int64))
        )
    return clients


@pytest.fixture(scope="session")
def tiny_test():
    from ssfl.data import load_test

    X, y = load_test()
    return np.asarray(X[:128], dtype=np.float32), np.asarray(y[:128], dtype=np.int64)


@pytest.fixture(scope="session")
def init_w():
    return fl_logic.init_weights(model_fn, seed=1234)

# ---------------------------------------------------------------------------
# Server step: sample-count-weighted average (Eq. 1)
# ---------------------------------------------------------------------------


def test_aggregate_weighted_average_hand_computed():
    # Client A: N=1, Client B: N=3  ->  w = (1*wA + 3*wB) / 4
    w_a = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    ]
    w_b = [
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ]
    result = fl_logic.aggregate([w_a, w_b], [1, 3])

    expected = [
        np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
        np.array([2.5, 3.0], dtype=np.float32),
    ]
    assert len(result) == 2
    for got, want in zip(result, expected, strict=True):
        assert got.dtype == np.float32
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-7)


def test_aggregate_single_client_is_identity():
    w = [np.arange(6, dtype=np.float32).reshape(2, 3)]
    result = fl_logic.aggregate([w], [17])
    np.testing.assert_array_equal(result[0], w[0])


def test_set_weights_rejects_wrong_length_payload(init_w):
    model = model_fn()
    with pytest.raises(ValueError):
        fl_logic.set_weights(model, init_w[:-1])


def test_aggregate_rejects_bad_input():
    w = [np.zeros(3, dtype=np.float32)]
    with pytest.raises(ValueError):
        fl_logic.aggregate([], [])
    with pytest.raises(ValueError):
        fl_logic.aggregate([w, w], [1])  # length mismatch
    with pytest.raises(ValueError):
        fl_logic.aggregate([w], [0])  # zero total samples


# ---------------------------------------------------------------------------
# Client step: local training on private labeled data
# ---------------------------------------------------------------------------


def test_client_step_returns_updated_weights_and_sample_count(tiny_clients, init_w):
    X, y = tiny_clients[0]
    before = [w.copy() for w in init_w]

    new_w, n, loss = fl_logic.client_step(
        model_fn, init_w, X, y, epochs=1, batch=16, seed=0, device="cpu"
    )

    assert n == len(X)
    assert isinstance(new_w, list) and len(new_w) == len(init_w)
    for got, orig in zip(new_w, init_w, strict=True):
        assert got.dtype == np.float32
        assert got.shape == orig.shape
    # Training must actually change the parameters...
    assert any(not np.array_equal(g, o) for g, o in zip(new_w, init_w, strict=True))
    # ...without mutating the caller's copy of the global weights.
    for orig, snap in zip(init_w, before, strict=True):
        np.testing.assert_array_equal(orig, snap)
    # Cross-entropy training loss is a finite positive scalar.
    assert np.isfinite(loss) and loss > 0.0


def test_client_step_defaults_are_paper_hyperparameters():
    params = inspect.signature(fl_logic.client_step).parameters
    assert params["epochs"].default == 5
    assert params["lr"].default == pytest.approx(1e-4)
    assert params["batch"].default == 80


def test_client_step_rejects_empty_dataset(init_w):
    X = np.zeros((0, 23, 5), dtype=np.float32)
    y = np.zeros((0,), dtype=np.int64)
    with pytest.raises(ValueError):
        fl_logic.client_step(model_fn, init_w, X, y, epochs=1, seed=0)


# ---------------------------------------------------------------------------
# Evaluation helper: top-1 accuracy + final-run metrics (macro-F1, precision,
# confusion matrix) compatible with ssfl.metrics storage
# ---------------------------------------------------------------------------


def test_classification_metrics_hand_computed():
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 0], dtype=np.int64)

    m = fl_logic.classification_metrics(y_true, y_pred, num_classes=3)

    expected_cm = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]], dtype=np.int64)
    np.testing.assert_array_equal(m["confusion_matrix"], expected_cm)
    assert m["accuracy"] == pytest.approx(4 / 6)
    # per-class precision: 1/2, 2/3, 1/1; recall: 1/2, 1, 1/2
    assert m["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    # per-class F1: 0.5, 0.8, 2/3
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m["per_class_precision"] == pytest.approx([0.5, 2 / 3, 1.0])
    assert m["per_class_f1"] == pytest.approx([0.5, 0.8, 2 / 3])


def test_classification_metrics_absent_class_scores_zero():
    # Class 2 never occurs and is never predicted -> precision/recall 0/0 -> 0.
    y_true = np.array([0, 1], dtype=np.int64)
    y_pred = np.array([0, 1], dtype=np.int64)
    m = fl_logic.classification_metrics(y_true, y_pred, num_classes=3)
    assert m["per_class_f1"][2] == 0.0
    assert m["macro_f1"] == pytest.approx(2 / 3)


def test_evaluate_top1_accuracy(tiny_test, init_w):
    X, y = tiny_test
    acc = fl_logic.evaluate(model_fn, init_w, X, y, device="cpu")
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    preds = fl_logic.predict(model_fn, init_w, X, device="cpu")
    assert preds.shape == y.shape and preds.dtype == np.int64
    assert acc == pytest.approx(float(np.mean(preds == y)))


def test_run_round_composes_client_steps_and_aggregate(tiny_clients, init_w):
    """One full round == every client trains from the global weights with its
    derive_seed(run_seed, client_id, round_num) seed, then Eq. 1 aggregation."""
    from ssfl.config import derive_seed

    new_w, info = fl_logic.run_round(
        model_fn, init_w, tiny_clients, round_num=3, run_seed=42,
        epochs=1, batch=16, device="cpu",
    )

    results = [
        fl_logic.client_step(
            model_fn, init_w, X, y,
            epochs=1, batch=16,
            seed=derive_seed(42, client_id=cid, round_num=3),
            device="cpu",
        )
        for cid, (X, y) in enumerate(tiny_clients)
    ]
    expected = fl_logic.aggregate([w for w, _, _ in results], [n for _, n, _ in results])

    assert len(new_w) == len(init_w)
    for got, want in zip(new_w, expected, strict=True):
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, want)
    # Round info: per-client sample counts and a finite mean training loss.
    assert info["num_examples"] == [n for _, n, _ in results]
    assert info["client_losses"] == pytest.approx([l for _, _, l in results])
    assert np.isfinite(info["train_loss"])


def test_run_round_rejects_empty_client_data(init_w):
    with pytest.raises(ValueError):
        fl_logic.run_round(model_fn, init_w, [], round_num=0, run_seed=0)


def test_run_round_deterministic_and_seed_sensitive(tiny_clients, init_w):
    kwargs = dict(round_num=0, epochs=1, batch=16, device="cpu")
    w1, _ = fl_logic.run_round(model_fn, init_w, tiny_clients, run_seed=7, **kwargs)
    w2, _ = fl_logic.run_round(model_fn, init_w, tiny_clients, run_seed=7, **kwargs)
    for a, b in zip(w1, w2, strict=True):
        np.testing.assert_array_equal(a, b)  # bit-identical

    w3, _ = fl_logic.run_round(model_fn, init_w, tiny_clients, run_seed=8, **kwargs)
    assert any(not np.array_equal(a, c) for a, c in zip(w1, w3, strict=True))


def test_two_round_micro_run_training_loss_decreases(tiny_clients):
    """2-round FedAvg micro-run on tiny real subsets: loss goes down."""
    w = fl_logic.init_weights(model_fn, seed=42)
    losses = []
    for round_num in range(2):
        w, info = fl_logic.run_round(
            model_fn, w, tiny_clients, round_num=round_num, run_seed=42,
            epochs=2, batch=16, device="cpu",
        )
        losses.append(info["train_loss"])
    assert losses[1] < losses[0]


def test_weight_payload_matches_fl_contract(init_w):
    """Payload = float32 arrays; total bytes match the FL contract accounting."""
    from ssfl.methods.payloads import payload_nbytes, payload_spec

    spec = payload_spec("fl", "client_to_server")
    assert all(w.dtype == np.dtype(spec.dtype) for w in init_w)
    param_count = sum(w.size for w in init_w)
    assert payload_nbytes("fl", "client_to_server", param_count=param_count) == sum(
        w.nbytes for w in init_w
    )


def test_no_fl_framework_imports():
    """ADR-4: pure logic module — no Flower/Ray anywhere in the source."""
    src = inspect.getsource(fl_logic)
    assert "flwr" not in src and "ray" not in src.replace("array", "")


def test_evaluate_full_is_metrics_store_compatible(tiny_test, init_w, tmp_path):
    from ssfl.metrics import MetricsStore

    X, y = tiny_test
    m = fl_logic.evaluate_full(model_fn, init_w, X, y, num_classes=11, device="cpu")

    cm = m.pop("confusion_matrix")
    assert cm.shape == (11, 11) and cm.dtype == np.int64
    assert cm.sum() == len(y)
    # accuracy consistent with the confusion matrix diagonal
    assert m["accuracy"] == pytest.approx(np.trace(cm) / cm.sum())

    # Remaining fields must be plain JSON types for final.json...
    store = MetricsStore(tmp_path, "fl-test-run")
    store.write_final(m)  # raises TypeError if any numpy scalars leak through
    # ...and the matrix must round-trip through cm.npy.
    store.save_confusion_matrix(cm)
    np.testing.assert_array_equal(np.load(tmp_path / "fl-test-run" / "cm.npy"), cm)
