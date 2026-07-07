"""Tests for the pure SSFL method logic (Zhao et al. 2023, Eqs. 11-18,
Algorithm 1): client round (distill -> train -> confidence -> discriminator
-> filter), hard-label upload, server majority vote, server training,
ablation flags (each altering only its own mechanism), evaluation, and a
2-round micro-run driver. No Flower/Ray anywhere (ADR-4)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from ssfl.config import RunConfig
from ssfl.methods import ssfl_logic
from ssfl.methods.payloads import UNLABELED, payload_spec

NUM_CLASSES = 11
N_PRIVATE = 24
N_OPEN = 16

# ---------------------------------------------------------------------------
# Helpers: tiny deterministic models + synthetic data
# ---------------------------------------------------------------------------


def _classifier(seed: int = 0, num_classes: int = NUM_CLASSES) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(23 * 5, num_classes))


def _discriminator(seed: int = 1) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(23 * 5, 2))


def _data(rng, n: int = N_PRIVATE, num_classes: int = NUM_CLASSES):
    X = rng.normal(size=(n, 23, 5)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n).astype(np.int64)
    return X, y


def _open(rng, n: int = N_OPEN):
    return rng.normal(size=(n, 23, 5)).astype(np.float32)


def _state(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _states_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


class _ConstantLogits(nn.Module):
    """Always emits the given logits; has a zero-gradient trainable param so
    optimizer steps are well-defined no-ops (verdict can't drift)."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("base", torch.as_tensor(logits, dtype=torch.float32))
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.base.expand(x.shape[0], -1) + self.dummy * 0.0


_ROUND_KW = dict(lr=1e-3, batch=8, local_epochs=1, seed=5, device="cpu")


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Server-side majority vote (Eq. 17) — solution.md walkthrough is normative
# ---------------------------------------------------------------------------


def test_vote_reference_case():
    labels = np.array([[2, -1, 0, 1], [2, -1, 1, 1], [0, -1, 1, 2]], dtype=np.int64)
    result = ssfl_logic.vote(labels, num_classes=3)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, [2, -1, 1, 1])


def test_vote_tie_breaks_to_lowest_class_index():
    labels = np.array([[0], [1]], dtype=np.int64)
    np.testing.assert_array_equal(ssfl_logic.vote(labels, num_classes=2), [0])
    labels = np.array([[2], [1]], dtype=np.int64)
    np.testing.assert_array_equal(ssfl_logic.vote(labels, num_classes=3), [1])


def test_vote_zero_votes_yield_unlabeled():
    labels = np.array([[-1, 0], [-1, 0]], dtype=np.int64)
    np.testing.assert_array_equal(
        ssfl_logic.vote(labels, num_classes=3), [UNLABELED, 0]
    )


def test_vote_all_unfamiliar_clients_returns_all_unlabeled():
    labels = np.full((3, 5), UNLABELED, dtype=np.int64)
    np.testing.assert_array_equal(
        ssfl_logic.vote(labels, num_classes=11), np.full(5, UNLABELED)
    )


def test_vote_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        ssfl_logic.vote(np.zeros(5, dtype=np.int64), num_classes=3)


# ---------------------------------------------------------------------------
# Aggregation wrapper: vote / no_voting / soft modes + diagnostics
# ---------------------------------------------------------------------------

_REF_LABELS = [
    np.array([2, -1, 0, 1], dtype=np.int64),
    np.array([2, -1, 1, 1], dtype=np.int64),
    np.array([0, -1, 1, 2], dtype=np.int64),
]


def test_aggregate_hard_default_matches_vote_with_diagnostics():
    global_labels, diag = ssfl_logic.aggregate(_REF_LABELS, num_classes=3)
    np.testing.assert_array_equal(global_labels, [2, -1, 1, 1])
    assert diag["zero_vote"] == 1
    # 9 valid votes; agreeing with global: s0 2/3, s2 2/3, s3 2/3 -> 6/9
    assert diag["vote_agreement"] == pytest.approx(6 / 9)


def test_aggregate_all_unlabeled_round_completes():
    payloads = [np.full(4, UNLABELED, dtype=np.int64) for _ in range(3)]
    global_labels, diag = ssfl_logic.aggregate(payloads, num_classes=3)
    np.testing.assert_array_equal(global_labels, np.full(4, UNLABELED))
    assert diag["zero_vote"] == 4
    assert diag["vote_agreement"] == 0.0


def test_aggregate_no_voting_uses_one_hot_mean():
    global_labels, diag = ssfl_logic.aggregate(
        _REF_LABELS, num_classes=3, no_voting=True
    )
    # direct aggregation (one-hot mean + argmax) reproduces the same
    # semantics on this case, including zero-vote -> -1
    np.testing.assert_array_equal(global_labels, [2, -1, 1, 1])
    assert diag["zero_vote"] == 1


def test_aggregate_rejects_no_voting_with_soft_label_mode():
    """no_voting would be silently indistinguishable from soft-mode
    aggregation (both are already a mean); reject the combination outright
    rather than letting the flag silently do nothing."""
    with pytest.raises(ValueError, match="no_voting"):
        ssfl_logic.aggregate(_REF_LABELS, num_classes=3, no_voting=True, label_mode="soft2")


def test_aggregate_soft_mode_averages_and_argmaxes():
    a = np.array([[0.7, 0.2, 0.1], [0.0, 0.0, 0.0], [0.1, 0.8, 0.1]], dtype=np.float32)
    b = np.array([[0.1, 0.1, 0.8], [0.0, 0.0, 0.0], [0.2, 0.6, 0.2]], dtype=np.float32)
    global_labels, diag = ssfl_logic.aggregate(
        [a, b], num_classes=3, label_mode="soft2"
    )
    # sample 0: mean [0.4, 0.15, 0.45] -> 2; sample 1: all zero rows -> -1;
    # sample 2: mean [0.15, 0.7, 0.15] -> 1
    np.testing.assert_array_equal(global_labels, [2, UNLABELED, 1])
    assert global_labels.dtype == np.int64
    assert diag["zero_vote"] == 1
    # valid client hard views: s0 {0, 2} vs global 2 -> 1/2 agree;
    # s2 {1, 1} vs 1 -> 2/2 agree  => 3/4
    assert diag["vote_agreement"] == pytest.approx(3 / 4)


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError):
        ssfl_logic.aggregate([], num_classes=3)


# ---------------------------------------------------------------------------
# Client round (Algorithm 1 unrolled, Eqs. 11-16)
# ---------------------------------------------------------------------------


class TestClientRound:
    def test_hard_payload_matches_contract(self, rng):
        spec = payload_spec("ssfl", "client_to_server")
        X, y = _data(rng)
        open_X = _open(rng)
        payload, diag = ssfl_logic.client_round(
            _classifier(), _discriminator(), X, y, open_X, None, **_ROUND_KW
        )
        assert payload.dtype == np.dtype(spec.dtype)  # int64
        assert payload.shape == (N_OPEN,)  # (n_open,)
        assert (
            (payload == UNLABELED) | ((payload >= 0) & (payload < NUM_CLASSES))
        ).all()
        assert isinstance(diag["unfamiliar"], int)
        assert 0 <= diag["unfamiliar"] <= N_OPEN
        assert isinstance(diag["threshold"], float)

    def test_rejects_empty_private_dataset(self, rng):
        open_X = _open(rng)
        with pytest.raises(ValueError, match="non-empty"):
            ssfl_logic.client_round(
                _classifier(), _discriminator(),
                np.zeros((0, 23, 5), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                open_X, None, **_ROUND_KW,
            )

    def test_trains_classifier_on_private_data(self, rng):
        X, y = _data(rng)
        open_X = _open(rng)
        clf = _classifier()
        before = _state(clf)
        ssfl_logic.client_round(clf, _discriminator(), X, y, open_X, None, **_ROUND_KW)
        assert not _states_equal(before, _state(clf))

    def test_deterministic_given_seed(self, rng):
        X, y = _data(rng)
        open_X = _open(rng)
        results = []
        for _ in range(2):
            clf, disc = _classifier(), _discriminator()
            payload, _ = ssfl_logic.client_round(
                clf, disc, X, y, open_X, None, **_ROUND_KW
            )
            results.append((payload, _state(clf), _state(disc)))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert _states_equal(results[0][1], results[1][1])
        assert _states_equal(results[0][2], results[1][2])

    def test_first_round_and_all_unlabeled_globals_skip_distillation(self, rng):
        """global_labels=None (round 1) and an all-UNLABELED vector must both
        skip distillation and produce bit-identical rounds."""
        X, y = _data(rng)
        open_X = _open(rng)
        states = []
        for globals_ in (None, np.full(N_OPEN, UNLABELED, dtype=np.int64)):
            clf = _classifier()
            ssfl_logic.client_round(
                clf, _discriminator(), X, y, open_X, globals_, **_ROUND_KW
            )
            states.append(_state(clf))
        assert _states_equal(states[0], states[1])

    def test_distillation_on_previous_global_labels_changes_classifier(self, rng):
        X, y = _data(rng)
        open_X = _open(rng)
        globals_ = rng.integers(0, NUM_CLASSES, size=N_OPEN).astype(np.int64)
        states = []
        for g in (None, globals_):
            clf = _classifier()
            ssfl_logic.client_round(clf, _discriminator(), X, y, open_X, g, **_ROUND_KW)
            states.append(_state(clf))
        assert not _states_equal(states[0], states[1])

    def test_unroll_order_and_adr8_epochs(self, rng, monkeypatch):
        """Distill (1 epoch) -> private train (local_epochs) -> discriminator
        (1 epoch), in that order, on the right models and sample counts."""
        X, y = _data(rng)
        open_X = _open(rng)
        globals_ = np.array([UNLABELED, 3] * (N_OPEN // 2), dtype=np.int64)
        clf, disc = _classifier(), _discriminator()

        calls = []
        real_train = ssfl_logic._train

        def spy(model, Xa, ya, *, epochs, **kw):
            calls.append((model, len(Xa), epochs))
            return real_train(model, Xa, ya, epochs=epochs, **kw)

        monkeypatch.setattr(ssfl_logic, "_train", spy)
        ssfl_logic.client_round(
            clf,
            disc,
            X,
            y,
            open_X,
            globals_,
            lr=1e-3,
            batch=8,
            local_epochs=3,
            seed=5,
            device="cpu",
        )
        assert len(calls) == 3
        distill_call, private_call, disc_call = calls
        assert distill_call[0] is clf and distill_call[1] == N_OPEN // 2
        assert distill_call[2] == 1  # ADR-8: distillation 1 epoch
        assert private_call[0] is clf and private_call[1] == N_PRIVATE
        assert private_call[2] == 3  # Eq. 11: configured local epochs
        assert disc_call[0] is disc
        assert disc_call[2] == 1  # ADR-8: discriminator 1 epoch

    def test_discriminator_verdict_decides_filtering(self, rng):
        """Eqs. 15-16: -1 exactly where the *discriminator* says unfamiliar,
        classifier argmax everywhere else; the discriminator gets trained."""
        X, y = _data(rng)
        open_X = _open(rng)
        clf, disc = _classifier(), _discriminator()
        disc_before = _state(disc)
        payload, diag = ssfl_logic.client_round(
            clf, disc, X, y, open_X, None, **_ROUND_KW
        )
        assert not _states_equal(disc_before, _state(disc))

        probs = ssfl_logic.softmax(
            ssfl_logic.predict_logits(clf, open_X, batch=8, device="cpu")
        )
        preds = probs.argmax(axis=1).astype(np.int64)
        verdict = (
            ssfl_logic.predict_logits(disc, open_X, batch=8, device="cpu").argmax(
                axis=1
            )
            == 1
        )
        np.testing.assert_array_equal(payload, np.where(verdict, UNLABELED, preds))
        assert diag["unfamiliar"] == int(verdict.sum())

    def test_median_threshold_reported_in_diag(self, rng):
        X, y = _data(rng)
        open_X = _open(rng)
        clf = _classifier()
        _, diag = ssfl_logic.client_round(
            clf, _discriminator(), X, y, open_X, None, **_ROUND_KW
        )
        probs = ssfl_logic.softmax(
            ssfl_logic.predict_logits(clf, open_X, batch=8, device="cpu")
        )
        assert diag["threshold"] == float(np.median(probs.max(axis=1)))

    def test_all_unfamiliar_client_round_completes(self, rng):
        """A discriminator that rejects everything yields an all -1 upload;
        the round, the vote and the server step must all still complete."""
        X, y = _data(rng)
        open_X = _open(rng)
        rigged = _ConstantLogits([0.0, 10.0])  # always class 1 = unfamiliar
        payload, diag = ssfl_logic.client_round(
            _classifier(), rigged, X, y, open_X, None, **_ROUND_KW
        )
        np.testing.assert_array_equal(payload, np.full(N_OPEN, UNLABELED))
        assert diag["unfamiliar"] == N_OPEN

        global_labels, agg_diag = ssfl_logic.aggregate(
            [payload], num_classes=NUM_CLASSES
        )
        np.testing.assert_array_equal(global_labels, np.full(N_OPEN, UNLABELED))
        assert agg_diag["zero_vote"] == N_OPEN

        server = _classifier(seed=9)
        before = _state(server)
        trained = ssfl_logic.server_step(
            server, open_X, global_labels, lr=1e-3, batch=8, seed=0, device="cpu"
        )
        assert trained == 0
        assert _states_equal(before, _state(server))  # nothing to train on

    def test_single_class_client_completes_all_steps(self, rng):
        """Scenario 2 shape: one attack class per client. All five steps run."""
        X = rng.normal(size=(N_PRIVATE, 23, 5)).astype(np.float32)
        y = np.full(N_PRIVATE, 7, dtype=np.int64)  # single class
        open_X = _open(rng)
        globals_ = rng.integers(0, NUM_CLASSES, size=N_OPEN).astype(np.int64)
        clf, disc = _classifier(), _discriminator()
        clf_before, disc_before = _state(clf), _state(disc)
        payload, diag = ssfl_logic.client_round(
            clf, disc, X, y, open_X, globals_, **_ROUND_KW
        )
        assert not _states_equal(clf_before, _state(clf))  # distill + train
        assert not _states_equal(disc_before, _state(disc))  # discriminator
        assert payload.dtype == np.int64 and payload.shape == (N_OPEN,)
        assert (
            (payload == UNLABELED) | ((payload >= 0) & (payload < NUM_CLASSES))
        ).all()
        assert 0 <= diag["unfamiliar"] <= N_OPEN

    def test_rejects_bad_global_labels_shape_and_label_mode(self, rng):
        X, y = _data(rng)
        open_X = _open(rng)
        with pytest.raises(ValueError):
            ssfl_logic.client_round(
                _classifier(),
                _discriminator(),
                X,
                y,
                open_X,
                np.zeros(N_OPEN + 1, dtype=np.int64),
                **_ROUND_KW,
            )
        with pytest.raises(ValueError):
            ssfl_logic.client_round(
                _classifier(),
                _discriminator(),
                X,
                y,
                open_X,
                None,
                label_mode="soft3",
                **_ROUND_KW,
            )


# ---------------------------------------------------------------------------
# Ablation flags: each alters only its own mechanism
# ---------------------------------------------------------------------------


class TestAblationFlags:
    def _run(self, rng_seed=42, **flags):
        rng = np.random.default_rng(rng_seed)
        X, y = _data(rng)
        open_X = _open(rng)
        clf, disc = _classifier(), _discriminator()
        payload, diag = ssfl_logic.client_round(
            clf, disc, X, y, open_X, None, **_ROUND_KW, **flags
        )
        return payload, diag, clf, disc

    def test_no_discriminating_uploads_all_predictions(self):
        base_payload, _, base_clf, _ = self._run()
        payload, diag, clf, disc = self._run(no_discriminating=True)
        # classifier training untouched by the flag
        assert _states_equal(_state(base_clf), _state(clf))
        # discriminator never trained
        assert _states_equal(_state(_discriminator()), _state(disc))
        # no filtering: prediction uploaded for every open sample
        assert (payload >= 0).all()
        assert diag["unfamiliar"] == 0
        # Replicate _run()'s RNG consumption order (_data then _open on the
        # same generator) to recover the exact open_X it used.
        rng = np.random.default_rng(42)
        _data(rng)
        probs = ssfl_logic.softmax(
            ssfl_logic.predict_logits(
                clf, np.asarray(_open(rng)), batch=8, device="cpu"
            )
        )
        np.testing.assert_array_equal(payload, probs.argmax(axis=1).astype(np.int64))
        # familiar entries agree with the unfiltered run everywhere they exist
        kept = base_payload != UNLABELED
        np.testing.assert_array_equal(base_payload[kept], payload[kept])

    def test_simply_filtering_thresholds_without_discriminator_model(self):
        _, _, base_clf, _ = self._run()
        payload, diag, clf, disc = self._run(simply_filtering=True)
        assert _states_equal(_state(base_clf), _state(clf))
        assert _states_equal(_state(_discriminator()), _state(disc))  # untrained
        # unfamiliar = confidence strictly below the per-client median (Eq. 13)
        # Replicate _run()'s RNG consumption order (_data then _open on the
        # same generator) to recover the exact open_X it used.
        rng = np.random.default_rng(42)
        _data(rng)
        probs = ssfl_logic.softmax(
            ssfl_logic.predict_logits(clf, _open(rng), batch=8, device="cpu")
        )
        conf = probs.max(axis=1)
        preds = probs.argmax(axis=1).astype(np.int64)
        low = conf < float(np.median(conf))
        np.testing.assert_array_equal(payload, np.where(low, UNLABELED, preds))
        assert diag["unfamiliar"] == int(low.sum())

    def test_fixed_threshold_changes_only_the_filter_set(self):
        p_med, d_med, clf_med, _ = self._run(simply_filtering=True)
        p_fix, d_fix, clf_fix, _ = self._run(simply_filtering=True, threshold=0.9)
        assert _states_equal(_state(clf_med), _state(clf_fix))
        assert d_fix["threshold"] == 0.9
        assert d_med["threshold"] != 0.9
        # untrained-ish tiny model: max softmax over 11 classes < 0.9 everywhere
        assert d_fix["unfamiliar"] == N_OPEN
        assert d_med["unfamiliar"] < N_OPEN  # median leaves the upper half in

    @pytest.mark.parametrize(
        "mode,decimals",
        [
            ("soft2", 2),
            ("soft4", 4),
            ("soft6", 6),
            ("soft8", 8),
        ],
    )
    def test_soft_label_mode_changes_only_the_payload(self, mode, decimals):
        hard_payload, hard_diag, clf_h, disc_h = self._run()
        payload, diag, clf_s, disc_s = self._run(label_mode=mode)
        # training identical in both runs
        assert _states_equal(_state(clf_h), _state(clf_s))
        assert _states_equal(_state(disc_h), _state(disc_s))
        assert diag == hard_diag
        # payload contract: float32 [N_o, L], rounded to `decimals`
        spec = payload_spec("ssfl", "client_to_server", label_mode=mode)
        assert payload.dtype == np.dtype(spec.dtype)  # float32
        assert payload.shape == (N_OPEN, NUM_CLASSES)
        np.testing.assert_array_equal(
            payload, np.round(payload.astype(np.float64), decimals).astype(np.float32)
        )
        unfamiliar = hard_payload == UNLABELED
        assert (payload[unfamiliar] == 0.0).all()
        # familiar rows: near-distributions whose argmax is the hard label
        if (~unfamiliar).any():
            np.testing.assert_array_equal(
                payload[~unfamiliar].argmax(axis=1), hard_payload[~unfamiliar]
            )
            np.testing.assert_allclose(
                payload[~unfamiliar].sum(axis=1),
                1.0,
                atol=NUM_CLASSES * 10.0 ** (-decimals),
            )

    def test_no_voting_is_a_server_side_flag(self):
        """no_voting alters aggregation only — the client round has no such
        knob and client payloads are untouched by it."""
        assert "no_voting" not in inspect.signature(ssfl_logic.client_round).parameters
        assert "no_voting" in inspect.signature(ssfl_logic.aggregate).parameters


# ---------------------------------------------------------------------------
# Server step: train on voted labels (!= -1) only
# ---------------------------------------------------------------------------


class TestServerStep:
    def test_trains_on_labeled_samples_only(self, rng):
        open_X = _open(rng)
        labels = np.full(N_OPEN, UNLABELED, dtype=np.int64)
        labels[:4] = [0, 3, 3, 7]
        model = _classifier()
        before = _state(model)
        trained = ssfl_logic.server_step(
            model, open_X, labels, lr=1e-3, batch=8, seed=0, device="cpu"
        )
        assert trained == 4
        assert not _states_equal(before, _state(model))

    def test_noop_when_everything_unlabeled(self, rng):
        open_X = _open(rng)
        model = _classifier()
        before = _state(model)
        trained = ssfl_logic.server_step(
            model,
            open_X,
            np.full(N_OPEN, UNLABELED, dtype=np.int64),
            lr=1e-3,
            batch=8,
            seed=0,
            device="cpu",
        )
        assert trained == 0
        assert _states_equal(before, _state(model))

    def test_deterministic_given_seed(self, rng):
        open_X = _open(rng)
        labels = rng.integers(0, NUM_CLASSES, size=N_OPEN).astype(np.int64)
        states = []
        for _ in range(2):
            model = _classifier()
            ssfl_logic.server_step(
                model, open_X, labels, lr=1e-3, batch=8, seed=3, device="cpu"
            )
            states.append(_state(model))
        assert _states_equal(states[0], states[1])


# ---------------------------------------------------------------------------
# Evaluation: per-round top-1 accuracy + final macro metrics
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_constant_predictor_accuracy(self, rng):
        X, y = _data(rng, n=50)
        logits = np.zeros(NUM_CLASSES, dtype=np.float32)
        logits[3] = 10.0
        acc = ssfl_logic.evaluate(_ConstantLogits(logits), X, y, batch=16, device="cpu")
        assert acc == pytest.approx(float((y == 3).mean()))

    def test_final_metrics_reports_full_set(self, rng):
        X, y = _data(rng, n=30)
        logits = np.zeros(NUM_CLASSES, dtype=np.float32)
        logits[0] = 10.0
        m = ssfl_logic.final_metrics(
            _ConstantLogits(logits),
            X,
            y,
            num_classes=NUM_CLASSES,
            batch=16,
            device="cpu",
        )
        assert set(m) >= {"accuracy", "macro_f1", "macro_precision", "confusion_matrix"}
        assert m["accuracy"] == pytest.approx(float((y == 0).mean()))
        cm = m["confusion_matrix"]
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
        assert cm.sum() == 30
        assert cm[:, 1:].sum() == 0  # everything predicted as class 0

    def test_classification_metrics_hand_example(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        m = ssfl_logic.classification_metrics(y_true, y_pred, num_classes=3)
        np.testing.assert_array_equal(
            m["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
        )
        assert m["accuracy"] == pytest.approx(4 / 6)
        assert m["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
        assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


# ---------------------------------------------------------------------------
# Round driver: one full SSFL round without any FL framework (ADR-4)
# ---------------------------------------------------------------------------


class TestRunRound:
    def _setup(self, rng, n_clients=3):
        client_states = [
            (_classifier(seed=k), _discriminator(seed=100 + k))
            for k in range(n_clients)
        ]
        clients_data = [_data(rng) for _ in range(n_clients)]
        open_X = _open(rng)
        server_model = _classifier(seed=99)
        return client_states, clients_data, open_X, server_model

    def test_round_returns_globals_and_diagnostics(self, rng):
        client_states, clients_data, open_X, server = self._setup(rng)
        server_before = _state(server)
        globals_, diag = ssfl_logic.run_round(
            client_states,
            server,
            clients_data,
            open_X,
            None,
            round_num=1,
            run_seed=0,
            num_classes=NUM_CLASSES,
            lr=1e-3,
            batch=8,
            local_epochs=1,
            device="cpu",
        )
        assert globals_.dtype == np.int64 and globals_.shape == (N_OPEN,)
        assert (
            (globals_ == UNLABELED) | ((globals_ >= 0) & (globals_ < NUM_CLASSES))
        ).all()
        assert diag["round"] == 1
        assert len(diag["unfamiliar_per_client"]) == 3
        assert diag["zero_vote"] == int((globals_ == UNLABELED).sum())
        assert 0.0 <= diag["vote_agreement"] <= 1.0
        assert diag["server_trained_on"] == N_OPEN - diag["zero_vote"]
        if diag["server_trained_on"]:
            assert not _states_equal(server_before, _state(server))

    def test_run_round_deterministic_given_seed(self):
        """Bit-for-bit determinism (module docstring's load-bearing claim),
        checked on synthetic data rather than the slow real-cache fixture."""
        kw = dict(round_num=2, run_seed=13, num_classes=NUM_CLASSES,
                   lr=1e-3, batch=8, local_epochs=1, device="cpu")
        results = []
        for _ in range(2):
            client_states, clients_data, open_X, server = self._setup(
                np.random.default_rng(21)
            )
            globals_, _ = ssfl_logic.run_round(
                client_states, server, clients_data, open_X, None, **kw
            )
            results.append((globals_, _state(server)))

        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert _states_equal(results[0][1], results[1][1])

    @pytest.mark.parametrize(
        "flags",
        [
            dict(no_voting=True),
            dict(no_discriminating=True),
            dict(simply_filtering=True),
            dict(no_discriminating=True, no_voting=True),
            dict(threshold=0.7),
            dict(label_mode="soft2"),
        ],
    )
    def test_every_ablation_variant_completes_a_round(self, rng, flags):
        client_states, clients_data, open_X, server = self._setup(rng, n_clients=2)
        globals_, diag = ssfl_logic.run_round(
            client_states,
            server,
            clients_data,
            open_X,
            None,
            round_num=1,
            run_seed=3,
            num_classes=NUM_CLASSES,
            lr=1e-3,
            batch=8,
            local_epochs=1,
            device="cpu",
            **flags,
        )
        assert globals_.dtype == np.int64 and globals_.shape == (N_OPEN,)
        assert (
            (globals_ == UNLABELED) | ((globals_ >= 0) & (globals_ < NUM_CLASSES))
        ).all()
        assert len(diag["unfamiliar_per_client"]) == 2

    def test_kwargs_align_with_runconfig_fields(self, rng):
        """The driver is drivable straight from a validated RunConfig."""
        cfg = RunConfig(
            method="ssfl",
            model="cnn",
            scenario=1,
            seed=11,
            lr=1e-3,
            batch=8,
            local_epochs=1,
            threshold=0.8,
            simply_filtering=True,
        )
        client_states, clients_data, open_X, server = self._setup(rng, n_clients=2)
        globals_, _ = ssfl_logic.run_round(
            client_states,
            server,
            clients_data,
            open_X,
            None,
            round_num=1,
            run_seed=cfg.seed,
            num_classes=NUM_CLASSES,
            lr=cfg.lr,
            batch=cfg.batch,
            local_epochs=cfg.local_epochs,
            threshold=cfg.threshold,
            no_voting=cfg.no_voting,
            no_discriminating=cfg.no_discriminating,
            simply_filtering=cfg.simply_filtering,
            label_mode=cfg.label_mode,
            device="cpu",
        )
        assert globals_.shape == (N_OPEN,)

    def test_rejects_mismatched_models_and_data(self, rng):
        client_states, clients_data, open_X, server = self._setup(rng)
        with pytest.raises(ValueError):
            ssfl_logic.run_round(
                client_states[:1],
                server,
                clients_data,
                open_X,
                None,
                round_num=1,
                run_seed=0,
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
        clients.append(
            (
                np.asarray(X[:48], dtype=np.float32),
                np.asarray(y[:48], dtype=np.int64),
            )
        )
    open_X = np.asarray(load_open()[:40], dtype=np.float32)
    Xt, yt = load_test()
    return (
        clients,
        open_X,
        (
            np.asarray(Xt[:128], dtype=np.float32),
            np.asarray(yt[:128], dtype=np.int64),
        ),
    )


class TestTwoRoundMicroRun:
    def _run(self, cache_subsets, run_seed=42):
        from ssfl.models import build_model

        clients_data, open_X, (X_test, y_test) = cache_subsets
        torch.manual_seed(run_seed)
        # classifier = paper model; discriminator = the 2-class CNN variant
        client_states = [
            (build_model("mlp"), build_model("cnn", num_classes=2))
            for _ in clients_data
        ]
        server_model = build_model("mlp")

        global_labels, accs, diags = None, [], []
        for round_num in (1, 2):
            global_labels, diag = ssfl_logic.run_round(
                client_states,
                server_model,
                clients_data,
                open_X,
                global_labels,
                round_num=round_num,
                run_seed=run_seed,
                lr=1e-4,
                batch=16,
                local_epochs=1,
                device="cpu",
            )
            accs.append(ssfl_logic.evaluate(server_model, X_test, y_test, device="cpu"))
            diags.append(diag)
        return global_labels, accs, diags, server_model, (X_test, y_test)

    def test_two_rounds_report_server_accuracy_and_final_metrics(self, cache_subsets):
        global_labels, accs, diags, server_model, (X_test, y_test) = self._run(
            cache_subsets
        )
        assert len(accs) == 2
        assert all(0.0 <= a <= 1.0 for a in accs)
        assert [d["round"] for d in diags] == [1, 2]
        for d in diags:
            assert len(d["unfamiliar_per_client"]) == 2
            assert 0 <= d["zero_vote"] <= 40
            assert 0.0 <= d["vote_agreement"] <= 1.0
        assert global_labels.shape == (40,) and global_labels.dtype == np.int64
        m = ssfl_logic.final_metrics(
            server_model, X_test, y_test, num_classes=NUM_CLASSES, device="cpu"
        )
        assert set(m) >= {"accuracy", "macro_f1", "macro_precision", "confusion_matrix"}
        assert m["accuracy"] == pytest.approx(accs[-1])
        assert m["confusion_matrix"].sum() == len(y_test)

    def test_micro_run_is_reproducible(self, cache_subsets):
        g1, accs1, _, model1, _ = self._run(cache_subsets, run_seed=7)
        g2, accs2, _, model2, _ = self._run(cache_subsets, run_seed=7)
        np.testing.assert_array_equal(g1, g2)
        assert accs1 == accs2
        for k, v in model1.state_dict().items():
            assert torch.equal(v, model2.state_dict()[k])
