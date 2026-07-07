"""Tests for ssfl.methods.payloads: the per-method payload contract."""

import math

import numpy as np
import pytest

from ssfl.methods import payloads
from ssfl.methods.payloads import UNLABELED, payload_nbytes, payload_spec


class TestContractDeclarations:
    def test_unlabeled_sentinel(self):
        assert UNLABELED == -1

    def test_fl_weights_both_directions(self):
        for direction in ("server_to_client", "client_to_server"):
            spec = payload_spec("fl", direction)
            assert spec.dtype == np.float32
            assert spec.shape == ("param_count",)

    def test_fd_per_class_logits(self):
        for direction in ("server_to_client", "client_to_server"):
            spec = payload_spec("fd", direction)
            assert spec.dtype == np.float32
            assert spec.shape == ("num_classes", "num_classes")

    def test_dsfl_open_set_logits(self):
        for direction in ("server_to_client", "client_to_server"):
            spec = payload_spec("dsfl", direction)
            assert spec.dtype == np.float32
            assert spec.shape == ("n_open", "num_classes")

    def test_ssfl_hard_labels(self):
        for direction in ("server_to_client", "client_to_server"):
            spec = payload_spec("ssfl", direction)
            assert spec.dtype == np.int64
            assert spec.shape == ("n_open",)
            assert spec.decimals is None

    def test_ssfl_soft_modes_upload_spec(self):
        for mode, decimals in [("soft2", 2), ("soft4", 4), ("soft6", 6), ("soft8", 8)]:
            spec = payload_spec("ssfl", "client_to_server", label_mode=mode)
            assert spec.dtype == np.float32
            assert spec.shape == ("n_open", "num_classes")
            assert spec.decimals == decimals

    def test_ssfl_soft_mode_download_stays_hard(self):
        spec = payload_spec("ssfl", "server_to_client", label_mode="soft4")
        assert spec.dtype == np.int64
        assert spec.shape == ("n_open",)

    def test_unknown_method_or_direction_rejected(self):
        with pytest.raises(ValueError):
            payload_spec("fedavg", "server_to_client")
        with pytest.raises(ValueError):
            payload_spec("fl", "sideways")
        with pytest.raises(ValueError):
            payload_spec("ssfl", "client_to_server", label_mode="soft3")


class TestByteSizes:
    def test_fl_bytes_from_param_count(self):
        assert payload_nbytes("fl", "server_to_client", param_count=1000) == 4000
        assert payload_nbytes("fl", "client_to_server", param_count=1000) == 4000

    def test_fd_bytes(self):
        assert payload_nbytes("fd", "client_to_server", num_classes=11) == 11 * 11 * 4

    def test_dsfl_bytes(self):
        assert (
            payload_nbytes("dsfl", "client_to_server", n_open=8900, num_classes=11)
            == 8900 * 11 * 4
        )

    def test_ssfl_hard_bytes(self):
        for direction in ("server_to_client", "client_to_server"):
            assert payload_nbytes("ssfl", direction, n_open=8900) == 8900 * 8

    def test_ssfl_soft_bytes_use_decimal_precision(self):
        n_open, L = 100, 11
        for mode, decimals in [("soft2", 2), ("soft4", 4), ("soft6", 6), ("soft8", 8)]:
            bits_per_value = math.ceil(math.log2(10**decimals + 1))
            expected = math.ceil(n_open * L * bits_per_value / 8)
            got = payload_nbytes(
                "ssfl", "client_to_server", n_open=n_open, num_classes=L, label_mode=mode
            )
            assert got == expected, mode

    def test_ssfl_soft_bytes_monotonic_in_precision(self):
        sizes = [
            payload_nbytes("ssfl", "client_to_server", n_open=1000, num_classes=11, label_mode=m)
            for m in ("soft2", "soft4", "soft6", "soft8")
        ]
        assert sizes == sorted(sizes) and len(set(sizes)) == 4

    def test_ssfl_soft_download_still_hard_sized(self):
        assert (
            payload_nbytes("ssfl", "server_to_client", n_open=500, num_classes=11, label_mode="soft4")
            == 500 * 8
        )

    def test_missing_required_dims_rejected(self):
        with pytest.raises(ValueError):
            payload_nbytes("fl", "server_to_client")  # no param_count
        with pytest.raises(ValueError):
            payload_nbytes("fd", "client_to_server")  # no num_classes
        with pytest.raises(ValueError):
            payload_nbytes("dsfl", "client_to_server", n_open=10)  # no num_classes
        with pytest.raises(ValueError):
            payload_nbytes("ssfl", "client_to_server")  # no n_open


class TestContractIsSingleSourceOfTruth:
    def test_all_four_methods_declared(self):
        assert set(payloads.PAYLOAD_CONTRACT) == {"fl", "fd", "dsfl", "ssfl"}

    def test_every_method_declares_both_directions(self):
        for method, directions in payloads.PAYLOAD_CONTRACT.items():
            assert set(directions) == {"server_to_client", "client_to_server"}, method
