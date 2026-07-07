"""Tests for the model zoo (Table I CNN, MLP, LSTM) and the device helper."""

import pytest
import torch

from ssfl.models import MLP, LSTMNet, TableICNN, build_model, resolve_device

BATCH = 80
IN_CHANNELS = 23
SEQ_LEN = 5
NUM_CLASSES = 11


def _batch(batch_size: int = BATCH) -> torch.Tensor:
    return torch.randn(batch_size, IN_CHANNELS, SEQ_LEN)


class TestTableICNNOutputs:
    def test_classifier_head_outputs_11(self):
        model = TableICNN(num_classes=NUM_CLASSES)
        out = model(_batch())
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_discriminator_head_outputs_2(self):
        model = TableICNN(num_classes=2)
        out = model(_batch())
        assert out.shape == (BATCH, 2)

    def test_output_size_selected_at_construction(self):
        clf = TableICNN(num_classes=NUM_CLASSES)
        disc = TableICNN(num_classes=2)
        x = _batch(4)
        assert clf(x).shape == (4, NUM_CLASSES)
        assert disc(x).shape == (4, 2)


class TestTableICNNLayerShapes:
    """Intermediate activations must match Table I exactly for batch 80."""

    # Table I: (name of conv attribute, expected output shape)
    EXPECTED_CONV_SHAPES = [
        ("conv1", (BATCH, 64, 5)),
        ("conv2", (BATCH, 64, 5)),
        ("conv3", (BATCH, 64, 5)),
        ("conv4", (BATCH, 64, 5)),
        ("conv5", (BATCH, 128, 5)),
        ("conv6", (BATCH, 128, 5)),
        ("conv7", (BATCH, 128, 3)),
        ("conv8", (BATCH, 128, 2)),
    ]

    def _capture_shapes(self, model):
        shapes = {}
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                def hook(mod, args, out, name=name):
                    shapes[name] = tuple(out.shape)
                hooks.append(module.register_forward_hook(hook))
        model(_batch())
        for h in hooks:
            h.remove()
        return shapes

    def test_conv_layer_shapes_match_table_i(self):
        shapes = self._capture_shapes(TableICNN(num_classes=NUM_CLASSES))
        for name, expected in self.EXPECTED_CONV_SHAPES:
            assert shapes[name] == expected, (
                f"{name}: expected {expected}, got {shapes[name]}"
            )

    def test_fc_layer_shape_is_80x128(self):
        shapes = self._capture_shapes(TableICNN(num_classes=NUM_CLASSES))
        assert shapes["fc"] == (BATCH, 128)

    def test_conv_hyperparameters_match_table_i(self):
        model = TableICNN(num_classes=NUM_CLASSES)
        for name, expected in self.EXPECTED_CONV_SHAPES:
            conv = getattr(model, name)
            assert conv.kernel_size == (3,), name
            assert conv.out_channels == expected[1], name
        for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6"):
            assert getattr(model, name).stride == (1,), name
        for name in ("conv7", "conv8"):
            assert getattr(model, name).stride == (2,), name


class TestComparisonModels:
    """MLP and LSTM share the CNN's input/output signature: [B,23,5] -> [B,11]."""

    @pytest.mark.parametrize("model_cls", [MLP, LSTMNet])
    def test_default_signature(self, model_cls):
        model = model_cls()
        out = model(_batch())
        assert out.shape == (BATCH, NUM_CLASSES)

    @pytest.mark.parametrize("model_cls", [MLP, LSTMNet])
    def test_num_classes_configurable(self, model_cls):
        model = model_cls(num_classes=2)
        out = model(_batch(4))
        assert out.shape == (4, 2)

    @pytest.mark.parametrize("model_cls", [MLP, LSTMNet])
    def test_sized_comparably_to_cnn(self, model_cls):
        """Comparison models are the same order of magnitude as the CNN."""
        cnn_params = sum(p.numel() for p in TableICNN().parameters())
        params = sum(p.numel() for p in model_cls().parameters())
        assert cnn_params / 10 <= params <= cnn_params * 10


class TestResolveDevice:
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        device = resolve_device("auto")
        if torch.cuda.is_available():
            expected = "cuda"
        elif torch.backends.mps.is_available():
            expected = "mps"
        else:
            expected = "cpu"
        assert device.type == expected

    def test_returns_torch_device(self):
        assert isinstance(resolve_device("auto"), torch.device)

    def test_explicit_cpu_respected(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_explicit_device_passthrough(self):
        # Explicit names are honored verbatim (validated by torch itself).
        assert resolve_device("cuda").type == "cuda"
        assert resolve_device("mps").type == "mps"

    def test_unknown_name_rejected(self):
        with pytest.raises((ValueError, RuntimeError)):
            resolve_device("not-a-device")


class TestBuildModel:
    @pytest.mark.parametrize(
        "name, cls",
        [("cnn", TableICNN), ("mlp", MLP), ("lstm", LSTMNet)],
    )
    def test_builds_named_model(self, name, cls):
        model = build_model(name)
        assert isinstance(model, cls)
        assert model(_batch(4)).shape == (4, NUM_CLASSES)

    def test_num_classes_forwarded(self):
        disc = build_model("cnn", num_classes=2)
        assert disc(_batch(4)).shape == (4, 2)

    def test_name_is_case_insensitive(self):
        assert isinstance(build_model("CNN"), TableICNN)

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="(?i)unknown model"):
            build_model("transformer")


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


class TestTrainStepOnDevice:
    """Every model completes one forward+backward+optimizer step on each
    available device, including the one resolve_device("auto") picks."""

    @pytest.mark.parametrize("device", _available_devices())
    @pytest.mark.parametrize("name", ["cnn", "mlp", "lstm"])
    def test_one_train_step(self, name, device):
        torch.manual_seed(0)
        model = build_model(name).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        x = torch.randn(BATCH, IN_CHANNELS, SEQ_LEN, device=device)
        y = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)
        before = [p.detach().clone() for p in model.parameters()]

        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

        assert torch.isfinite(loss).item()
        changed = any(
            not torch.equal(b, p.detach())
            for b, p in zip(before, model.parameters())
        )
        assert changed, "optimizer step did not update any parameter"

    def test_train_step_on_resolved_auto_device(self):
        device = resolve_device("auto")
        model = build_model("cnn", num_classes=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        x = torch.randn(BATCH, IN_CHANNELS, SEQ_LEN, device=device)
        y = torch.randint(0, 2, (BATCH,), device=device)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss).item()
