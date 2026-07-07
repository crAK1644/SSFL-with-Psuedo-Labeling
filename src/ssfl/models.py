"""Model zoo for the SSFL paper reproduction.

Pure PyTorch (no Flower imports). Provides:

- :class:`TableICNN` — the paper's Table I Conv1D network. ``num_classes=11``
  builds the classifier head, ``num_classes=2`` the discriminator head.
- :class:`MLP` and :class:`LSTMNet` — the paper's comparison models, sized
  comparably to the CNN and sharing its input/output signature.
- A device helper resolving ``"auto"`` to the best available backend.

Input signature for all models: ``[B, 23, 5]`` (115 N-BaIoT features
reshaped per Eq. 19 — 23 channels over a length-5 sequence).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "IN_CHANNELS",
    "SEQ_LEN",
    "NUM_CLASSES",
    "TableICNN",
    "MLP",
    "LSTMNet",
    "build_model",
    "resolve_device",
]

IN_CHANNELS = 23
SEQ_LEN = 5
NUM_CLASSES = 11


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device name to a :class:`torch.device`.

    ``"auto"`` picks cuda if available, else mps if available, else cpu.
    Any other name is passed to ``torch.device`` verbatim (torch raises on
    invalid names).
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class TableICNN(nn.Module):
    """Table I Conv1D CNN.

    Layers 1-4: Conv1d 64 filters, kernel 3, stride 1  -> (B, 64, 5)
    Layers 5-6: Conv1d 128 filters, kernel 3, stride 1 -> (B, 128, 5)
    Layer 7:    Conv1d 128 filters, kernel 3, stride 2 -> (B, 128, 3)
    Layer 8:    Conv1d 128 filters, kernel 3, stride 2 -> (B, 128, 2)
    Flatten -> Linear(256, 128) -> Linear(128, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes

        def conv(cin: int, cout: int, stride: int = 1) -> nn.Conv1d:
            return nn.Conv1d(cin, cout, kernel_size=3, stride=stride, padding=1)

        self.conv1 = conv(IN_CHANNELS, 64)
        self.conv2 = conv(64, 64)
        self.conv3 = conv(64, 64)
        self.conv4 = conv(64, 64)
        self.conv5 = conv(64, 128)
        self.conv6 = conv(128, 128)
        self.conv7 = conv(128, 128, stride=2)
        self.conv8 = conv(128, 128, stride=2)
        self.fc = nn.Linear(128 * 2, 128)
        self.head = nn.Linear(128, num_classes)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4,
                      self.conv5, self.conv6, self.conv7, self.conv8):
            x = self.act(layer(x))
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc(x))
        return self.head(x)


class MLP(nn.Module):
    """Fully-connected comparison model: flattened 23*5=115 features in."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IN_CHANNELS * SEQ_LEN, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMNet(nn.Module):
    """LSTM comparison model.

    Treats the ``[B, 23, 5]`` input as a length-5 sequence of 23 features
    (time axis last, permuted internally to batch-first ``[B, 5, 23]``).

    Note: on Apple MPS some LSTM ops may be unimplemented; run with
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so those ops fall back to CPU.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_size: int = 128,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=IN_CHANNELS,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)          # [B, 23, 5] -> [B, 5, 23]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last time step


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn": TableICNN,
    "mlp": MLP,
    "lstm": LSTMNet,
}


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a model by name (``"cnn"``, ``"mlp"``, ``"lstm"``), case-insensitive.

    ``num_classes=11`` for classifiers, ``num_classes=2`` for the SSFL
    discriminator (CNN backbone).
    """
    try:
        cls = _MODEL_REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown model {name!r}; expected one of {sorted(_MODEL_REGISTRY)}"
        ) from None
    return cls(num_classes=num_classes)
