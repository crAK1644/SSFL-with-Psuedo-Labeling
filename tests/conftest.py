"""Shared test setup.

PYTORCH_ENABLE_MPS_FALLBACK must be set before torch initializes the MPS
backend: some ops (historically LSTM internals) lack MPS kernels and then
fall back to CPU per-op instead of raising. Documented in the SDD
(Known Technical Issues) and in ssfl.models.LSTMNet.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
