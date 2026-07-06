---
unit: ds1
framework: pytest
---
# E2E Test Stubs — DS-FL method logic

## Setup

```python
# tests/e2e/test_dsfl.py
import numpy as np
import pytest

def _entropy(p, axis=-1):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=axis)
```

## Stubs

### era-sharpens-distribution (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_era_sharpens_distribution():
    from ssfl.methods import dsfl_logic
    rng = np.random.default_rng(0)
    mats = [rng.normal(size=(10, 11)).astype(np.float32) for _ in range(3)]
    avg_probs = dsfl_logic.softmax(np.mean(mats, axis=0))
    out = dsfl_logic.aggregate(mats)          # ERA applied, T < 1
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)
    assert (out >= 0).all()
    assert (_entropy(out) < _entropy(avg_probs)).all()
    assert (out.argmax(axis=1) == avg_probs.argmax(axis=1)).all()
```

### dsfl-micro-run-server-model (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_dsfl_micro_run_server_model():
    from ssfl.methods import dsfl_logic
    import sys
    result = dsfl_logic.run_rounds(num_clients=3, rounds=2, seed=0, tiny=True)
    assert "flwr" not in sys.modules and "ray" not in sys.modules
    assert 0.0 <= result.accuracies[-1] <= 1.0
    assert result.evaluated_model == "server"
```
