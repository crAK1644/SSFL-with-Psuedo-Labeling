---
unit: fd1
framework: pytest
---
# E2E Test Stubs — FD method logic

## Setup

```python
# tests/e2e/test_federated_distillation.py
import numpy as np
import pytest
```

## Stubs

### absent-class-and-exclude-self (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_absent_class_and_exclude_self():
    from ssfl.methods import fd_logic
    X = np.random.rand(20, 23, 5).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)   # class 2 absent, L=3
    local = fd_logic.per_class_logits(model=None, X=X, y=y, num_classes=3, untrained_ok=True)
    np.testing.assert_allclose(local[2], 0.0)
    mats = [np.full((3, 3), v, dtype=np.float32) for v in (1.0, 2.0, 3.0)]
    targets = fd_logic.aggregate(mats, counts=[1, 1, 1])
    np.testing.assert_allclose(targets[0][0], 2.5)  # avg of others: (2+3)/2
```

### fd-micro-run-completes (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_fd_micro_run_completes():
    from ssfl.methods import fd_logic
    import sys
    result = fd_logic.run_rounds(num_clients=3, rounds=2, seed=0, tiny=True)
    assert "flwr" not in sys.modules and "ray" not in sys.modules
    assert 0.0 <= result.accuracies[-1] <= 1.0
```
