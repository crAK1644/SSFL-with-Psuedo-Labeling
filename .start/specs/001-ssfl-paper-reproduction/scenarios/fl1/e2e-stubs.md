---
unit: fl1
framework: pytest
---
# E2E Test Stubs — FL method logic

## Setup

```python
# tests/e2e/test_fedavg.py
import numpy as np
import pytest
```

## Stubs

### fedavg-weighted-average (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_fedavg_weighted_average():
    from ssfl.methods import fl_logic
    wa = [np.full((4, 4), 1.0, dtype=np.float32)]
    wb = [np.full((4, 4), 5.0, dtype=np.float32)]
    agg = fl_logic.aggregate([(wa, 300), (wb, 100)])
    np.testing.assert_allclose(agg[0], 2.0)
    solo = fl_logic.aggregate([(wb, 100)])
    np.testing.assert_allclose(solo[0], wb[0])
```

### micro-run-learns-deterministically (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_fl_micro_run_learns_deterministically():
    from ssfl.methods import fl_logic
    import sys
    r1 = fl_logic.run_rounds(num_clients=3, rounds=2, seed=0, tiny=True)
    r2 = fl_logic.run_rounds(num_clients=3, rounds=2, seed=0, tiny=True)
    assert "flwr" not in sys.modules and "ray" not in sys.modules
    assert r1.losses[-1] < r1.losses[0]
    assert r1.accuracies == r2.accuracies
```
