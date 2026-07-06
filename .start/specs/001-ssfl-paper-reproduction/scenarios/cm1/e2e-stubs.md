---
unit: cm1
framework: pytest
---
# E2E Test Stubs — Communication-cost accounting

## Setup

```python
# tests/e2e/test_comm_costs.py
import pytest
```

## Stubs

### per-method-costs (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_per_method_costs():
    from ssfl import comm
    from ssfl.methods import payloads
    P = payloads.cnn_param_count()
    fl = comm.cumulative_upload_bytes("fl", rounds=200, clients=27, n_open=8900, num_classes=11)
    ssfl = comm.cumulative_upload_bytes("ssfl", rounds=200, clients=27, n_open=8900, num_classes=11)
    dsfl = comm.cumulative_upload_bytes("dsfl", rounds=200, clients=27, n_open=8900, num_classes=11)
    assert fl == 200 * 27 * 4 * P
    assert ssfl == 200 * 27 * 8 * 8900
    assert ssfl < dsfl < fl
    assert comm.open_set_cost_bytes(n_open=8900) == pytest.approx(0.96e6, rel=0.5)
    assert comm.open_set_cost_bytes(n_open=8900, method="fl") is None
```

### c-at-x-from-curve (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_c_at_x_from_curve():
    from ssfl import comm
    acc = [0.30] * 9 + [0.55] * 90 + [0.70] * 101   # rounds 1..200
    result = comm.c_at(acc_curve=acc, per_round_mb=1.0, targets=[50, 75, "top"])
    assert result[50] == 10.0
    assert result["top"] == 100.0
    assert result[75] == comm.UNREACHED and result[75] != 0
```
