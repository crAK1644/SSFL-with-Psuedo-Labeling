---
unit: ss1
framework: pytest
---
# E2E Test Stubs — SSFL method logic

## Setup

```python
# tests/e2e/test_ssfl_method.py
import numpy as np
import pytest
```

## Stubs

### vote-matches-walkthrough (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_vote_matches_walkthrough():
    from ssfl.methods import ssfl_logic
    labels = np.array([[2, -1, 0, 1], [2, -1, 1, 1], [0, -1, 1, 2]], dtype=np.int64)
    out = ssfl_logic.vote(labels, num_classes=3)
    np.testing.assert_array_equal(out, [2, -1, 1, 1])
    tie = np.array([[0, 0], [1, -1]], dtype=np.int64)
    assert ssfl_logic.vote(tie, num_classes=3)[0] == 0
    assert ssfl_logic.vote(tie, num_classes=3)[0] == 0  # deterministic
```

### client-round-filters-unfamiliar (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_client_round_filters_unfamiliar():
    from ssfl.methods import ssfl_logic
    normal = ssfl_logic.run_client_round_fixture(kind="normal", seed=0)
    assert normal.dtype == np.int64
    assert (normal == -1).any() and (normal >= 0).any()
    single = ssfl_logic.run_client_round_fixture(kind="single_class", seed=0)
    assert single.shape == normal.shape
    allunf = ssfl_logic.run_client_round_fixture(kind="all_unfamiliar", seed=0)
    assert (allunf == -1).all()
    assert (ssfl_logic.vote(allunf[None, :], num_classes=11) == -1).all()
```

### ablation-flags-are-isolated (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_ablation_flags_are_isolated():
    from ssfl.methods import ssfl_logic
    base = ssfl_logic.run_client_round_fixture(kind="normal", seed=1)
    nodisc = ssfl_logic.run_client_round_fixture(kind="normal", seed=1, no_discriminating=True)
    assert (nodisc >= 0).all()
    soft = ssfl_logic.run_client_round_fixture(kind="normal", seed=1, label_mode="soft2")
    assert soft.dtype == np.float32 and soft.ndim == 2
    np.testing.assert_allclose(soft, np.round(soft, 2))
```
