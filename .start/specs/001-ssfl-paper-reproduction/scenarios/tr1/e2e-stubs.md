---
unit: tr1
framework: pytest
---
# E2E Test Stubs — Flower transport

## Setup

```python
# tests/e2e/test_flower_transport.py
import json
from pathlib import Path

import numpy as np
import pytest
```

## Stubs

### flwr-two-round-e2e (P0)

```python
@pytest.mark.skip(reason="evaluation phase; slow — uses Ray")
@pytest.mark.parametrize("method", ["fl", "ssfl"])
def test_flwr_two_round_e2e(method, tmp_path):
    from ssfl.run import launch  # documented launch path
    run_dir = launch(method=method, model="cnn", scenario=1, seed=0,
                     rounds=2, num_clients=4, results_root=tmp_path)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["method"] == method and cfg["rounds"] == 2
    lines = [json.loads(l) for l in (run_dir / "rounds.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert all(0.0 <= l["test_acc"] <= 1.0 for l in lines)
    final = json.loads((run_dir / "final.json").read_text())
    assert {"accuracy", "f1", "precision"} <= final.keys()
    assert np.load(run_dir / "cm.npy").shape == (11, 11)
    if method == "ssfl":
        assert "diag" in lines[-1]
    assert any((run_dir / "ckpt").iterdir())
```

### failed-client-excluded (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_failed_client_excluded():
    from ssfl.transport.server_app import build_strategy
    strategy = build_strategy(method="ssfl", num_classes=3, n_open=4)
    valid = np.array([[2, -1, 0, 1], [2, -1, 1, 1]], dtype=np.int64)
    result = strategy.aggregate_test_hook(valid_labels=valid, num_errors=1)
    np.testing.assert_array_equal(result.global_labels, [2, -1, 1, 1])
    assert result.failed_clients == 1
```
