---
unit: fb1
framework: pytest
---
# E2E Test Stubs — Framework base

## Setup

```python
# tests/e2e/test_framework_base.py
import json
from pathlib import Path

import pytest
```

## Stubs

### config-validation-and-runid (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_config_validation_and_runid():
    from ssfl.config import RunConfig, ConfigError
    with pytest.raises(ConfigError) as e1:
        RunConfig(method="fl", no_voting=True).validate()
    with pytest.raises(ConfigError) as e2:
        RunConfig(method="ssfl", threshold="0.85").validate()
    assert "ssfl" in str(e1.value) or "no_voting" in str(e1.value)
    a = RunConfig(method="ssfl", model="cnn", scenario=2, seed=7)
    b = RunConfig(method="ssfl", model="cnn", scenario=2, seed=7)
    assert a.run_id() == b.run_id() == "ssfl-cnn-s2-seed7"
    d = RunConfig(method="fl")
    assert (d.rounds, d.lr, d.batch, d.local_epochs) == (200, 1e-4, 80, 5)
```

### durable-metrics-survive-interrupt (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_durable_metrics_survive_interrupt(tmp_path):
    from ssfl.metrics import MetricsStore, read_rounds
    store = MetricsStore(tmp_path / "run")
    for r in range(1, 6):
        store.append_round(round=r, test_acc=0.1 * r, wall_s=1.0)
    # no close/finalize — simulate abrupt death
    records = read_rounds(tmp_path / "run")
    assert [rec["round"] for rec in records] == [1, 2, 3, 4, 5]
    assert all({"round", "test_acc", "wall_s"} <= rec.keys() for rec in records)
    assert not (tmp_path / "run" / "final.json").exists()
```

### payload-byte-sizes (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_payload_byte_sizes():
    from ssfl.methods import payloads
    P = payloads.cnn_param_count()
    assert payloads.upload_bytes("fl", n_open=8900, num_classes=11) == 4 * P
    assert payloads.upload_bytes("fd", n_open=8900, num_classes=11) == 484
    assert payloads.upload_bytes("dsfl", n_open=8900, num_classes=11) == 391_600
    hard = payloads.upload_bytes("ssfl", n_open=8900, num_classes=11)
    assert hard == 8 * 8900
    soft = payloads.upload_bytes("ssfl", n_open=8900, num_classes=11, label_mode="soft2")
    assert soft > hard
```
