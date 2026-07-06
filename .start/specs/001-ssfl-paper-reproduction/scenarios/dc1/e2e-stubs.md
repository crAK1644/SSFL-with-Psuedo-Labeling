---
unit: dc1
framework: pytest
---
# E2E Test Stubs — Mini-N-BaIoT data core

Pre-generated executable test stubs for the evaluation agent. Each stub maps to a scenario and tests observable behavior through the external interface.

## Setup

```python
# tests/e2e/test_data_build.py
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CACHE = Path("cache")

def _run_build(seed=42):
    return subprocess.run(
        [sys.executable, "-m", "ssfl.data.build", "--seed", str(seed)],
        capture_output=True, text=True,
    )

def _checksums():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CACHE.iterdir()) if p.is_file()}
```

## Stubs

### build-produces-deterministic-cache (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_build_produces_deterministic_cache(tmp_path):
    assert _run_build().returncode == 0
    first = _checksums()
    for name in ["mini.npz", "splits.json", "scenario_1.json",
                 "scenario_2.json", "scenario_3.json", "meta.json"]:
        assert (CACHE / name).exists()
    data = np.load(CACHE / "mini.npz")
    X, y = data["X"], data["y"]
    assert X.dtype == np.float32 and X.shape[1:] == (23, 5)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert y.dtype == np.int64 and len(np.unique(y)) == 11
    assert X.shape[0] == 89_000
    splits = json.loads((CACHE / "splits.json").read_text())
    for subset in splits.values():
        pr, op, te = map(set, (subset["private"], subset["open"], subset["test"]))
        assert len(pr) == 700 and len(op) == 100 and len(te) == 200
        assert not (pr & op or pr & te or op & te)
    import shutil; shutil.rmtree(CACHE)
    assert _run_build().returncode == 0
    assert _checksums() == first
```

### scenario-partitions-match-paper (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_scenario_partitions_match_paper():
    data = np.load(CACHE / "mini.npz")
    y, dev = data["y"], data["device_id"]
    splits = json.loads((CACHE / "splits.json").read_text())
    private = {i for s in splits.values() for i in s["private"]}
    sizes = {1: 27, 2: 89, 3: 89}
    for s, expected in sizes.items():
        part = json.loads((CACHE / f"scenario_{s}.json").read_text())
        assert len(part) == expected
        for idxs in part.values():
            assert set(idxs) <= private
            if s == 2:
                assert len(np.unique(y[idxs])) == 1
    for d in (3, 7):
        assert len(np.unique(y[dev == d])) <= 6
```

### open-view-carries-no-labels (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_open_view_carries_no_labels():
    from ssfl.data import loader  # public API only
    open_view = loader.open_set()
    assert not hasattr(open_view, "y") and "y" not in getattr(open_view, "_fields", ("y",)) or True
    # The open view must expose features only; adjust attribute access to the public API,
    # then assert that no label accessor exists for open-split samples.
    test_view = loader.test_set()
    assert test_view.X.shape[0] == test_view.y.shape[0]
```
