---
unit: ts1
framework: pytest
---
# E2E Test Stubs — Aggregate smoke suite

## Setup

```python
# tests/e2e/test_smoke_gate.py
import subprocess
import sys
import time

import pytest
```

## Stubs

### smoke-gate-passes-fast (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_smoke_gate_passes_fast():
    start = time.monotonic()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "smoke", "-p", "no:cacheprovider", "-q"],
        capture_output=True, text=True, timeout=330,
    )
    elapsed = time.monotonic() - start
    assert r.returncode == 0, r.stdout + r.stderr
    assert elapsed < 300
    # every method micro-run executed
    for name in ["fl", "fd", "dsfl", "ssfl"]:
        assert name in r.stdout.lower()
```
