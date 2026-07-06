---
unit: rn1
framework: pytest
---
# E2E Test Stubs — Runner CLI and campaign

## Setup

```python
# tests/e2e/test_runner_cli.py
import json
import subprocess
import sys
from pathlib import Path

import pytest

def _run(*args):
    return subprocess.run([sys.executable, "-m", "ssfl.run", *args],
                          capture_output=True, text=True)
```

## Stubs

### cli-validates-and-launches (P0)

```python
@pytest.mark.skip(reason="evaluation phase; (d) is slow")
def test_cli_validates_and_launches():
    bad = _run("--method", "bogus")
    assert bad.returncode != 0 and "fl" in bad.stderr and "ssfl" in bad.stderr
    flag = _run("--method", "fl", "--no-voting")
    assert flag.returncode != 0
    ok = _run("--method", "fl", "--scenario", "1", "--seed", "0",
              "--rounds", "2", "--num-clients", "4")
    assert ok.returncode == 0
    assert "fl-cnn-s1-seed0" in ok.stdout and "results/" in ok.stdout
```

### campaign-skips-completed (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_campaign_skips_completed(tmp_path):
    from ssfl.campaign import plan_runs, pending_runs
    plan = plan_runs()
    assert len(plan) >= 28
    for cfg in plan[:3]:
        d = tmp_path / cfg.run_id(); d.mkdir(parents=True)
        (d / "final.json").write_text(json.dumps({"accuracy": 0.8}))
    crashed = tmp_path / plan[3].run_id(); crashed.mkdir()
    todo = pending_runs(plan, results_root=tmp_path)
    assert plan[0] not in todo and plan[3] in todo
    assert len(todo) == len(plan) - 3
```

### timing-projects-campaign (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_timing_projects_campaign():
    from ssfl.campaign import project_durations
    proj = project_durations(seconds_per_round=2.0, clients_measured=4)
    assert len(proj) >= 28
    assert all(p.projected_seconds > 0 for p in proj)
```
