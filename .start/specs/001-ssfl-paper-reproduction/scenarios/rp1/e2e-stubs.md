---
unit: rp1
framework: pytest
---
# E2E Test Stubs — Report generation

## Setup

```python
# tests/e2e/test_report.py
import subprocess
import sys
from pathlib import Path

import pytest

def _make_synthetic_results(root: Path, complete: bool = True):
    """Build a plausible results tree from the campaign plan (helper to implement)."""
    from ssfl.campaign import plan_runs
    ...
```

## Stubs

### report-renders-full-tree (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_report_renders_full_tree(tmp_path):
    _make_synthetic_results(tmp_path / "results")
    r = subprocess.run([sys.executable, "-m", "ssfl.report",
                        "--results", str(tmp_path / "results"),
                        "--out", str(tmp_path / "report")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    out = tmp_path / "report"
    for t in ["table2", "table3", "table4"]:
        assert (out / f"{t}.md").exists() and (out / f"{t}.csv").exists()
    assert len(list(out.glob("fig3*.png"))) == 3
    for fig in ["fig4", "fig5", "fig6"]:
        assert list(out.glob(f"{fig}*.png"))
    table2 = (out / "table2.md").read_text()
    assert "87.40" in table2  # paper value shown alongside ours
```

### report-tolerates-missing-runs (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_report_tolerates_missing_runs(tmp_path):
    _make_synthetic_results(tmp_path / "results", complete=False)
    r = subprocess.run([sys.executable, "-m", "ssfl.report",
                        "--results", str(tmp_path / "results"),
                        "--out", str(tmp_path / "report")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    header = (tmp_path / "report" / "table2.md").read_text()
    assert "missing" in (r.stdout + header).lower()
```
