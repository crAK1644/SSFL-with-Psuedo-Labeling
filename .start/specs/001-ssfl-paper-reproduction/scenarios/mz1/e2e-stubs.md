---
unit: mz1
framework: pytest
---
# E2E Test Stubs — Model zoo

## Setup

```python
# tests/e2e/test_model_zoo.py
import pytest
import torch
```

## Stubs

### cnn-shapes-match-table1 (P0)

```python
@pytest.mark.skip(reason="evaluation phase")
def test_cnn_shapes_match_table1():
    from ssfl import models
    x = torch.randn(80, 23, 5)
    shapes = []
    clf = models.build("cnn", num_classes=11)
    hooks = [m.register_forward_hook(lambda _m, _i, o: shapes.append(tuple(o.shape)))
             for m in clf.modules() if len(list(m.children())) == 0]
    out = clf(x)
    assert out.shape == (80, 11)
    assert (80, 64, 5) in shapes and (80, 128, 5) in shapes
    assert (80, 128, 3) in shapes and (80, 128, 2) in shapes
    assert (80, 128) in shapes
    disc = models.build("cnn", num_classes=2)
    assert disc(x).shape == (80, 2)
    for h in hooks: h.remove()
```

### all-models-train-on-device (P1)

```python
@pytest.mark.skip(reason="evaluation phase")
@pytest.mark.parametrize("name", ["cnn", "mlp", "lstm"])
@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_models_train_on_device(name, device):
    from ssfl import models
    dev = models.resolve_device(device)
    if device == "auto":
        assert dev.type in ("mps", "cuda")
    model = models.build(name, num_classes=11).to(dev)
    x = torch.randn(80, 23, 5, device=dev)
    y = torch.randint(0, 11, (80,), device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward(); opt.step()
    assert torch.isfinite(loss)
```
