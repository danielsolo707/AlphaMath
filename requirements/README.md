# Requirements

| File | Use |
|------|-----|
| [`core.txt`](./core.txt) | CPU agent, sandbox, eval, reports |
| [`dev.txt`](./dev.txt) | core + pytest / ruff / mypy |
| [`gpu.txt`](./gpu.txt) | core + PyTorch / Transformers stack |
| [`quantization.txt`](./quantization.txt) | gpu + bitsandbytes (Linux 4-bit) |

From repo root:

```bash
pip install -r requirements/core.txt
pip install -r requirements/dev.txt
pip install -r requirements/gpu.txt
pip install -r requirements/quantization.txt
```

Or the root shortcut (same as core):

```bash
pip install -r requirements.txt
```
