# Scripts

| Script | Role |
|--------|------|
| `run_solve.py` | Solve one problem |
| `run_eval.py` | Labeled evaluation |
| `run_preflight.py` | Environment / dependency checks |
| `run_kaggle_experiment.py` | One-load experiment driver |
| `build_kaggle_bundle.py` | Build Kaggle upload ZIPs |
| `download_math_model.py` | Download Qwen2.5-Math weights |
| `download_deepseek_math.py` | Legacy DeepSeek download helper |

Examples from repo root:

```bash
python scripts/run_preflight.py --config configs/smoke_mock.yaml
python scripts/run_eval.py --config configs/smoke_mock.yaml
```
