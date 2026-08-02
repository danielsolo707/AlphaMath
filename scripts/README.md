# Scripts

| Script | Role |
|--------|------|
| `run_solve.py` | Solve one problem |
| `run_eval.py` | Labeled evaluation |
| `run_preflight.py` | Environment / dependency checks |
| `run_kaggle_experiment.py` | One-load experiment driver |
| `build_kaggle_bundle.py` | Build Kaggle upload ZIPs |
| `prepare_aime_benchmark.py` | Download/convert AIME validation set |
| `freeze_kaggle_output.py` | Download kernel output → `results/kaggle_runs/` |
| `analyze_eval.py` | Summarize evaluation.json failures / accuracy |
| `download_math_model.py` | Download Qwen2.5-Math weights |
| `download_deepseek_math.py` | Legacy DeepSeek download helper |

Examples from repo root:

```bash
python scripts/run_preflight.py --config configs/smoke_mock.yaml
python scripts/run_eval.py --config configs/smoke_mock.yaml
```
