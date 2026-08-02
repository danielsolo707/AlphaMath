# Kaggle packaging

| Path | Role |
|------|------|
| `README_FIRST.md` | Short upload checklist |
| `kernel/` | Private GPU kernel entry + metadata |
| `runtime_dataset/` | Source bundle payload for Kaggle CLI |
| `aime_benchmark_dataset/` | Labeled AIME 2022–2024 eval payload |
| `*.zip` | Generated bundles (gitignored; rebuild with script) |

```bash
python scripts/build_kaggle_bundle.py
kaggle datasets version -p kaggle/runtime_dataset -m "runtime refresh" -r zip
kaggle datasets version -p kaggle/aime_benchmark_dataset -m "aime refresh" -r zip
kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4 -t 32400
```

Kernel id: `danielsolo1770/alpha-math-real-model-evaluation`

Full guide: [`../docs/KAGGLE.md`](../docs/KAGGLE.md).
