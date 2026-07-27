# Kaggle packaging

| Path | Role |
|------|------|
| `README_FIRST.md` | Short upload checklist |
| `kernel/` | Private GPU kernel entry + metadata |
| `runtime_dataset/` | Dataset payload for Kaggle CLI |
| `*.zip` | Generated bundles (gitignored; rebuild with script) |

```bash
python scripts/build_kaggle_bundle.py
```

Full guide: [`../docs/KAGGLE.md`](../docs/KAGGLE.md).
