# AIME benchmark (local copy)

| File | Role |
|------|------|
| `aime_2022_2024.json` | **90** problems — ALPHA-MATH eval format |
| `aime_2022_2024.csv` | Same rows as CSV |
| `DATASET_CARD.json` | Provenance metadata |

**Source:** HuggingFace [`AI-MO/aimo-validation-aime`](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)  
**Years (from AoPS URLs):** 2022, 2023, 2024  
**Answers:** integers in **[0, 999]** (AIME style)

Rebuild:

```bash
python scripts/prepare_aime_benchmark.py
```

Kaggle upload payload: `kaggle/aime_benchmark_dataset/`  
Dataset slug: `danielsolo1770/alphamath-aime-benchmark`
