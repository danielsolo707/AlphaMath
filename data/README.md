# Data

| Path | Role |
|------|------|
| `sample_problems.json` | 10 easy/medium demos used for the frozen 90% Kaggle sanity run (`bundled_sanity`) |
| `benchmark_template.json` | Schema example for larger external benchmarks |
| `benchmarks/aime/` | **AIME 2022–2024 validation set** (90 problems, answers in 0–999) |

## AIME benchmark

Source: HuggingFace [`AI-MO/aimo-validation-aime`](https://huggingface.co/datasets/AI-MO/aimo-validation-aime).

Rebuild locally:

```bash
python scripts/prepare_aime_benchmark.py
```

Kaggle upload payload: `kaggle/aime_benchmark_dataset/`  
Dataset slug: `danielsolo1770/alphamath-aime-benchmark`

Scores on this set are **labeled validation accuracy**, not an AoPS or AIMO leaderboard.

## Licensing note

Do not commit non-redistributable olympiad datasets. Prefer attaching private Kaggle datasets and recording the path/name in the run manifest.
