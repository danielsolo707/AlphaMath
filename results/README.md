# Results

| Path | What it is |
|------|------------|
| [`pipeline_summary.json`](./pipeline_summary.json) | Machine-readable project status |
| [`sample_eval.json`](./sample_eval.json) | Older / sample eval stub |
| [`kaggle_runs/v1_real_qwen_sample10/`](./kaggle_runs/v1_real_qwen_sample10/) | Frozen real-model sanity run — **90% on 10 demos** (`bundled_sanity`) |
| `kaggle_runs/v2_aime_*/` | Real-model AIME 2022–2024 labeled runs (`external_labeled`) when complete |
| `smoke_run/` | Local mock smoke (gitignored) — plumbing only |

Public README numbers must point at a frozen run with `backend=transformers` and a labeled dataset tier.
