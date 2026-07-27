# Results

| Path | What it is |
|------|------------|
| [`pipeline_summary.json`](./pipeline_summary.json) | Machine-readable project status |
| [`sample_eval.json`](./sample_eval.json) | Older / sample eval stub |
| [`kaggle_runs/v1_real_qwen_sample10/`](./kaggle_runs/v1_real_qwen_sample10/) | **Frozen real-model Kaggle run (90% on 10 demos)** |
| `smoke_run/` | Local mock smoke (gitignored) — plumbing only |

Public README numbers must point at a frozen run with `backend=transformers` and a labeled dataset.
