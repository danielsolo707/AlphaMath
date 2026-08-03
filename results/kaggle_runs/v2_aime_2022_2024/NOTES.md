# Frozen Kaggle run — v2_aime_2022_2024

```json
{
  "run_id": "v2_aime_2022_2024",
  "slug": "danielsolo1770/alpha-math-real-model-evaluation",
  "frozen_at_utc": "2026-08-03T05:38:37.670565+00:00",
  "evaluation_summary": {
    "total": 90,
    "metrics": {
      "accuracy": 0.1333,
      "solved_rate": 0.6333,
      "execution_success_rate": 0.6111,
      "default_rate": 0.3667,
      "avg_attempts": 4.922,
      "avg_latency_s": 304.717,
      "median_latency_s": 329.955,
      "max_latency_s": 370.988,
      "mean_vote_agreement": 0.9123,
      "sandbox_timeouts": 14
    },
    "dataset_tier": "external_labeled"
  }
}
```

# ALPHA-MATH Kaggle Run - Final Report

**Backend:** `transformers`  
**Model:** `/kaggle/input/datasets/mehedi457/qwen25-math-7b-instruct`  
**Preflight:** PASS  
**Dependency bootstrap:** `{"installed": true, "version": "0.49.2", "network_used": true}`  

## Labeled evaluation

- Problems: 90
- Accuracy: 13.33%
- Execution success: 61.11%
- Mean vote agreement: 91.23%
- Average latency: 304.717 seconds/problem
- Detailed report: `evaluation/REPORT.md`

## Evidence policy

Only metrics produced with `backend=transformers` and a labeled external or bundled dataset are model results. Mock results validate plumbing only. Keep `run_manifest.json` beside any number copied into the public README.

## Included artifacts

- `run_manifest.json`: resolved config, environment, GPU, package versions, preflight
- `evaluation/`: full JSON traces, flat CSV, and Markdown report
- `submission.csv`: checkpointed predictions when competition data was present
- `submission_trace.json`: per-problem latency, votes, repairs, and failures
