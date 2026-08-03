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
