# ALPHA-MATH Evaluation Report

**Evidence level:** REAL MODEL EVALUATION
**Model:** `/kaggle/input/datasets/mehedi457/qwen25-math-7b-instruct`  
**Backend:** `transformers`  
**Dataset:** `/kaggle/working/alphamath_source/AlphaMath/data/sample_problems.json`  
**Dataset SHA-256:** `a6d6d99c686174abc878b69c7a0a4c39e9f68744575fe07564fa577ca67cfb5f`  
**Dataset tier:** `bundled_sanity`  
**Problems:** 10

## Headline metrics

| Metric | Value |
|---|---:|
| Accuracy | 90.00% |
| Execution success | 90.00% |
| Mean vote agreement | 96.30% |
| Average attempts | 3.1 |
| Average latency | 83.987 s |
| Sandbox timeouts | 0 |

## Per-problem results

| ID | Correct | Gold | Prediction | Attempts | Time (s) | Agreement |
|---|:---:|---:|---:|---:|---:|---:|
| demo_01 | yes | 2 | 2 | 2 | 32.548 | 100.00% |
| demo_02 | yes | 820 | 820 | 3 | 76.397 | 66.67% |
| demo_03 | yes | 24 | 24 | 3 | 112.49 | 100.00% |
| demo_04 | yes | 807 | 807 | 2 | 46.818 | 100.00% |
| demo_05 | yes | 56 | 56 | 2 | 62.578 | 100.00% |
| demo_06 | yes | 7 | 7 | 2 | 65.238 | 100.00% |
| demo_07 | yes | 21 | 21 | 2 | 26.702 | 100.00% |
| demo_08 | yes | 40 | 40 | 5 | 202.128 | 100.00% |
| demo_09 | yes | 385 | 385 | 2 | 34.795 | 100.00% |
| demo_10 | no | 24 | 0 | 8 | 180.18 | 0.00% |

## Accuracy by difficulty

| Difficulty | Correct | Total | Accuracy |
|---|---:|---:|---:|
| easy | 3 | 3 | 100.00% |
| medium | 6 | 7 | 85.71% |

## Execution failure taxonomy

| Failure | Count |
|---|---:|
| IndentationError | 8 |
| SyntaxError | 5 |

## Reproducibility

- Python: `3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]`
- Platform: `Linux-6.12.90+-x86_64-with-glibc2.35`
- GPU: `Tesla T4`
- GPU memory: `14.56 GiB`
- Torch / Transformers: `2.10.0+cu128` / `5.0.0`

The accompanying `run_manifest.json` contains the full resolved config, package versions, hardware information, Git commit, seed, and preflight checks. `per_problem.csv` is suitable for analysis, while `evaluation.json` retains complete correction traces.
