# Frozen run — v1 real Qwen sample-10 (Kaggle T4)

**Do not overwrite.** This is the first **real-model** (Transformers) labeled evaluation frozen into the portfolio repo.

## Headline

| Field | Value |
|-------|--------|
| Accuracy | **90% (9 / 10)** |
| Execution success | 90% |
| Mean vote agreement | ~96.3% |
| Avg latency | ~84 s / problem |
| Backend | `transformers` (not mock) |
| Model | Qwen2.5-Math-7B-Instruct |
| Weights path (Kaggle) | `/kaggle/input/datasets/mehedi457/qwen25-math-7b-instruct` |
| Load | `float16` + **4-bit (NF4 / bitsandbytes)** on Tesla T4 |
| Dataset | bundled `data/sample_problems.json` (10 easy/medium demos) |
| Dataset tier | `bundled_sanity` |
| GPU | NVIDIA Tesla T4 (~14.6 GiB) |
| Regression tests on kernel | passed |
| Preflight | PASS |

## What this proves

- Offline end-to-end agent works on real open weights (no external LLM API).
- Sandbox + stateful repair + majority voting produce integer answers on a sanity set.
- Pipeline emits audit artifacts (manifest, JSON traces, CSV, Markdown).

## What this does **not** prove

- AIMO competition leaderboard placement  
- Large public olympiad benchmark performance  
- That 90% generalizes beyond these **10 demo** problems  

One failure: `demo_10` (gold 24, pred default 0) after repeated Syntax/Indentation errors in generated code.

## Files

| Path | Role |
|------|------|
| `alphamath_artifacts.zip` | Original Kaggle download |
| `artifacts/FINAL_REPORT.md` | Run summary |
| `artifacts/run_manifest.json` | Config + env + packages + GPU |
| `artifacts/preflight.json` | Preflight checks |
| `artifacts/evaluation/` | Full metrics + per-problem CSV + traces |
| `artifacts/regression_tests.log` | unittest log from kernel |

## Policy

- New real runs → `results/kaggle_runs/v2_.../` (new folder).  
- Only promote a stronger public claim when dataset tier and size justify it.  
