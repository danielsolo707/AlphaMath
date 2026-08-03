# Frozen run — v2 AIME 2022–2024 (Kaggle T4)

**Do not overwrite.** First full **external_labeled** AIME evaluation frozen into the portfolio.

## Headline

| Field | Value |
|-------|--------|
| Accuracy | **13.33% (12 / 90)** |
| Execution success | 61.11% |
| Default-answer rate | 36.67% |
| Mean vote agreement | ~91.2% (on successful votes) |
| Avg latency | ~305 s / problem |
| Backend | `transformers` (not mock) |
| Model | Qwen2.5-Math-7B-Instruct |
| Load | 4-bit NF4 / bitsandbytes on Tesla T4 |
| Dataset | AI-MO/aimo-validation-aime → `aime_2022_2024.json` |
| Dataset tier | `external_labeled` |
| GPU | NVIDIA Tesla T4 (sm_75) |
| Kernel | `danielsolo1770/alpha-math-real-model-evaluation` v7 |

## What this proves

- End-to-end offline agent loop works on a real olympiad-style labeled set (90 problems).
- Pipeline produces full audit artifacts under Kaggle constraints.
- Failure modes are measurable (syntax/LaTeX dumps, timeouts, defaults).

## What this does **not** prove

- AIMO competition leaderboard placement  
- That 13% is an upper bound (post-run fixes target LaTeX-as-code extraction)  
- That the 90% sanity set generalizes  

## Dominant failure modes (attempt-level)

| Type | Count (approx) |
|------|---------------:|
| SyntaxError | 208 |
| IndentationError | 73 |
| NameError | 35 |
| TimeoutError | 14 |

Root cause observed in traces: the model frequently emitted **LaTeX equations** instead of fenced Python; the extractor previously treated `=` lines as code. Fixed in follow-up commits (reject LaTeX, require ```python fences, AST precheck).

## Correct problem ids

`aime_2022_000`, `aime_2022_007`, `aime_2022_011`, `aime_2022_016`, `aime_2022_028`,  
`aime_2023_042`, `aime_2023_045`, `aime_2023_052`, `aime_2023_055`,  
`aime_2024_064`, `aime_2024_069`, `aime_2024_079`

## Files

| Path | Role |
|------|------|
| `alphamath_artifacts.zip` | Original Kaggle download |
| `artifacts/FINAL_REPORT.md` | Run summary |
| `artifacts/evaluation/` | Full metrics + traces + CSV |
| `artifacts/run_manifest.json` | Config + env + packages + GPU |
