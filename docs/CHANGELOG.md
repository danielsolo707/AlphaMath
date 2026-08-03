# Changelog

## 0.3.0 - AIME labeled benchmark + reliability

- Added AIME 2022–2024 validation set (90 problems) under `data/benchmarks/aime/`
  and Kaggle dataset `danielsolo1770/alphamath-aime-benchmark`.
- Kernel evaluates full attached AIME set (no silent 10-problem cap).
- GPU safety: refuse unsupported arches (e.g. P100 sm_60) instead of emitting
  default-zero scores; auto-disable bitsandbytes on old GPUs; CUDA generate probe.
- Fixed fenced code extraction so shared leading indent is not stripped from the
  first line only (major IndentationError source).
- **AIME v2 freeze (Kaggle T4):** **13.33% (12/90)** external labeled —
  `results/kaggle_runs/v2_aime_2022_2024/`. Dominant failure was LaTeX dumps
  treated as Python; follow-up rejects LaTeX, requires ```python fences, and
  AST-prechecks before sandbox.
- AIME-friendly budgets: sandbox timeout 12s, per-problem time budget 420s,
  temperature 0.5.
- Smarter labeled-benchmark discovery prefers AIME / larger external sets.
- Scripts: `prepare_aime_benchmark.py`, `freeze_kaggle_output.py`, `analyze_eval.py`.
- Evaluation checkpoint/resume for multi-hour labeled runs (`checkpoint.json`).

## 0.2.2 - category folder layout

- Clean repo root: only README, LICENSE, pyproject, requirements entrypoint, env example.
- Moved dependency files → `requirements/`; changelog/security → `docs/`;
  interview notes → `docs/interview/`; added `models/` placeholder.
- Updated CI, Kaggle bundle allowlist, and install paths.

## 0.2.1 - real Kaggle evidence + repo cleanup

- Froze first **real Transformers** evaluation from Kaggle:
  - Run id: `v1_real_qwen_sample10`
  - Path: `results/kaggle_runs/v1_real_qwen_sample10/`
  - **Accuracy 90% (9/10)** on bundled sample problems (`bundled_sanity`)
  - Qwen2.5-Math-7B-Instruct, 4-bit load, Tesla T4
- Updated README, `results/pipeline_summary.json`, and results index to publish
  that number **with explicit limits** (not a leaderboard claim).
- Cleanup: removed `tmp/` bundle copies, nested `results/kaggle_runs/v1` source
  dumps, `__pycache__`, and loose zip clutter.

## 0.2.0 - local portfolio hardening

- Replaced thread timeout with killable process-isolated execution.
- Added source/output caps, actionable dependency failures, and Unix memory limits.
- Made correction turns stateful and clarified initial-attempt vs correction semantics.
- Added deterministic per-attempt seeds, majority metadata, and early stopping.
- Removed silent real-model-to-mock fallback.
- Added rich JSON/CSV/Markdown reports, manifests, dataset fingerprints, and ablations.
- Added preflight checks, submission checkpoint/resume, and schema auto-detection.
- Added Kaggle one-load experiment notebook and reproducible upload bundle builder.
- Added automated regression tests, CI, packaging metadata, and split dependencies.

## 0.1.0

- Initial modular Qwen2.5-Math inference pipeline and Kaggle submission driver.
