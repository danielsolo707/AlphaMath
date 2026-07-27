# Changelog

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
