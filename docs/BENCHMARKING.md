# Benchmarking protocol

## Minimum credible run

- A licensed, labeled held-out dataset (this repo uses **90** AIME 2022–2024
  validation problems from `AI-MO/aimo-validation-aime`; ≥100 is still preferred
  when available).
- Exact dataset name/version and selection rule.
- No prompt or answer tuning on the held-out slice.
- Fixed config and deterministic seed policy.
- Full run manifest and per-problem traces.
- Accuracy, execution success, latency, attempts, timeouts, and vote agreement.

## Recommended staged execution

1. Three-problem real-model dry run.
2. Full bundled sanity run (`data/sample_problems.json`).
3. External labeled benchmark (`data/benchmarks/aime/`) with `RUN_ABLATION=False`.
4. Same external benchmark with `RUN_ABLATION=True` if time permits.
5. Competition submission separately; do not infer accuracy from hidden labels.

## AIME protocol used here

| Field | Value |
|-------|--------|
| Source | HuggingFace `AI-MO/aimo-validation-aime` |
| Local path | `data/benchmarks/aime/aime_2022_2024.json` |
| Kaggle dataset | `danielsolo1770/alphamath-aime-benchmark` |
| n | 90 |
| Answer range | 0–999 (agent clamps with `% 1000`) |
| Selection | all train rows with integer answers in range |
| Rebuild | `python scripts/prepare_aime_benchmark.py` |

## Ablation interpretation

- `tool_repair - tool_pass1` estimates the contribution of execution feedback.
- `tool_repair_vote - tool_repair` estimates the contribution of additional
  stochastic samples and aggregation.
- Report both accuracy delta and compute/latency cost.
- A negative or zero delta is a valid result and should not be hidden.

## Leakage and licensing

Do not copy non-redistributable problems into this repository. Attach benchmark
data privately in Kaggle and retain its path/name in the manifest. Document any
filtering, answer normalization, or overlap checks used.
