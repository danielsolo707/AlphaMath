# Evidence-driven roadmap

This roadmap lists concrete, testable increments. It does not claim custom
weights, distillation, pruning, or benchmark improvements before artifacts exist.

## Shipped locally in 0.2

- Stateful code-repair conversations.
- Killable worker-process sandbox with resource caps.
- Reproducible vote/correction seeds and strict-majority early stopping.
- Rich evaluation, environment manifests, dataset fingerprints, and ablations.
- Kaggle one-load workflow, checkpoint/resume, and final evidence archive.
- Regression tests and CI.

## Evidence obtained (0.2.1)

1. **Done:** bundled sanity set on real Qwen2.5-Math-7B (Kaggle T4, 4-bit) —
   **90% (9/10)**, frozen at `results/kaggle_runs/v1_real_qwen_sample10/`.
2. **Done:** public README updated from that manifest (with explicit n=10 limit).

## Next: stronger evidence

1. **In progress / frozen under `results/kaggle_runs/v2_aime_*`:** AIME 2022–2024
   (90 labeled problems from `AI-MO/aimo-validation-aime`).
2. Enable ablation and report accuracy delta alongside latency/attempt cost.
3. Review failure traces (syntax/indent vs math) and iterate prompts/repair.
4. Only then consider a competition-style score claim with full traces.

## Candidate research increments

Each item requires a baseline and an ablation before being described as an
improvement:

- Direct-answer vs tool-integrated baseline.
- Adaptive compute: allocate extra votes only to low-agreement problems.
- Candidate verifier/reranker beyond frequency voting.
- 4-bit vs fp16 accuracy, latency, and memory comparison.
- Batching/prefix caching under the competition time budget.
- QLoRA on licensed, execution-verified tool traces.

## Explicitly not claimed

- Neuron/weight grafting from a teacher model.
- A custom logic-distilled model.
- Structured pruning that preserves math accuracy.
- Released specialized weights.
- IMO-level accuracy or a Kaggle medal.

Those claims become appropriate only after code, weights, protocol, and measured
results are independently reproducible.
