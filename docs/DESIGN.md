# ALPHA-MATH design and invariants

## Objective

Produce auditable integer predictions for olympiad-style problems with a local
math model and executable exact computation. The system is optimized for
evidence quality under a one-GPU, offline Kaggle runtime—not for serving
untrusted public code.

## Agent state machine

For every vote round:

1. Send the original problem and code-only contract to the model.
2. Execute the generated program in a fresh restricted worker.
3. On failure, send the original problem, previous response/code, captured
   stdout, and error in a stateful correction conversation.
4. Stop after `max_corrections`, or add the first valid integer to the vote set.
5. Stop sampling when a strict majority of the planned `k` votes is reached.

`max_corrections=2` therefore means up to **three total generations** per vote:
one initial attempt and two corrections.

## Correctness invariants

- A vote exists only when restricted execution succeeds and yields an integer.
- Model text alone cannot vote; the tool output is authoritative.
- Default answer `0` is marked `defaulted=true` and never counted as a successful
  labeled evaluation.
- Every attempt records its vote round, correction index, deterministic seed,
  generated code, stdout, error, and execution result.
- Mock backend selection is explicit; real backends cannot silently fall back.
- Labeled reports preserve gold and normalized gold separately.

## Sandbox lifecycle

The parent launches `python -m src.sandbox --worker`, sends a JSON request, and
waits with a hard timeout. On timeout the OS process is killed. The child:

- rewrites allow-listed math imports;
- validates the resulting AST;
- exposes a minimal builtin set and selected math modules;
- captures output under a character cap;
- executes and coerces a final integer;
- returns one JSON protocol message.

This design trades roughly 0.1 seconds of process startup for deterministic
termination. Model generation dominates the runtime for 7B inference.

## Reproducibility

Seeds follow:

```text
base_seed + (vote_round - 1) * 1000 + correction_index
```

The run manifest records the resolved config, dataset path, package versions,
Python/platform, GPU model/memory, CUDA version, Git commit/dirty state, and
preflight results.

## Evidence levels

1. `mock_pipeline_only`: tests integration, never model quality.
2. `bundled_sanity`: real model on original easy/medium sanity problems.
3. `external_labeled`: real model on a named labeled dataset.
4. `competition_submission`: predictions without hidden accuracy.

Public claims must identify the evidence level and retain the manifest.

## Known boundaries

- The worker is not a container/microVM and should not accept arbitrary public
  user code.
- The per-problem time budget is checked between generations; a model generation
  itself is not preempted.
- Majority voting handles stochastic disagreement but does not prove correctness.
- No training/fine-tuned weights are claimed by the shipped inference system.
