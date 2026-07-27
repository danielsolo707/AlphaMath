# ALPHA-MATH — Design Notes

This document summarizes the research roadmap and what is **implemented in this repository today**, aligned with the published Kaggle kernel [`danielsolo1770/alpha-math`](https://www.kaggle.com/code/danielsolo1770/alpha-math).

## Vision

Build a lightweight, high-reliability agent for **IMO / AIME-style** problems by combining:

1. **Logical reasoning** from a math-specialized open language model  
2. **Mathematical precision** via Python + SymPy execution  
3. **Self-correction** through a verifier loop (System 2)  
4. **Self-consistency** via majority vote over independent samples  

Target competition framing:  
[AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)

## What shipped on Kaggle

| Piece | Kaggle notebook (v3) | This repo |
|-------|----------------------|-----------|
| Model | **Qwen2.5-Math-7B** via Kaggle Model `urvishp80/qwen-2.5-math-7b` | Same default (`llm.model` / `model_path`) |
| Precision | `float16`, `device_map=auto` | Same (`load_in_4bit: false`) |
| Samples | `num_generations=3` | `agent.majority_vote_k: 3` |
| Retries | `max_correction_attempts=2` | `agent.max_attempts: 2` |
| Sampling | `temperature=0.7`, `top_p=0.9` | Same |
| Prompt | Strict “Python generator” + sympy | `src/prompts.py` |
| Sandbox | Multiprocessing `exec` + timeout | AST-gated sandbox (stricter, safer) |
| Submission | `id,prediction` | Configurable (`paths.answer_column`) |
| Fail-safe | Return `0` if all paths fail | `default_answer_on_fail: 0` |

## Architecture (runtime)

```
Problem text
    │
    ▼
Qwen2.5-Math-7B  (Transformers, local weights, chat template)
    │  Python code in ```python``` fence
    ▼
AST-gated sandbox  (math, sympy, … · timeout · no OS / network)
    │
    ▼
integer from stdout / ANSWER  ──► retry on error  ──► majority vote (k=3)
    │
    ▼
submission.csv  (id, prediction)
```

### Backends

| `llm.backend` | Use case |
|---------------|----------|
| `transformers` / `qwen_math` (**default**) | Open-weight math model on local/Kaggle GPU |
| `mock` | CPU smoke test of the tool loop only |
| `openai` / `openai_compatible` | Optional servers (not for AIMO submit) |
| `anthropic` | Optional cloud (not for AIMO submit) |

## Roadmap (research beyond the shipped agent)

| Phase | Goal | Status |
|------:|------|--------|
| 1 | Logic-enriched student via teacher distillation | Not implemented (needs multi-GPU) |
| 2 | QLoRA specialization on olympiad CoT + tool traces | Scaffold only (`docs/TRAINING.md`) |
| 3 | Sandbox + generate→execute→verify + majority vote | **Implemented** (Kaggle-aligned) |
| 4 | Quantization / pruning for tighter VRAM | Optional 4-bit flag only |
| 5 | Portfolio release, eval harness, documentation | **Implemented** |

## Honesty policy

- Default `mock` accuracy on `data/sample_problems.json` measures the **tool loop**, not olympiad intelligence.
- No fabricated Kaggle leaderboard score is claimed.
- Distillation / QLoRA weights are **not** included unless separately trained and released.
- The original notebook used unrestricted `exec` in a subprocess; this repo replaces that with an **AST-gated** sandbox for safer demos.

## Security note

The sandbox blocks imports (rewrites allowed math imports), dunder attributes, and common OS primitives, and enforces a wall-clock timeout. It is **not** a multi-tenant security boundary. Untrusted code should run in a container / microVM with no network.
