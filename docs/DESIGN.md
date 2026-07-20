# ALPHA-MATH — Design Notes

This document summarizes the research roadmap (from the original project plan)
and what is **implemented in this repository today**.

## Vision

Build a lightweight, high-reliability agent for **IMO / AIME-style** problems by
combining:

1. **Logical reasoning** from strong language models  
2. **Mathematical precision** via Python + SymPy execution  
3. **Self-correction** through a verifier loop (System 2)

Target competition framing:  
[AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)

## Roadmap (research)

| Phase | Goal | Status in this repo |
|------:|------|---------------------|
| 1 | Logic-enriched 8B via teacher activation mapping / weight-space distillation | **Not implemented** (research — needs multi-GPU) |
| 2 | QLoRA specialization on formal + olympiad CoT data; train to emit SymPy scripts | **Scaffold only** (see `docs/TRAINING.md`) |
| 3 | Sandbox + generate→execute→verify loop | **Implemented** (`src/agent.py`, `src/sandbox.py`) |
| 4 | Task-specific pruning + quantization for local VRAM | **Not implemented** |
| 5 | Portfolio release, eval harness, documentation | **Implemented** |

## What ships now (Phase 3 engine)

```
Problem text
    │
    ▼
LLM / mock backend  ──►  REASONING + Python CODE
    │
    ▼
AST-gated sandbox (math, sympy, …)  ──►  ANSWER : int
    │
    ▼
Verifier  ──►  success | feedback → retry (≤ max_attempts)
```

### Backends

| `llm.backend` | Use case |
|---------------|----------|
| `mock` (default) | Offline portfolio demo on bundled sample problems |
| `openai` / `openai_compatible` | GPT, Ollama, vLLM, any Chat Completions server |
| `anthropic` | Claude |

### Honesty policy

- Default `mock` accuracy on `data/sample_problems.json` measures the **tool loop**,
  not olympiad intelligence.
- No fabricated Kaggle leaderboard score is claimed.
- Distillation / QLoRA weights are **not** included unless separately trained and released.

## Security note

The sandbox blocks imports, dunder attributes, and common OS primitives, and
enforces a wall-clock timeout. It is **not** a multi-tenant security boundary.
Untrusted code should run in a container / microVM with no network.
