# Training notes (research scaffold)

Phase 1–2 from the master plan (logic distillation + QLoRA) require substantial
GPU time and curated datasets. This file records the intended recipe so the
portfolio is complete as an architecture doc, without pretending weights exist.

## Intended stack

| Component | Choice |
|-----------|--------|
| Student | Llama-3 / Qwen2.5 class 7–8B instruct |
| Teacher | Stronger open reasoner (70B-class or math-specialized) |
| PEFT | QLoRA (4-bit base, LoRA rank 16–64) |
| Objective | Supervised fine-tune on (problem → reasoning + Python solution) |
| Libraries | `transformers`, `peft`, `bitsandbytes`, `trl` or `axolotl` |

## Data sources (examples)

- Public olympiad problem sets with worked solutions (respect licenses)
- Synthetic tool-use traces: model drafts SymPy code → sandbox filters valid traces
- Formal-adjacent corpora (optional Lean/Isabelle for future work)

## Suggested training objective

```
<input>  olympiad problem statement
<output> REASONING: ...
         CODE:
         ```python
         ANSWER = ...
         print(ANSWER)
         ```
```

Filter training rows where sandbox execution yields the gold integer answer.

## Local inference after training

Serve the adapter with vLLM or Ollama (GGUF), then point ALPHA-MATH at it:

```yaml
# configs/local_vllm.yaml
llm:
  backend: openai_compatible
  base_url: http://localhost:8000/v1
  model: alphamath-8b
  api_key_env: LLM_API_KEY
```

```bash
export LLM_API_KEY=not-needed
python scripts/run_eval.py --backend openai --config configs/local_vllm.yaml
```

## What is intentionally out of scope here

- Shipping multi-GB checkpoints in git  
- Claiming AIMO medal scores without a public submission  
- Unreproducible “logic surgery” weight grafts without code + logs  
