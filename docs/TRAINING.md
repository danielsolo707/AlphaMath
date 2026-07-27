# Training / specialization notes

The **published Kaggle ALPHA-MATH run is inference-only**: it loads a public
math-specialized 7B checkpoint and runs a System-2 tool loop. No LoRA or
full fine-tune was trained in that notebook.

This file records an optional **research specialization recipe** so the
portfolio is complete as an architecture doc, without pretending custom
weights exist.

## What was run on Kaggle (actual)

| Item | Value |
|------|-------|
| Kernel | [`danielsolo1770/alpha-math`](https://www.kaggle.com/code/danielsolo1770/alpha-math) |
| Hardware | Nvidia Tesla T4 |
| Model | Qwen2.5-Math-7B (Kaggle Model mirror) |
| Training | **None** — offline inference + code execution |
| Loop | generate code → execute → self-repair (×2) → majority of 3 |

See [`results/pipeline_summary.json`](../results/pipeline_summary.json) and
[`docs/KAGGLE.md`](./KAGGLE.md).

## Optional specialization stack (not shipped)

| Component | Choice |
|-----------|--------|
| Student | Qwen2.5-Math / Llama-3 class 7–8B instruct |
| Teacher | Stronger open reasoner (70B-class or math-specialized) |
| PEFT | QLoRA (4-bit base, LoRA rank 16–64) |
| Objective | Supervised fine-tune on (problem → sympy script) |
| Libraries | `transformers`, `peft`, `bitsandbytes`, `trl` or `axolotl` |

## Data sources (examples)

- Public olympiad problem sets with worked solutions (respect licenses)
- Synthetic tool-use traces: model drafts SymPy code → sandbox filters valid traces
- Formal-adjacent corpora (optional Lean/Isabelle for future work)

## Suggested training objective

```
<input>  olympiad problem statement
<output>
```python
# exact sympy / integer solution
print(<integer>)
```
```

Filter training rows where sandbox execution yields the gold integer answer.

## Local inference after (optional) training

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

## Intentionally out of scope in this repo

- Shipping multi-GB checkpoints in git  
- Claiming AIMO medal scores without a public submission  
- Unreproducible “logic surgery” weight grafts without code + logs  
