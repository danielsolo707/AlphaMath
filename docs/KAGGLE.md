# Kaggle / AIMO offline runbook

ALPHA-MATH is built for **AIMO Progress Prize** constraints and matches the
published kernel [`danielsolo1770/alpha-math`](https://www.kaggle.com/code/danielsolo1770/alpha-math).

| Constraint | How we satisfy it |
|------------|-------------------|
| No external LLM APIs | Local **Qwen2.5-Math-7B** via Transformers |
| No internet at score time | `local_files_only: true` + weights attached as input |
| GPU notebook | `float16` + `device_map=auto` on **T4 16GB** (optional 4-bit) |
| Integer answers | Sandbox stdout / `ANSWER` + majority vote |

## Model (published run)

Kaggle Model data source attached to the notebook:

```text
urvishp80/qwen-2.5-math-7b/Transformers/default/1
```

Resolved on disk as:

```text
/kaggle/input/models/urvishp80/qwen-2.5-math-7b/transformers/default/1
```

Hub equivalent for local download:

```text
Qwen/Qwen2.5-Math-7B-Instruct
```

(or `Qwen/Qwen2.5-Math-7B` base — use the instruct/chat checkpoint when available)

### Why Qwen2.5-Math

Math-specialized open weights with a strong tool-integrated solving pattern
(generate code → execute). Drop-in alternatives via the same backend:

- `deepseek-ai/deepseek-math-7b-instruct`
- other chat-tuned math 7B-class checkpoints with `apply_chat_template`

## Inference recipe (from notebook)

| Knob | Value |
|------|------:|
| `num_generations` / `majority_vote_k` | 3 |
| `max_correction_attempts` / `max_attempts` | 2 |
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `max_new_tokens` | 1024 |
| `torch_dtype` | float16 |
| Fail-safe answer | 0 |

## One-time: get weights offline

### Local machine

```bash
pip install -r requirements-gpu.txt
python scripts/download_math_model.py
# → models/qwen2.5-math-7b-instruct/
```

### Kaggle

1. Add the Qwen2.5-Math Kaggle Model (or upload a Dataset of weights).
2. Set `model_path` in `configs/kaggle.yaml` to the folder with `config.json`.
3. **Turn Internet OFF** before final submit.

## Run submission

See `notebooks/kaggle_aimo_deepseek_math.ipynb`, or:

```python
from src.kaggle_submit import run_submission
run_submission(
    config_path="configs/kaggle.yaml",
    test_csv="/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",
    out_csv="/kaggle/working/submission.csv",
    model_path="/kaggle/input/models/urvishp80/qwen-2.5-math-7b/transformers/default/1",
)
```

Submission columns default to **`id,prediction`** (notebook format). If the
competition grader expects `answer`, set:

```yaml
paths:
  answer_column: answer
```

## Config knobs (`configs/kaggle.yaml`)

| Key | Meaning |
|-----|---------|
| `llm.model_path` | Local weights directory |
| `llm.local_files_only` | Must be `true` on Kaggle |
| `llm.load_in_4bit` | Optional NF4 if VRAM is tight |
| `agent.majority_vote_k` | Independent samples → majority answer |
| `agent.max_attempts` | Sandbox self-corrections per sample |
| `agent.temperature` / `top_p` | Sampling for diversity |
| `paths.answer_column` | `prediction` or `answer` |

## CPU pipeline test (no GPU)

```bash
python scripts/run_eval.py --config configs/smoke_mock.yaml
```

This only checks the agent/sandbox plumbing; it does **not** load Qwen2.5-Math.
