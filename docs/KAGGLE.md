# Kaggle / AIMO offline runbook

ALPHA-MATH is built for **AIMO Progress Prize** constraints:

| Constraint | How we satisfy it |
|------------|-------------------|
| No external LLM APIs | Local **DeepSeek-Math-7B-Instruct** via Transformers |
| No internet at score time | `local_files_only: true` + weights attached as input |
| GPU notebook | 4-bit load (`bitsandbytes`) fits **T4 16GB** |
| Integer answers | Sandbox `ANSWER` + optional majority vote |

## Model

Primary open-source math model:

```text
deepseek-ai/deepseek-math-7b-instruct
```

Why this model: trained specifically for mathematical reasoning and
**tool-integrated** solving (generate code → execute), which matches the
ALPHA-MATH System-2 loop.

Alternatives that drop in with the same backend (change `llm.model` /
folder):

- `Qwen/Qwen2.5-Math-7B-Instruct`
- other chat-tuned math 7B-class checkpoints

## One-time: get weights offline

### Local machine

```bash
pip install -r requirements-gpu.txt
python scripts/download_deepseek_math.py
# → models/deepseek-math-7b-instruct/
```

### Kaggle

1. Download weights locally **or** use a public Kaggle Model/Dataset mirror of DeepSeek-Math-7B-Instruct.
2. Add that Dataset/Model to your competition notebook.
3. Set `model_path` in `configs/kaggle.yaml` to the folder that contains `config.json`.
4. **Turn Internet OFF** before final submit.

## Run submission

See `notebooks/kaggle_aimo_deepseek_math.ipynb`, or:

```python
from src.kaggle_submit import run_submission
run_submission(
    config_path="configs/kaggle.yaml",
    test_csv="/kaggle/input/.../test.csv",
    out_csv="/kaggle/working/submission.csv",
    model_path="/kaggle/input/deepseek-math-7b-instruct",
)
```

## Config knobs (`configs/kaggle.yaml`)

| Key | Meaning |
|-----|---------|
| `llm.model_path` | Local weights directory |
| `llm.local_files_only` | Must be `true` on Kaggle |
| `llm.load_in_4bit` | NF4 quantization for T4 |
| `agent.majority_vote_k` | Independent samples → majority answer |
| `agent.max_attempts` | Sandbox self-corrections per sample |

## CPU pipeline test (no GPU)

```bash
python scripts/run_eval.py --config configs/smoke_mock.yaml
```

This only checks the agent/sandbox plumbing; it does **not** load DeepSeek-Math.
