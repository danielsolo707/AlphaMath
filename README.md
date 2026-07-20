# ALPHA-MATH

**Open-weight math agent for AIMO-style olympiad problems**  
**DeepSeek-Math-7B-Instruct** plans → **Python / SymPy sandbox** executes → verifier retries + majority vote.

Built for the [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) setting: **Kaggle GPU, no external APIs, no internet at inference.**

| | |
|---|---|
| **Math model** | [`deepseek-ai/deepseek-math-7b-instruct`](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct) (open source) |
| **Inference** | Hugging Face Transformers · optional 4-bit NF4 (T4 16GB) |
| **Tools** | Restricted Python + SymPy sandbox (System 2) |
| **Competition** | Offline Kaggle notebook · `local_files_only=True` |
| **Stack** | PyTorch · Transformers · SymPy · Accelerate |

---

## Why DeepSeek-Math

Generic chat models are weak on contest math precision. ALPHA-MATH uses a
**math-specialized open model** (DeepSeek-Math) as the planner and forces every
numeric step through executable code — the same tool-integrated pattern
DeepSeek-Math was trained for.

```
Problem
   │
   ▼
DeepSeek-Math-7B-Instruct   (local weights, Kaggle GPU)
   │  REASONING + Python
   ▼
AST sandbox  (math · sympy · … · timeout · no OS / network)
   │
   ▼
ANSWER ∈ ℤ   →  retry on failure  →  majority vote (k samples)
```

No OpenAI / Anthropic keys. No outbound HTTP during scoring.

---

## Repository layout

```
AlphaMath/
├── configs/
│   ├── default.yaml              # DeepSeek-Math + tool loop
│   ├── kaggle.yaml               # offline competition profile
│   └── smoke_mock.yaml           # CPU pipeline test (no GPU)
├── data/sample_problems.json
├── docs/
│   ├── KAGGLE.md                 # offline runbook
│   ├── DESIGN.md
│   └── TRAINING.md
├── notebooks/
│   └── kaggle_aimo_deepseek_math.ipynb
├── scripts/
│   ├── download_deepseek_math.py
│   ├── run_eval.py
│   └── run_solve.py
├── src/
│   ├── local_model.py            # Transformers DeepSeek-Math backend
│   ├── llm.py                    # backend factory
│   ├── agent.py                  # generate → execute → vote
│   ├── sandbox.py
│   ├── kaggle_submit.py          # writes submission.csv
│   ├── evaluate.py
│   ├── solve.py
│   └── prompts.py
└── requirements-gpu.txt
```

---

## Setup

```bash
git clone https://github.com/danielsolo707/AlphaMath.git
cd AlphaMath
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate

# CPU agent plumbing only:
pip install -r requirements.txt

# Full GPU stack (Linux + CUDA recommended for 4-bit):
pip install -r requirements-gpu.txt
```

### Download DeepSeek-Math (once, needs network)

```bash
python scripts/download_deepseek_math.py
# → models/deepseek-math-7b-instruct/
```

Then run fully offline:

```bash
python scripts/run_eval.py --model-path models/deepseek-math-7b-instruct --limit 3
python scripts/run_solve.py --model-path models/deepseek-math-7b-instruct \
  -p "What is gcd(252, 105)?"
```

---

## Kaggle submission (AIMO)

1. Attach competition data + **DeepSeek-Math weights** as Dataset/Model.  
2. Open [`notebooks/kaggle_aimo_deepseek_math.ipynb`](./notebooks/kaggle_aimo_deepseek_math.ipynb).  
3. Point `MODEL_PATH` at the folder containing `config.json`.  
4. **GPU on · Internet off** · Run All → `submission.csv`.

Details: [`docs/KAGGLE.md`](./docs/KAGGLE.md) · config: [`configs/kaggle.yaml`](./configs/kaggle.yaml)

```python
from src.kaggle_submit import run_submission
run_submission(
    config_path="configs/kaggle.yaml",
    test_csv="/kaggle/input/.../test.csv",
    out_csv="/kaggle/working/submission.csv",
    model_path="/kaggle/input/deepseek-math-7b-instruct",
)
```

| Kaggle knob | Default | Role |
|-------------|---------|------|
| `llm.model_path` | attached input dir | Offline weights |
| `llm.local_files_only` | `true` | No Hub calls |
| `llm.load_in_4bit` | `true` | Fit T4 16GB |
| `agent.majority_vote_k` | `5` | Self-consistency |
| `agent.max_attempts` | `3` | Sandbox self-repair |

---

## CPU smoke test (no model download)

Validates the agent/sandbox loop without GPU:

```bash
python scripts/run_eval.py --config configs/smoke_mock.yaml
# → 10/10 on bundled demos (mock templates, not DeepSeek-Math)
```

---

## Design honesty

| Claim | Status |
|-------|--------|
| Uses open-weight **DeepSeek-Math** for competition path | Yes (code + configs + notebook) |
| Offline / no external API | Yes (`local_files_only`, local `model_path`) |
| Tool loop + majority vote | Yes |
| Weights stored in this git repo | **No** (download or attach on Kaggle) |
| Guaranteed AIMO medal / public LB score | **Not claimed** |
| Custom distilled 8B from 70B “logic surgery” | Research notes only (`docs/`) |

---

## Competition context

[AIMO Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
requires solving contest problems with **integer answers** under Kaggle’s
compute rules. Strong open solutions pair a **math-specialized open model**
with **code execution**. ALPHA-MATH is that stack, packaged for portfolio and
Kaggle reuse.

---

## License

MIT — see [LICENSE](./LICENSE).  
DeepSeek-Math weights are subject to their own license on Hugging Face.  
Bundled demo problems are original.
