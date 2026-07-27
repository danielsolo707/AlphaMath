# ALPHA-MATH

**Open-weight math agent for AIMO-style olympiad problems**

A **System-2** solver: a math-specialized language model **writes Python**, a **sandbox executes** it, failed attempts **self-repair**, and several independent samples take a **majority vote**. Built for the [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) setting: **Kaggle GPU, no external APIs, no internet at inference.**

The real experiment lives on Kaggle  
([`danielsolo1770/alpha-math`](https://www.kaggle.com/code/danielsolo1770/alpha-math)); this repo packages that pipeline as clean modules, CLIs, configs, and docs — not a notebook dump.

| | |
|---|---|
| **Math model (published run)** | **Qwen2.5-Math-7B** (Kaggle Model: `urvishp80/qwen-2.5-math-7b`) |
| **Inference** | Hugging Face Transformers · `float16` · `device_map=auto` |
| **Loop** | Generate code → execute → repair (×2) → majority of 3 samples |
| **Tools** | Restricted Python + SymPy sandbox |
| **Hardware** | Nvidia Tesla T4 (Kaggle) |
| **Competition** | Offline notebook · `local_files_only=True` |
| **Stack** | PyTorch · Transformers · SymPy · Accelerate |

---

## What is this project? (beginner-friendly)

Hard contest math problems often need **exact** answers (integers), not fuzzy chat.

1. **The model** (Qwen2.5-Math) reads the problem and emits a short **Python program**, usually using **SymPy** for exact algebra/number theory.
2. **The sandbox** runs that program with a time limit and without dangerous OS access.
3. If the code crashes or prints nothing useful, we **show the error back to the model** and ask it to fix the script (self-correction).
4. We do this for **several independent samples** and take the **most common integer** (self-consistency / majority vote).

That combination is sometimes called a **tool-integrated** or **System-2** math agent: language model for strategy, code execution for precision.

```
Problem
   │
   ▼
Qwen2.5-Math-7B   (local weights, Kaggle GPU)
   │  ```python ... print(answer)```
   ▼
Sandbox  (math · sympy · timeout · no OS / network)
   │
   ▼
integer  →  retry on failure  →  majority vote (k=3)
   │
   ▼
submission.csv   (id, prediction)
```

No OpenAI / Anthropic keys. No outbound HTTP during scoring.

---

## Problem formulation

| Item | Detail |
|------|--------|
| **Task** | Given a natural-language contest problem, predict a single **integer** answer |
| **Competition** | AIMO Progress Prize 3 (Kaggle) |
| **Constraints** | Offline GPU notebook; no proprietary APIs at submit time |
| **Evaluation (competition)** | Match hidden integer labels (leaderboard) |
| **Evaluation (this repo)** | Labeled demo problems + modular agent smoke tests |

This repository does **not** claim a public medal or fabricated LB score. It ships the **agent architecture** that was run on Kaggle, plus honest local demos.

---

## Dataset / approach

### Competition data

- Source: [AIMO Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) (`test.csv` with problem text + id).
- The notebook auto-discovers `test.csv` under `/kaggle/input` (skipping model folders).
- If no test file is attached (e.g. competition locked), the runner can fall back to **mock problems** so the pipeline still exercises end-to-end.

### Model choice

| Field | Published Kaggle value |
|-------|------------------------|
| Weights | Qwen2.5-Math-7B via Kaggle Model mirror |
| Path | `/kaggle/input/models/urvishp80/qwen-2.5-math-7b/transformers/default/1` |
| Load | `AutoTokenizer` + `AutoModelForCausalLM`, `float16`, `local_files_only=True` |
| Hub default (local) | `Qwen/Qwen2.5-Math-7B-Instruct` |

**Why not only chat?** Generic chat models are weak on contest precision. Math-specialized open models + **executable code** catch arithmetic and algebra mistakes the raw token stream would miss.

**Alternatives** (same backend): `deepseek-ai/deepseek-math-7b-instruct`, other chat-tuned math 7Bs with a chat template.

### Demo problems (portfolio)

[`data/sample_problems.json`](./data/sample_problems.json) — 10 original easy/medium integer problems for the **mock** backend (tool-loop unit tests). Perfect scores there measure plumbing, **not** olympiad intelligence.

---

## Architecture

| Module | Role |
|--------|------|
| [`src/local_model.py`](./src/local_model.py) | Transformers load + generate (`apply_chat_template`) |
| [`src/llm.py`](./src/llm.py) | Backend factory (`transformers` / `mock` / optional APIs) |
| [`src/prompts.py`](./src/prompts.py) | Strict “Python generator” system prompt (Kaggle-aligned) |
| [`src/agent.py`](./src/agent.py) | Generate → execute → repair → majority vote |
| [`src/sandbox.py`](./src/sandbox.py) | AST-gated execution + timeout (safer than raw `exec`) |
| [`src/kaggle_submit.py`](./src/kaggle_submit.py) | Discover test CSV, write `submission.csv` |
| [`src/evaluate.py`](./src/evaluate.py) | Batch accuracy on labeled JSON |
| [`src/solve.py`](./src/solve.py) | Single-problem CLI |

### Agent hyperparameters (published Kaggle recipe)

| Param | Value | Notebook name |
|-------|------:|---------------|
| Independent samples | **3** | `num_generations` |
| Self-repairs per sample | **2** | `max_correction_attempts` |
| Temperature | **0.7** | `temperature` |
| Top-p | **0.9** | `top_p` |
| Max new tokens | **1024** | `max_new_tokens` |
| Fail-safe answer | **0** | all paths failed |

Configs: [`configs/default.yaml`](./configs/default.yaml) · [`configs/kaggle.yaml`](./configs/kaggle.yaml) · [`configs/smoke_mock.yaml`](./configs/smoke_mock.yaml)

### Sandbox design note

The Kaggle notebook used **multiprocessing + unrestricted `exec`**. This repo keeps the same **timeout + integer parse + retry** behavior but adds:

- AST bans on `import` / `open` / `os` / `subprocess` / dunder tricks  
- Preloaded `math`, `sympy` (`sp`), etc., with import-line rewriting for model-emitted `import sympy`  

Safer for demos; still **not** a multi-tenant security boundary.

---

## Training recipe & results

### What actually ran on Kaggle

**Inference only** — no LoRA / full fine-tune in the published kernel.

| Item | Value |
|------|------:|
| Kernel | [`danielsolo1770/alpha-math`](https://www.kaggle.com/code/danielsolo1770/alpha-math) (v3) |
| GPU | Tesla T4 |
| Model | Qwen2.5-Math-7B (float16) |
| Loop | code gen + repair + majority vote (table above) |
| Submission format | `id,prediction` |
| Custom trained weights in repo | **No** |

Machine-readable summary: [`results/pipeline_summary.json`](./results/pipeline_summary.json)

Optional future specialization (QLoRA on tool traces) is documented in [`docs/TRAINING.md`](./docs/TRAINING.md) — **scaffold only**, not claimed as shipped weights.

### Local CPU smoke (tool loop)

```bash
python scripts/run_eval.py --config configs/smoke_mock.yaml
```

| Metric | Value |
|--------|------:|
| Backend | `mock` (template solver) |
| Demo set | 10 problems in `data/sample_problems.json` |
| Expected | **10/10** on mock templates |
| Artifact | [`results/sample_eval.json`](./results/sample_eval.json) |

> **Interview note:** mock 10/10 proves the agent/sandbox wiring. Real olympiad skill needs the Qwen (or DeepSeek-Math) weights on GPU.

---

## Repository layout

```
AlphaMath/
├── configs/
│   ├── default.yaml              # Qwen2.5-Math + tool loop
│   ├── kaggle.yaml               # offline competition profile
│   └── smoke_mock.yaml           # CPU pipeline test (no GPU)
├── data/sample_problems.json
├── docs/
│   ├── KAGGLE.md                 # offline runbook
│   ├── DESIGN.md                 # architecture + honesty
│   └── TRAINING.md               # optional specialization notes
├── notebooks/
│   └── kaggle_aimo_deepseek_math.ipynb   # thin Kaggle driver
├── scripts/
│   ├── download_math_model.py    # Hub → local weights (Qwen default)
│   ├── download_deepseek_math.py # alternate weights helper
│   ├── run_eval.py
│   └── run_solve.py
├── src/
│   ├── local_model.py
│   ├── llm.py
│   ├── agent.py
│   ├── sandbox.py
│   ├── kaggle_submit.py
│   ├── evaluate.py
│   ├── solve.py
│   └── prompts.py
├── results/
│   ├── pipeline_summary.json
│   └── sample_eval.json
├── requirements.txt
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

# CPU agent plumbing (mock eval works):
pip install -r requirements.txt

# Full GPU stack (Linux + CUDA recommended for 7B float16 / 4-bit):
pip install -r requirements-gpu.txt
```

### Download math weights (once, needs network)

```bash
python scripts/download_math_model.py
# → models/qwen2.5-math-7b-instruct/

# Alternate:
python scripts/download_deepseek_math.py
```

Then run fully offline:

```bash
python scripts/run_eval.py --model-path models/qwen2.5-math-7b-instruct --limit 3
python scripts/run_solve.py --model-path models/qwen2.5-math-7b-instruct \
  -p "What is gcd(252, 105)?"
```

---

## Usage

### CPU smoke test (no model download)

```bash
python scripts/run_eval.py --config configs/smoke_mock.yaml
# → 10/10 on bundled demos (mock templates, not Qwen)
```

### Solve one problem

```bash
python scripts/run_solve.py --config configs/smoke_mock.yaml \
  -p "What is gcd(252, 105)?"
```

### Kaggle submission (AIMO)

1. Attach competition data + **Qwen2.5-Math** weights as Model/Dataset.  
2. Open [`notebooks/kaggle_aimo_deepseek_math.ipynb`](./notebooks/kaggle_aimo_deepseek_math.ipynb).  
3. Point `MODEL_PATH` at the folder containing `config.json` (auto-detects the published path when present).  
4. **GPU on · Internet off** · Run All → `/kaggle/working/submission.csv`.

Details: [`docs/KAGGLE.md`](./docs/KAGGLE.md) · config: [`configs/kaggle.yaml`](./configs/kaggle.yaml)

```python
from src.kaggle_submit import run_submission
run_submission(
    config_path="configs/kaggle.yaml",
    test_csv="/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",
    out_csv="/kaggle/working/submission.csv",
    model_path="/kaggle/input/models/urvishp80/qwen-2.5-math-7b/transformers/default/1",
)
```

| Kaggle knob | Default | Role |
|-------------|---------|------|
| `llm.model_path` | attached Qwen folder | Offline weights |
| `llm.local_files_only` | `true` | No Hub calls |
| `llm.torch_dtype` | `float16` | Match notebook load |
| `agent.majority_vote_k` | `3` | Self-consistency |
| `agent.max_attempts` | `2` | Sandbox self-repair |
| `agent.temperature` | `0.7` | Sample diversity |
| `paths.answer_column` | `prediction` | Notebook CSV column |

If the grader requires `answer` instead of `prediction`, set `paths.answer_column: answer`.

---

## Design notes / limitations / next steps

### Design notes

- **Tool loop > pure CoT** for integer olympiad answers: one arithmetic slip ruins a free-form chain of thought; the sandbox catches many of them.  
- **Majority vote** trades GPU time for robustness when the model is stochastic (`temperature=0.7`).  
- **Strict code-only prompt** (from the Kaggle notebook) reduces rambling and makes parsing reliable.  
- **Modular backends** keep Kaggle offline while still allowing optional cloud LLMs for research.

### Limitations

- 7B float16 is near the edge of a 16GB T4; long contexts or large vote-`k` can OOM — enable `load_in_4bit: true` if needed.  
- AST sandbox is not a full security boundary.  
- No public LB score is claimed; competition test labels are hidden.  
- Mock eval ≠ model quality.  
- Optional QLoRA specialization is **not** implemented end-to-end in this repo.

### Next steps

1. Run full `test.csv` offline with Qwen weights and log per-problem latency / vote agreement.  
2. Try 4-bit load + higher `majority_vote_k` under the competition time budget.  
3. Curate tool-use traces where sandbox output matches gold → QLoRA (see `docs/TRAINING.md`).  
4. Add a small held-out integer set (AIME-style public problems) with real-model metrics for the README.

---

## Design honesty

| Claim | Status |
|-------|--------|
| Uses open-weight **Qwen2.5-Math** on the competition path | Yes (Kaggle kernel + configs + notebook) |
| Pipeline aligned with `danielsolo1770/alpha-math` | Yes (model, sampling, vote, retries, CSV column) |
| Offline / no external API | Yes (`local_files_only`, local `model_path`) |
| Tool loop + majority vote | Yes |
| Safer sandbox than notebook `exec` | Yes (AST-gated) |
| Weights stored in this git repo | **No** (download or attach on Kaggle) |
| Guaranteed AIMO medal / public LB score | **Not claimed** |
| Custom distilled student from 70B “logic surgery” | Research notes only (`docs/`) |

---

## Competition context

[AIMO Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
requires solving contest problems with **integer answers** under Kaggle’s
compute rules. Strong open solutions pair a **math-specialized open model**
with **code execution**. ALPHA-MATH is that stack, packaged for portfolio reuse
and aligned with the author’s Kaggle notebook.

---

## Author

**Daniel Soleimani** · [github.com/danielsolo707](https://github.com/danielsolo707)

---

## License

MIT — see [LICENSE](./LICENSE).  
Qwen / DeepSeek-Math weights are subject to their own licenses on Hugging Face / Kaggle.  
Bundled demo problems are original.
