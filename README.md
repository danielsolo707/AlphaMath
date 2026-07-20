# ALPHA-MATH

**System-2 mathematical reasoning agent** for AIME / olympiad-style problems:  
language-model planning → **Python + SymPy execution** → verifier retries.

| | |
|---|---|
| **Competition framing** | [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) |
| **Core idea** | Tool-using agent (code as the calculator) with self-correction |
| **Stack** | Python · SymPy · pluggable LLM backends · AST sandbox |
| **Status** | Phase-3 engine **implemented** · distillation / QLoRA **research** (not claimed as done) |
| **Offline demo** | `mock` backend · **10/10** on bundled sample set |

---

## Why this project

Closed-book LLMs still drop integer precision on contest math. ALPHA-MATH treats
the model as a **planner** and forces computation through a restricted Python
sandbox (System 2). That is the same architecture used by strong open AIMO
pipelines: generate → execute → check → retry.

This repository packages a clean, portfolio-ready implementation of that loop,
plus an honest research roadmap for logic distillation and QLoRA specialization.

---

## Architecture

```
Problem statement
        │
        ▼
 ┌──────────────┐     REASONING + Python CODE
 │  LLM backend │ ──────────────────────────►
 │ mock/openai/ │
 │  anthropic   │
 └──────────────┘
        │
        ▼
 ┌──────────────────────────────────────────┐
 │  AST-gated sandbox                       │
 │  allowed: math, sympy, itertools, …      │
 │  timeout + no imports / OS access        │
 └──────────────────────────────────────────┘
        │
        ▼
   ANSWER : int  ──►  ok? ── yes ──► return
                 └── no  ──► feedback → retry (≤ N)
```

**Answer convention:** AIMO / AIME-style non-negative integers (often 0–999).

---

## Repository layout

```
AlphaMath/
├── configs/default.yaml      # agent, sandbox, LLM settings
├── data/sample_problems.json # 10 labeled demo problems
├── docs/
│   ├── DESIGN.md             # roadmap vs implemented
│   └── TRAINING.md           # QLoRA / distillation notes
├── results/sample_eval.json  # offline eval summary (generated)
├── scripts/
│   ├── run_eval.py
│   └── run_solve.py
├── src/
│   ├── agent.py              # generate → execute → verify loop
│   ├── sandbox.py            # restricted Python executor
│   ├── llm.py                # mock / OpenAI-compatible / Anthropic
│   ├── prompts.py
│   ├── evaluate.py
│   ├── solve.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## Results (offline demo)

Default backend is **`mock`**: deterministic solution templates for the bundled
problems so the **pipeline** is fully runnable without API keys or GPUs.

| Split | Problems | Metric |
|-------|----------:|--------|
| Bundled sample set | 10 | **Accuracy 10/10 (100%)** with `mock` |

This measures the System-2 tool loop on known demos — **not** open-ended olympiad
skill and **not** a Kaggle leaderboard score. Plug in a real model backend to
evaluate genuine capability.

```bash
python scripts/run_eval.py
# → results/sample_eval.json
```

---

## Setup

```bash
git clone https://github.com/danielsolo707/AlphaMath.git
cd AlphaMath
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Optional LLM clients:

```bash
pip install openai        # GPT / Ollama / vLLM
pip install anthropic     # Claude
```

---

## Usage

### Evaluate sample problems (offline)

```bash
python scripts/run_eval.py
python scripts/run_eval.py --limit 5
```

### Solve one problem

```bash
python scripts/run_solve.py -p "What is gcd(252, 105)?"
python scripts/run_solve.py -p "Compute 2^10 mod 7." --json
```

### Use a real LLM

```bash
# OpenAI
set OPENAI_API_KEY=sk-...
python scripts/run_eval.py --backend openai

# Local Ollama (OpenAI-compatible)
# In configs or env: base_url=http://localhost:11434/v1
```

Example config override (`configs/local_ollama.yaml`):

```yaml
llm:
  backend: openai_compatible
  base_url: http://localhost:11434/v1
  model: qwen2.5-math:7b
  api_key_env: LLM_API_KEY
```

```bash
set LLM_API_KEY=not-needed
python scripts/run_eval.py --config configs/local_ollama.yaml
```

---

## Design honesty

| Claim | Status |
|-------|--------|
| Working agent loop + sandbox + eval harness | Yes |
| Offline reproducible demo | Yes (`mock`) |
| Trained 8B distilled olympiad weights in this repo | **No** |
| Public AIMO Progress Prize 3 leaderboard score | **Not claimed** |
| Full “logic surgery” distillation pipeline | Research plan only — see `docs/` |

For the full phase plan (distillation → QLoRA → pruning → launch), read
[`docs/DESIGN.md`](./docs/DESIGN.md).

---

## Competition context

[AIMO Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
asks models to solve international-level contest problems with **integer answers**.
Strong open solutions combine open-weight math models with **code execution** and
self-consistency. ALPHA-MATH implements that engineering core in a form you can
clone, run, and extend.

---

## License

MIT — see [LICENSE](./LICENSE) if present; otherwise code is provided for
portfolio / educational use. Contest problem statements remain under their
original competition terms; bundled demos are original.
