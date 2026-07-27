#!/usr/bin/env python
"""Backward-compatible wrapper — prefer scripts/download_math_model.py.

Downloads DeepSeek-Math-7B-Instruct (alternate math specialist).
The published Kaggle ALPHA-MATH run used Qwen2.5-Math-7B.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_math_model.py"

if __name__ == "__main__":
    # Inject defaults if user did not pass flags
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "deepseek-ai/deepseek-math-7b-instruct"])
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(ROOT / "models" / "deepseek-math-7b-instruct")])
    sys.argv[0] = str(SCRIPT)
    runpy.run_path(str(SCRIPT), run_name="__main__")
