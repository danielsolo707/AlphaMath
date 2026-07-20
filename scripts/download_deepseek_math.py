#!/usr/bin/env python
"""Download DeepSeek-Math-7B-Instruct weights for local / Kaggle packaging.

Requires network once. After download, point configs at the local folder and
set local_files_only: true for offline runs.

  python scripts/download_deepseek_math.py
  python scripts/download_deepseek_math.py --out models/deepseek-math-7b-instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "deepseek-ai/deepseek-math-7b-instruct"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out",
        default=str(ROOT / "models" / "deepseek-math-7b-instruct"),
        help="Local directory to store weights",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"Downloading {args.model} → {out}")
    path = snapshot_download(
        repo_id=args.model,
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print(f"Ready: {path}")
    print()
    print("Then run:")
    print(f'  python scripts/run_eval.py --model-path "{out}" --backend transformers --limit 3')
    print("Or on Kaggle, upload this folder as a Dataset and set configs/kaggle.yaml model_path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
