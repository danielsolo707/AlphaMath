#!/usr/bin/env python
"""Download open-weight math model for local / Kaggle packaging.

Default: Qwen2.5-Math-7B-Instruct (matches the published ALPHA-MATH Kaggle run).

Requires network once. After download, point configs at the local folder and
set local_files_only: true for offline runs.

  python scripts/download_math_model.py
  python scripts/download_math_model.py --model deepseek-ai/deepseek-math-7b-instruct
  python scripts/download_math_model.py --out models/qwen2.5-math-7b-instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download math LLM weights")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out",
        default=None,
        help="Local directory to store weights (default under models/)",
    )
    args = parser.parse_args()

    slug = args.model.split("/")[-1].lower()
    out = Path(args.out) if args.out else ROOT / "models" / slug
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"Downloading {args.model} -> {out}")
    path = snapshot_download(
        repo_id=args.model,
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print(f"Ready: {path}")
    print()
    print("Then run:")
    print(f'  python scripts/run_eval.py --model-path "{out}" --backend transformers --limit 3')
    print("Or on Kaggle, upload this folder as a Dataset/Model and set configs/kaggle.yaml model_path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
