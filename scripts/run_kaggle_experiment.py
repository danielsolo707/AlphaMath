#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kaggle_experiment import run_kaggle_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run auditable ALPHA-MATH Kaggle experiment")
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--output-dir", default="/kaggle/working/alphamath_artifacts")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--submission-limit", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="Run pass1 vs repair vs voting study")
    args = parser.parse_args()
    run_kaggle_experiment(
        args.config,
        model_path=args.model_path,
        benchmark_path=args.benchmark,
        test_csv=args.test,
        output_dir=args.output_dir,
        eval_limit=args.eval_limit,
        submission_limit=args.submission_limit,
        run_evaluation=not args.skip_eval,
        run_competition_submission=not args.skip_submission,
        run_ablation=args.ablation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
