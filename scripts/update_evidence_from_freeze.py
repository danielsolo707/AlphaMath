"""Update pipeline_summary.json and print README metrics from a frozen AIME run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "results" / "kaggle_runs" / "v2_aime_2022_2024",
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    eval_matches = list(run_dir.rglob("evaluation.json"))
    if not eval_matches:
        raise SystemExit(f"No evaluation.json under {run_dir}")
    payload = json.loads(eval_matches[0].read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    total = int(payload.get("total") or 0)
    correct = int(payload.get("correct") or 0)
    accuracy = float(payload.get("accuracy") or metrics.get("accuracy") or 0.0)
    summary_path = ROOT / "results" / "pipeline_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = summary.setdefault("evidence", {})
    run_dir_abs = run_dir.resolve()
    try:
        run_rel = str(run_dir_abs.relative_to(ROOT.resolve())).replace("\\", "/") + "/"
    except ValueError:
        run_rel = str(run_dir).replace("\\", "/").rstrip("/") + "/"
    evidence.update(
        {
            "automated_regression_tests": 23,
            "real_model_benchmark_committed": True,
            "real_model_run_id": run_dir_abs.name,
            "real_model_run_path": run_rel,
            "real_model_accuracy": accuracy,
            "real_model_n_problems": total,
            "real_model_correct": correct,
            "real_model_dataset_tier": payload.get("dataset_tier") or "external_labeled",
            "real_model_hardware": "Tesla T4 (Kaggle)",
            "public_leaderboard_score_claimed": False,
            "custom_finetuned_weights_claimed": False,
            "sanity_run_id": "v1_real_qwen_sample10",
            "sanity_accuracy": 0.9,
            "sanity_n_problems": 10,
            "caveat": (
                f"Primary labeled evidence is AIME validation ({correct}/{total} = {accuracy:.1%}), "
                "not an AIMO public leaderboard score. Sanity 90% (9/10) remains a pipeline check only."
            ),
            "execution_success_rate": metrics.get("execution_success_rate"),
            "mean_vote_agreement": metrics.get("mean_vote_agreement"),
            "avg_latency_s": metrics.get("avg_latency_s"),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "updated": str(summary_path),
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
