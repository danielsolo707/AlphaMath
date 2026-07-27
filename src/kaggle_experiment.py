"""One-load Kaggle experiment: preflight, real evaluation, submission, final report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.ablation import run_ablation_study
from src.dataio import discover_labeled_benchmark, load_labeled_problems
from src.evaluate import build_agent_from_config, evaluate
from src.kaggle_submit import find_test_csv, run_submission
from src.preflight import assert_preflight, print_preflight, run_preflight
from src.reporting import environment_snapshot, write_evaluation_artifacts, zip_artifacts
from src.utils import ensure_dir, load_config


def _write_final_report(
    output: Path,
    *,
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    evaluation_summary: dict[str, Any] | None,
    evaluation_report: Path | None,
    submission_path: Path | None,
    submission_trace: Path | None,
    ablation_ran: bool,
) -> Path:
    llm_cfg = cfg.get("llm", {})
    lines = [
        "# ALPHA-MATH Kaggle Run - Final Report",
        "",
        f"**Backend:** `{llm_cfg.get('backend')}`  ",
        f"**Model:** `{llm_cfg.get('model_path') or llm_cfg.get('model')}`  ",
        f"**Preflight:** {'PASS' if preflight.get('ok') else 'FAIL'}  ",
        f"**Dependency bootstrap:** `{os.getenv('ALPHAMATH_DEPENDENCY_BOOTSTRAP', 'none')}`  ",
        "",
    ]
    if evaluation_summary:
        metrics = evaluation_summary.get("metrics", {})
        lines.extend(
            [
                "## Labeled evaluation",
                "",
                f"- Problems: {evaluation_summary.get('total')}",
                f"- Accuracy: {metrics.get('accuracy', 0):.2%}",
                f"- Execution success: {metrics.get('execution_success_rate', 0):.2%}",
                f"- Mean vote agreement: {metrics.get('mean_vote_agreement', 0):.2%}",
                f"- Average latency: {metrics.get('avg_latency_s', 0)} seconds/problem",
                f"- Detailed report: `{evaluation_report.relative_to(output) if evaluation_report else None}`",
                "",
            ]
        )
        if ablation_ran:
            lines.extend(
                [
                    "## Ablation evidence",
                    "",
                    "A controlled pass1 vs repair vs repair+vote study was completed.",
                    "See `ablation/ABLATION.md` and `ablation/ablation.csv`.",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Labeled evaluation",
                "",
                "No labeled benchmark was run. Do not claim an accuracy score from this run.",
                "",
            ]
        )
    if submission_path:
        submission_metrics: dict[str, Any] = {}
        if submission_trace and submission_trace.exists():
            try:
                trace_payload = json.loads(submission_trace.read_text(encoding="utf-8"))
                trace_rows = trace_payload.get("rows", [])
                count = len(trace_rows)
                submission_metrics = {
                    "completed": count,
                    "success_rate": sum(bool(row.get("success")) for row in trace_rows) / count if count else 0.0,
                    "default_rate": sum(bool(row.get("meta", {}).get("defaulted")) for row in trace_rows) / count if count else 0.0,
                    "avg_latency_s": sum(float(row.get("elapsed_s", 0)) for row in trace_rows) / count if count else 0.0,
                    "mean_vote_agreement": sum(float(row.get("meta", {}).get("vote_agreement", 0)) for row in trace_rows) / count if count else 0.0,
                }
            except Exception:
                submission_metrics = {}
        lines.extend(
            [
                "## Competition inference",
                "",
                f"- Submission: `{submission_path.relative_to(output)}`",
                f"- Full traces/checkpoints: `{submission_trace.relative_to(output) if submission_trace else None}`",
                f"- Completed rows: {submission_metrics.get('completed', 'unknown')}",
                f"- Successful tool paths: {submission_metrics.get('success_rate', 0):.2%}",
                f"- Default-answer rate: {submission_metrics.get('default_rate', 0):.2%}",
                f"- Average latency: {submission_metrics.get('avg_latency_s', 0):.2f} seconds/problem",
                f"- Mean vote agreement: {submission_metrics.get('mean_vote_agreement', 0):.2%}",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence policy",
            "",
            "Only metrics produced with `backend=transformers` and a labeled external or bundled dataset "
            "are model results. Mock results validate plumbing only. Keep `run_manifest.json` beside any "
            "number copied into the public README.",
            "",
            "## Included artifacts",
            "",
            "- `run_manifest.json`: resolved config, environment, GPU, package versions, preflight",
            "- `evaluation/`: full JSON traces, flat CSV, and Markdown report",
            "- `submission.csv`: checkpointed predictions when competition data was present",
            "- `submission_trace.json`: per-problem latency, votes, repairs, and failures",
            "",
        ]
    )
    path = output / "FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_kaggle_experiment(
    config_path: str | Path,
    *,
    model_path: str | None = None,
    benchmark_path: str | Path | None = None,
    test_csv: str | Path | None = None,
    output_dir: str | Path = "/kaggle/working/alphamath_artifacts",
    eval_limit: int | None = None,
    submission_limit: int | None = None,
    run_evaluation: bool = True,
    run_competition_submission: bool = True,
    run_ablation: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    output = ensure_dir(output_dir)
    if model_path:
        cfg.setdefault("llm", {})["model_path"] = str(model_path)
    cfg.setdefault("llm", {})["backend"] = "transformers"
    cfg["llm"]["local_files_only"] = True
    cfg["llm"]["allow_mock_fallback"] = False

    preflight = run_preflight(cfg)
    print_preflight(preflight)
    assert_preflight(preflight)
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    print("Loading the math model once for all requested stages...")
    agent = build_agent_from_config(cfg)
    evaluation_summary = None
    evaluation_report = None
    ablation_result = None
    evaluation_problems: list[dict[str, Any]] | None = None

    selected_benchmark = Path(benchmark_path) if benchmark_path else None
    if run_evaluation and selected_benchmark is None:
        selected_benchmark = discover_labeled_benchmark(["/kaggle/input"])
    if run_evaluation and selected_benchmark is None:
        selected_benchmark = Path(cfg["paths"]["sample_problems"])
        print("No external labeled benchmark found; using bundled sanity problems.")

    if run_evaluation and selected_benchmark:
        problems = load_labeled_problems(selected_benchmark)
        if eval_limit:
            problems = problems[:eval_limit]
        evaluation_problems = problems
        evaluation_summary = evaluate(problems, agent, verbose=True)
        evaluation_summary["dataset_tier"] = (
            "bundled_sanity" if selected_benchmark.name == "sample_problems.json" else "external_labeled"
        )
        eval_paths = write_evaluation_artifacts(
            evaluation_summary,
            cfg,
            output / "evaluation",
            dataset_path=selected_benchmark,
            preflight=preflight,
        )
        evaluation_report = eval_paths["markdown"]
        if run_ablation:
            ablation_result = run_ablation_study(
                cfg,
                agent.llm,
                evaluation_problems,
                output / "ablation",
                dataset_path=selected_benchmark,
                full_summary=evaluation_summary,
                preflight=preflight,
            )

    submission_path = None
    submission_trace = None
    selected_test = Path(test_csv) if test_csv else find_test_csv(["/kaggle/input"])
    if run_competition_submission and selected_test:
        submission_path = output / "submission.csv"
        submission_trace = output / "submission_trace.json"
        run_submission(
            config_path=config_path,
            test_csv=selected_test,
            out_csv=submission_path,
            model_path=model_path,
            limit=submission_limit,
            allow_mock=False,
            agent=agent,
            trace_json=submission_trace,
            checkpoint_every=1,
        )
    elif run_competition_submission:
        print("No competition test.csv found; submission stage skipped.")

    manifest = {
        "schema_version": "1.0",
        "config": cfg,
        "environment": environment_snapshot(),
        "preflight": preflight,
        "benchmark": str(selected_benchmark) if selected_benchmark else None,
        "test_csv": str(selected_test) if selected_test else None,
        "dataset_tier": evaluation_summary.get("dataset_tier") if evaluation_summary else None,
        "ablation_ran": bool(ablation_result),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    final_report = _write_final_report(
        output,
        cfg=cfg,
        preflight=preflight,
        evaluation_summary=evaluation_summary,
        evaluation_report=evaluation_report,
        submission_path=submission_path,
        submission_trace=submission_trace,
        ablation_ran=bool(ablation_result),
    )
    excluded: set[str] = set()
    if evaluation_summary is None:
        excluded.add("evaluation")
    if not ablation_result:
        excluded.add("ablation")
    if submission_path is None:
        excluded.update({"submission.csv", "submission_trace.json"})
    archive = zip_artifacts(
        output,
        Path(output).parent / "alphamath_artifacts.zip",
        exclude_top_level=excluded,
    )
    print(f"Final report: {final_report}")
    print(f"Downloadable artifacts: {archive}")
    return {
        "output_dir": output,
        "final_report": final_report,
        "archive": archive,
        "evaluation": evaluation_summary,
        "submission": submission_path,
        "ablation": ablation_result,
    }
