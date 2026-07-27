"""Create auditable JSON, CSV, and Markdown evaluation artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from src.utils import ensure_dir


PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sympy",
    "numpy",
    "pyyaml",
)


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        git["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        git["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        pass
    bundle_manifest = Path(__file__).resolve().parents[1] / "kaggle" / "BUNDLE_MANIFEST.json"
    if bundle_manifest.exists():
        try:
            bundle = json.loads(bundle_manifest.read_text(encoding="utf-8"))
            git["bundle_source_commit"] = bundle.get("source_commit")
            git["bundle_source_dirty"] = bundle.get("source_dirty")
            git["bundle_created_at_utc"] = bundle.get("created_at_utc")
        except Exception:
            pass
    hardware: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        hardware["cuda_available"] = torch.cuda.is_available()
        hardware["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hardware.update(
                {
                    "gpu_name": props.name,
                    "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    except Exception as exc:
        hardware["torch_error"] = str(exc)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "hardware": hardware,
        "git": git,
        "kaggle_kernel_run_type": os.getenv("KAGGLE_KERNEL_RUN_TYPE"),
        "dependency_bootstrap": os.getenv("ALPHAMATH_DEPENDENCY_BOOTSTRAP"),
    }


def file_fingerprint(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "sha256": None, "size_bytes": None}
    source = Path(path)
    if not source.exists() or not source.is_file():
        return {"path": str(source), "sha256": None, "size_bytes": None}
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(source), "sha256": digest.hexdigest(), "size_bytes": source.stat().st_size}


def _markdown_report(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    backend = summary.get("backend")
    evidence = "MOCK PIPELINE TEST - NOT MODEL QUALITY" if backend == "mock" else "REAL MODEL EVALUATION"
    lines = [
        "# ALPHA-MATH Evaluation Report",
        "",
        f"**Evidence level:** {evidence}",
        f"**Model:** `{summary.get('model')}`  ",
        f"**Backend:** `{backend}`  ",
        f"**Dataset:** `{summary.get('dataset', {}).get('path', 'unspecified')}`  ",
        f"**Dataset SHA-256:** `{summary.get('dataset', {}).get('sha256', 'unavailable')}`  ",
        f"**Dataset tier:** `{summary.get('dataset_tier', 'unspecified')}`  ",
        f"**Problems:** {summary.get('total', 0)}",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {metrics.get('accuracy', summary.get('accuracy', 0)):.2%} |",
        f"| Execution success | {metrics.get('execution_success_rate', 0):.2%} |",
        f"| Mean vote agreement | {metrics.get('mean_vote_agreement', 0):.2%} |",
        f"| Average attempts | {metrics.get('avg_attempts', 0)} |",
        f"| Average latency | {metrics.get('avg_latency_s', 0)} s |",
        f"| Sandbox timeouts | {metrics.get('sandbox_timeouts', 0)} |",
        "",
        "## Per-problem results",
        "",
        "| ID | Correct | Gold | Prediction | Attempts | Time (s) | Agreement |",
        "|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("per_problem", []):
        lines.append(
            f"| {row['id']} | {'yes' if row['correct'] else 'no'} | {row['gold_cmp']} | "
            f"{row['pred']} | {row['attempts']} | {row['elapsed_s']} | "
            f"{float(row.get('vote_agreement', 0)):.2%} |"
        )
    lines.extend(["", "## Accuracy by difficulty", "", "| Difficulty | Correct | Total | Accuracy |", "|---|---:|---:|---:|"])
    for difficulty, values in summary.get("breakdown", {}).get("difficulty", {}).items():
        lines.append(
            f"| {difficulty} | {values['correct']} | {values['total']} | {values['accuracy']:.2%} |"
        )
    failure_counts: dict[str, int] = {}
    for row in summary.get("per_problem", []):
        for failure in row.get("failure_types", []):
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    lines.extend(["", "## Execution failure taxonomy", ""])
    if failure_counts:
        lines.extend(["| Failure | Count |", "|---|---:|"])
        for failure, count in sorted(failure_counts.items()):
            lines.append(f"| {failure} | {count} |")
    else:
        lines.append("No sandbox failures were recorded.")
    environment = summary.get("environment", {})
    hardware = environment.get("hardware", {})
    packages = environment.get("packages", {})
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Python: `{str(environment.get('python', '')).splitlines()[0]}`",
            f"- Platform: `{environment.get('platform')}`",
            f"- GPU: `{hardware.get('gpu_name', 'none')}`",
            f"- GPU memory: `{hardware.get('gpu_memory_gb', 'n/a')} GiB`",
            f"- Torch / Transformers: `{packages.get('torch')}` / `{packages.get('transformers')}`",
            "",
            "The accompanying `run_manifest.json` contains the full resolved config, package versions, "
            "hardware information, Git commit, seed, and preflight checks. `per_problem.csv` is suitable "
            "for analysis, while `evaluation.json` retains complete correction traces.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_artifacts(
    summary: dict[str, Any],
    cfg: dict[str, Any],
    out_dir: str | Path,
    *,
    dataset_path: str | Path | None = None,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output = ensure_dir(out_dir)
    summary["dataset"] = file_fingerprint(dataset_path)
    summary["environment"] = environment_snapshot()
    manifest = {
        "schema_version": "1.0",
        "config": cfg,
        "environment": summary["environment"],
        "dataset": summary["dataset"],
        "preflight": preflight,
        "evidence_level": "mock_pipeline_only" if summary.get("backend") == "mock" else "real_model",
    }
    json_path = output / "evaluation.json"
    manifest_path = output / "run_manifest.json"
    markdown_path = output / "REPORT.md"
    csv_path = output / "per_problem.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_markdown_report(summary), encoding="utf-8")

    csv_fields = [
        "id", "gold", "gold_cmp", "pred", "correct", "success", "execution_success",
        "attempts", "elapsed_s", "vote_agreement", "vote_tied", "sandbox_failures",
        "sandbox_timeouts", "difficulty", "source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary.get("per_problem", []):
            writer.writerow({key: row.get(key) for key in csv_fields})
    return {"json": json_path, "manifest": manifest_path, "markdown": markdown_path, "csv": csv_path}


def zip_artifacts(
    out_dir: str | Path,
    zip_path: str | Path | None = None,
    *,
    exclude_top_level: set[str] | None = None,
) -> Path:
    source = Path(out_dir)
    destination = Path(zip_path) if zip_path else source.parent / "alphamath_artifacts.zip"
    excluded = exclude_top_level or set()
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if not path.is_file() or (relative.parts and relative.parts[0] in excluded):
                continue
            archive.write(path, relative)
    return destination
