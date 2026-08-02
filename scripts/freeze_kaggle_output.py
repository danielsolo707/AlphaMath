"""Download latest AlphaMath Kaggle kernel outputs and freeze under results/kaggle_runs/."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "danielsolo1770/alpha-math-real-model-evaluation"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Kaggle kernel output as portfolio evidence")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--run-id", default=None, help="Folder name under results/kaggle_runs/")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"v2_aime_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    dest = ROOT / "results" / "kaggle_runs" / run_id
    if dest.exists() and not args.force:
        print(f"Refusing to overwrite {dest}; pass --force or a new --run-id")
        return 2

    raw = dest / "_download"
    if dest.exists():
        shutil.rmtree(dest)
    raw.mkdir(parents=True)

    cmd = ["kaggle", "kernels", "output", args.slug, "-p", str(raw), "-o"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    artifacts = dest / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    zip_path = next(raw.rglob("alphamath_artifacts.zip"), None)
    diag_zip = next(raw.rglob("alphamath_kernel_diagnostics.zip"), None)

    if zip_path is not None:
        shutil.copy2(zip_path, dest / zip_path.name)
        with zipfile.ZipFile(zip_path) as handle:
            handle.extractall(artifacts)
    elif diag_zip is not None:
        shutil.copy2(diag_zip, dest / diag_zip.name)
        with zipfile.ZipFile(diag_zip) as handle:
            handle.extractall(artifacts)
    else:
        for path in raw.rglob("*"):
            if path.is_file():
                target = artifacts / path.relative_to(raw)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    report = artifacts / "FINAL_REPORT.md"
    # artifacts may nest one level if zip rooted under alphamath_artifacts/
    if not report.exists():
        nested = list(artifacts.rglob("FINAL_REPORT.md"))
        if nested:
            report = nested[0]
            # Prefer nested evaluation tree as the artifact root for NOTES.
            candidate_root = report.parent
            if (candidate_root / "evaluation").exists() or (candidate_root / "preflight.json").exists():
                artifacts = candidate_root

    eval_json = artifacts / "evaluation" / "evaluation.json"
    if not eval_json.exists():
        found = list(artifacts.rglob("evaluation.json"))
        eval_json = found[0] if found else eval_json

    summary: dict = {}
    if eval_json.exists():
        try:
            payload = json.loads(eval_json.read_text(encoding="utf-8"))
            summary = {
                "total": payload.get("total"),
                "metrics": payload.get("metrics"),
                "dataset_tier": payload.get("dataset_tier"),
            }
        except Exception as exc:  # noqa: BLE001
            summary = {"error": str(exc)}

    notes = {
        "run_id": run_id,
        "slug": args.slug,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_summary": summary,
    }
    body = report.read_text(encoding="utf-8") if report.exists() else "_No FINAL_REPORT.md in archive._\n"
    (dest / "NOTES.md").write_text(
        f"# Frozen Kaggle run — {run_id}\n\n```json\n{json.dumps(notes, indent=2)}\n```\n\n{body}",
        encoding="utf-8",
    )
    print(json.dumps({"dest": str(dest), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
