#!/usr/bin/env python
"""Build the local-only Kaggle upload archive from an explicit allowlist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "AlphaMath_Kaggle_Bundle.zip"
PACKAGE_OUTPUT = ROOT / "kaggle" / "AlphaMath_Kaggle_Upload_Package.zip"
CLI_DATASET_OUTPUT = ROOT / "kaggle" / "runtime_dataset" / OUTPUT.name
TOP_LEVEL_FILES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
)
DIRECTORIES = (
    "src",
    "scripts",
    "configs",
    "data",
    "docs",
    "notebooks",
    "tests",
    "kaggle",
    "requirements",
    "models",
)
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".zip")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def iter_bundle_files():
    for name in TOP_LEVEL_FILES:
        path = ROOT / name
        if path.exists():
            yield path
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            if EXCLUDED_PARTS.intersection(path.parts):
                continue
            yield path


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_bundle_files())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        commit, dirty = None, None
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "source_dirty": dirty,
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("AlphaMath") / path.relative_to(ROOT))
        archive.writestr(
            "AlphaMath/kaggle/BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )
    with ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        required = {
            "AlphaMath/src/agent.py",
            "AlphaMath/src/sandbox.py",
            "AlphaMath/configs/kaggle.yaml",
            "AlphaMath/notebooks/alphamath_portfolio_kaggle.ipynb",
            "AlphaMath/kaggle/BUNDLE_MANIFEST.json",
        }
        missing = required.difference(names)
        if missing:
            raise RuntimeError(f"Bundle validation failed; missing: {sorted(missing)}")
    with ZipFile(PACKAGE_OUTPUT, "w", ZIP_DEFLATED) as package:
        package.write(OUTPUT, OUTPUT.name)
        package.write(
            ROOT / "notebooks" / "alphamath_portfolio_kaggle.ipynb",
            "alphamath_portfolio_kaggle.ipynb",
        )
        package.write(ROOT / "kaggle" / "README_FIRST.md", "README_FIRST.md")
        package.write(ROOT / "data" / "benchmark_template.json", "benchmark_template.json")
        package.write(
            ROOT / "kaggle" / "runtime_dataset" / "dataset-metadata.json",
            "cli/runtime_dataset/dataset-metadata.json",
        )
        package.write(
            ROOT / "kaggle" / "kernel" / "kernel-metadata.json",
            "cli/kernel/kernel-metadata.json",
        )
        package.write(
            ROOT / "kaggle" / "kernel" / "alpha_math_kaggle.py",
            "cli/kernel/alpha_math_kaggle.py",
        )
    CLI_DATASET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUT, CLI_DATASET_OUTPUT)
    print(f"Built code input: {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB, {len(names)} files)")
    print(f"Built upload package: {PACKAGE_OUTPUT} ({PACKAGE_OUTPUT.stat().st_size / 1024:.1f} KiB)")
    print(f"Prepared CLI dataset: {CLI_DATASET_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
