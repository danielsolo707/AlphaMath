"""Private Kaggle GPU entrypoint for a reproducible ALPHA-MATH evaluation."""

from __future__ import annotations

import json
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zipfile import ZIP_DEFLATED, ZipFile


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
SOURCE_ROOT = WORKING_ROOT / "alphamath_source"
OUTPUT_ROOT = WORKING_ROOT / "alphamath_artifacts"
BOOTSTRAP_REPORT = OUTPUT_ROOT / "kernel_bootstrap.json"
BITSANDBYTES_VERSION = "0.49.2"


def _prepare_repository() -> tuple[Path, str]:
    """Copy either Kaggle's expanded Dataset or the original ZIP to working storage."""
    bundles = sorted(INPUT_ROOT.rglob("AlphaMath_Kaggle_Bundle.zip"))
    expanded = sorted(
        path.parent
        for path in INPUT_ROOT.rglob("pyproject.toml")
        if path.parent.name == "AlphaMath" and (path.parent / "src" / "agent.py").is_file()
    )
    if len(bundles) + len(expanded) != 1:
        raise RuntimeError(
            "Expected exactly one AlphaMath source input (ZIP or expanded Dataset); "
            f"found bundles={list(map(str, bundles))}, trees={list(map(str, expanded))}"
        )
    if SOURCE_ROOT.exists():
        shutil.rmtree(SOURCE_ROOT)
    SOURCE_ROOT.mkdir(parents=True)
    repo = SOURCE_ROOT / "AlphaMath"
    if bundles:
        with ZipFile(bundles[0]) as archive:
            archive.extractall(SOURCE_ROOT)
        source = str(bundles[0])
    else:
        shutil.copytree(expanded[0], repo)
        source = str(expanded[0])
    if not (repo / "pyproject.toml").exists():
        raise RuntimeError(f"Prepared repository is incomplete: {repo}")
    return repo, source


def _discover_model_path() -> Path:
    candidates: list[tuple[int, int, Path]] = []
    for config in INPUT_ROOT.rglob("config.json"):
        directory = config.parent
        lowered = str(directory).lower()
        if "alphamath" in lowered:
            continue
        weight_files = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
        tokenizer_present = any(
            (directory / name).exists()
            for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json")
        )
        if not weight_files or not tokenizer_present:
            continue
        score = sum(token in lowered for token in ("qwen", "math", "instruct"))
        candidates.append((score, -len(directory.parts), directory))
    if not candidates:
        raise RuntimeError(
            "No offline Transformers model found under /kaggle/input. Attach the model source "
            "declared in kernel-metadata.json; its directory must contain config, tokenizer, and weights."
        )
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1], str(item[2])))
    return candidates[0][2]


def _write_bootstrap(payload: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_bitsandbytes() -> dict[str, str | bool]:
    """Install the pinned quantization runtime when the Kaggle image omits it."""
    if importlib.util.find_spec("bitsandbytes") is not None:
        return {
            "installed": False,
            "version": importlib.metadata.version("bitsandbytes"),
            "network_used": False,
        }
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        f"bitsandbytes=={BITSANDBYTES_VERSION}",
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    (OUTPUT_ROOT / "dependency_install.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"bitsandbytes installation failed with exit code {result.returncode}")
    importlib.invalidate_caches()
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("bitsandbytes installation completed but the module is not importable")
    return {
        "installed": True,
        "version": importlib.metadata.version("bitsandbytes"),
        "network_used": True,
    }


def _diagnostic_archive() -> Path:
    archive_path = WORKING_ROOT / "alphamath_kernel_diagnostics.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        if OUTPUT_ROOT.exists():
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(OUTPUT_ROOT.parent))
    return archive_path


def main() -> int:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "2026",
        }
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    bootstrap: dict = {"status": "starting", "python": sys.version, "input_root": str(INPUT_ROOT)}
    try:
        repo, source_input = _prepare_repository()

        test_result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
        (OUTPUT_ROOT / "regression_tests.log").write_text(test_result.stdout, encoding="utf-8")
        if test_result.returncode:
            raise RuntimeError(f"Regression tests failed with exit code {test_result.returncode}")

        model_path = _discover_model_path()
        dependency_bootstrap = _ensure_bitsandbytes()
        os.environ["ALPHAMATH_DEPENDENCY_BOOTSTRAP"] = json.dumps(dependency_bootstrap)

        # Log GPU early — Kaggle may assign P100 instead of T4.
        gpu_info: dict = {"cuda_available": False}
        try:
            import torch

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                gpu_info = {
                    "cuda_available": True,
                    "name": torch.cuda.get_device_name(0),
                    "capability": f"sm_{major}{minor}",
                    "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                    "bitsandbytes_ok": major >= 7,
                }
                print(f"GPU: {gpu_info}", flush=True)
                if major < 7:
                    print(
                        "[WARN] Old GPU (e.g. P100 sm_60): 4-bit load will be disabled automatically "
                        "to avoid bitsandbytes segfaults. Prefer T4/P100-safe float16 path.",
                        flush=True,
                    )
        except Exception as gpu_exc:  # pragma: no cover
            gpu_info = {"cuda_available": False, "error": str(gpu_exc)}

        bootstrap.update(
            {
                "status": "running_evaluation",
                "source_input": source_input,
                "repository": str(repo),
                "model_path": str(model_path),
                "regression_tests": "passed",
                "dependency_bootstrap": dependency_bootstrap,
                "gpu": gpu_info,
            }
        )
        _write_bootstrap(bootstrap)

        sys.path.insert(0, str(repo))
        os.chdir(repo)
        from src.kaggle_experiment import run_kaggle_experiment

        # Prefer attached AIME JSON; fall back to auto-discovery inside run_kaggle_experiment.
        aime_candidates = sorted(INPUT_ROOT.rglob("aime_2022_2024.json"))
        # Prefer the dedicated AIME dataset path over a copy nested inside the runtime tree.
        preferred = [
            path
            for path in aime_candidates
            if "alphamath-aime" in str(path).lower() or "aime_benchmark" in str(path).lower()
        ]
        chosen = preferred[0] if preferred else (aime_candidates[0] if aime_candidates else None)
        benchmark_path = str(chosen) if chosen else None
        if benchmark_path:
            print(f"Using AIME benchmark: {benchmark_path} (candidates={len(aime_candidates)})")
        else:
            print("No aime_2022_2024.json under /kaggle/input; auto-discovery will run.")

        # eval_limit=None runs the full set — do not silently cap at 10.
        result = run_kaggle_experiment(
            repo / "configs" / "kaggle.yaml",
            model_path=str(model_path),
            benchmark_path=benchmark_path,
            output_dir=OUTPUT_ROOT,
            eval_limit=None,
            run_competition_submission=False,
            run_ablation=False,
        )
        bootstrap.update(
            {
                "status": "completed",
                "final_report": str(result["final_report"]),
                "artifact_archive": str(result["archive"]),
            }
        )
        _write_bootstrap(bootstrap)
        report_text = Path(result["final_report"]).read_text(encoding="utf-8")
        print("\n" + "=" * 72 + "\n" + report_text + "\n" + "=" * 72)
        print(f"Download: {result['archive']}")
        return 0
    except Exception as exc:
        bootstrap.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_bootstrap(bootstrap)
        archive = _diagnostic_archive()
        print(f"ALPHA-MATH kernel failed. Diagnostics: {archive}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
