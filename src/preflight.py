"""Actionable environment checks for local and Kaggle runs."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from src.sandbox import run_code


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def run_preflight(cfg: dict[str, Any], *, exercise_sandbox: bool = True) -> dict[str, Any]:
    llm_cfg = cfg.get("llm", {})
    backend = str(llm_cfg.get("backend", "transformers")).lower()
    real_backend = backend not in {"mock", "demo"}
    checks: list[dict[str, Any]] = []

    checks.append(_check("python", sys.version_info >= (3, 10), sys.version.split()[0]))
    for module in ("yaml", "sympy", "numpy"):
        available = importlib.util.find_spec(module) is not None
        required = module == "yaml" or real_backend
        checks.append(
            _check(
                f"dependency:{module}",
                available,
                "installed" if available else "missing; install requirements/core.txt",
                required=required,
            )
        )

    if real_backend:
        for module in ("torch", "transformers", "accelerate"):
            available = importlib.util.find_spec(module) is not None
            checks.append(
                _check(
                    f"dependency:{module}",
                    available,
                    "installed" if available else "missing; install requirements/gpu.txt",
                )
            )
        if llm_cfg.get("load_in_4bit") or llm_cfg.get("load_in_8bit"):
            available = importlib.util.find_spec("bitsandbytes") is not None
            checks.append(
                _check(
                    "dependency:bitsandbytes",
                    available,
                    "installed" if available else "missing; install requirements/quantization.txt",
                )
            )

        model_path = llm_cfg.get("model_path")
        if model_path:
            path = Path(model_path)
            checks.append(_check("model_path", path.exists(), str(path)))
            checks.append(
                _check(
                    "model_config",
                    (path / "config.json").exists(),
                    str(path / "config.json"),
                )
            )
        elif llm_cfg.get("local_files_only"):
            checks.append(_check("model_path", False, "local_files_only requires a model_path"))
        else:
            checks.append(
                _check(
                    "model_path",
                    True,
                    f"Hub id: {llm_cfg.get('model')}; network required for first download",
                )
            )

        try:
            import torch

            cuda = torch.cuda.is_available()
            detail = torch.cuda.get_device_name(0) if cuda else "CUDA unavailable"
            checks.append(_check("cuda", cuda, detail))
        except Exception as exc:
            checks.append(_check("cuda", False, f"torch check failed: {exc}"))

    if exercise_sandbox:
        timeout = min(float(cfg.get("sandbox", {}).get("timeout_seconds", 5)), 2.0)
        result = run_code("ANSWER = 6 * 7\nprint(ANSWER)", timeout_seconds=timeout)
        checks.append(
            _check(
                "sandbox",
                result.ok and result.answer == 42,
                result.error or f"answer={result.answer}, elapsed={result.elapsed_s:.3f}s",
            )
        )

    required_failures = [item for item in checks if item["required"] and not item["ok"]]
    return {
        "ok": not required_failures,
        "backend": backend,
        "platform": platform.platform(),
        "checks": checks,
        "failures": required_failures,
    }


def assert_preflight(report: dict[str, Any]) -> None:
    if report["ok"]:
        return
    details = "\n".join(f"- {item['name']}: {item['detail']}" for item in report["failures"])
    raise RuntimeError(f"ALPHA-MATH preflight failed:\n{details}")


def print_preflight(report: dict[str, Any]) -> None:
    print("ALPHA-MATH PREFLIGHT")
    for item in report["checks"]:
        mark = "OK" if item["ok"] else ("FAIL" if item["required"] else "WARN")
        print(f"[{mark:4}] {item['name']}: {item['detail']}")
    print("READY" if report["ok"] else "NOT READY")


def main() -> int:
    import argparse
    from src.utils import load_config

    parser = argparse.ArgumentParser(description="Validate ALPHA-MATH runtime")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_preflight(load_config(args.config))
    print(json.dumps(report, indent=2) if args.json else "", end="")
    if not args.json:
        print_preflight(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
