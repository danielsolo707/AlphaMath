"""Shared helpers: config loading, answer parsing, path resolution."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return ROOT


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {cfg_path}")
    # Resolve relative paths against repo root
    paths = cfg.get("paths", {})
    for key, val in list(paths.items()):
        if val and not Path(val).is_absolute():
            paths[key] = str((ROOT / val).resolve())
    cfg["paths"] = paths
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Reject ambiguous or dangerous experiment settings before model loading."""
    answer_min = int(cfg.get("answer_min", 0))
    answer_max = int(cfg.get("answer_max", 999))
    if answer_min > answer_max:
        raise ValueError("answer_min must be <= answer_max")
    agent = cfg.get("agent", {}) or {}
    corrections = int(agent.get("max_corrections", agent.get("max_attempts", 2)))
    if corrections < 0:
        raise ValueError("agent.max_corrections must be >= 0")
    if int(agent.get("majority_vote_k", 3)) < 1:
        raise ValueError("agent.majority_vote_k must be >= 1")
    if agent.get("time_budget_seconds") is not None and float(agent["time_budget_seconds"]) <= 0:
        raise ValueError("agent.time_budget_seconds must be positive")
    sandbox = cfg.get("sandbox", {}) or {}
    if float(sandbox.get("timeout_seconds", 5)) <= 0:
        raise ValueError("sandbox.timeout_seconds must be positive")
    if int(sandbox.get("max_output_chars", 8000)) < 256:
        raise ValueError("sandbox.max_output_chars must be >= 256")
    if int(sandbox.get("max_source_chars", 50000)) < 1000:
        raise ValueError("sandbox.max_source_chars must be >= 1000")


def load_problems(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of problems in {path}")
    return data


def extract_integer_answer(
    text: str | None,
    answer_min: int = 0,
    answer_max: int = 999,
    clamp: bool = False,
) -> int | None:
    """Pull a final integer answer from free-form model / sandbox text.

    Preference order:
      1. Explicit markers: FINAL_ANSWER: N / Answer: N / \\\\boxed{N}
      2. Last standalone integer in the text
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    patterns = [
        r"FINAL_ANSWER\s*[:=]\s*(-?\d+)",
        r"(?i)final\s*answer\s*[:=]\s*(-?\d+)",
        r"(?i)answer\s*[:=]\s*(-?\d+)",
        r"\\boxed\{\s*(-?\d+)\s*\}",
        r"boxed\{\s*(-?\d+)\s*\}",
    ]
    for pat in patterns:
        matches = re.findall(pat, s)
        if matches:
            return _normalize_int(int(matches[-1]), answer_min, answer_max, clamp)

    # Last integer token
    nums = re.findall(r"-?\d+", s)
    if not nums:
        return None
    return _normalize_int(int(nums[-1]), answer_min, answer_max, clamp)


def _normalize_int(value: int, lo: int, hi: int, clamp: bool) -> int | None:
    if clamp:
        # AIME-style: last three digits of non-negative integers
        if value < 0:
            return None
        return value % 1000 if hi == 999 else max(lo, min(hi, value))
    if lo <= value <= hi:
        return value
    # Still accept out-of-range for demo problems that intentionally exceed 999
    return value


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
