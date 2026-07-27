"""Load labeled evaluation problems from JSON, JSONL, or CSV."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROBLEM_COLUMNS = ("problem", "question", "prompt", "text", "body")
ANSWER_COLUMNS = ("answer", "gold", "target", "label", "prediction")
ID_COLUMNS = ("id", "problem_id", "row_id")


def _normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    lowered = {str(key).lower(): key for key in row}
    problem_key = next((lowered[name] for name in PROBLEM_COLUMNS if name in lowered), None)
    answer_key = next((lowered[name] for name in ANSWER_COLUMNS if name in lowered), None)
    id_key = next((lowered[name] for name in ID_COLUMNS if name in lowered), None)
    if problem_key is None:
        raise ValueError(f"Row {index} has no problem column ({', '.join(PROBLEM_COLUMNS)})")
    if answer_key is None:
        raise ValueError(f"Row {index} has no answer column ({', '.join(ANSWER_COLUMNS)})")
    normalized = dict(row)
    normalized["id"] = str(row.get(id_key, f"p{index:04d}"))
    normalized["problem"] = str(row[problem_key])
    raw_answer = row[answer_key]
    if isinstance(raw_answer, float) and raw_answer.is_integer():
        normalized["answer"] = int(raw_answer)
    else:
        answer_text = str(raw_answer).strip()
        match = re.search(r"-?\d+", answer_text)
        if not match:
            raise ValueError(f"Row {index} answer is not an integer: {raw_answer!r}")
        normalized["answer"] = int(match.group())
    normalized.setdefault("source", "external-benchmark")
    normalized.setdefault("tags", [])
    return normalized


def load_labeled_problems(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("problems") if isinstance(payload, dict) else payload
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported benchmark format: {source.suffix}")
    if not isinstance(rows, list):
        raise ValueError("Benchmark must contain a list of problem records")
    return [_normalize_row(dict(row), index) for index, row in enumerate(rows, 1)]


def discover_labeled_benchmark(roots: list[str | Path]) -> Path | None:
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for suffix in ("*.jsonl", "*.json", "*.csv"):
            for path in sorted(root.rglob(suffix)):
                lowered = str(path).lower()
                if any(token in lowered for token in ("model", "submission", "sample_problems", "template")):
                    continue
                try:
                    problems = load_labeled_problems(path)
                except Exception:
                    continue
                if problems:
                    return path
    return None
