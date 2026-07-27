"""Kaggle AIMO submission runner — offline Qwen2.5-Math on competition GPU.

Aligned with the published kernel:
  https://www.kaggle.com/code/danielsolo1770/alpha-math

Usage (Kaggle notebook, internet OFF after model attached):

    from src.kaggle_submit import run_submission
    run_submission(config_path="configs/kaggle.yaml")

Writes submission.csv with id + prediction (notebook format).
AIMO sometimes expects id,answer — set paths.answer_column accordingly.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

from src.evaluate import build_agent_from_config
from src.utils import ensure_dir, load_config, project_root


PROBLEM_COLUMNS = ("problem", "question", "prompt", "text", "body")
ID_COLUMNS = ("id", "ID", "problem_id", "row_id")


def _detect_columns(fieldnames: list[str]) -> tuple[str, str]:
    ids = {c.lower(): c for c in fieldnames}
    id_col = next((ids[c.lower()] for c in ID_COLUMNS if c.lower() in ids), fieldnames[0])
    prob_col = next(
        (ids[c.lower()] for c in PROBLEM_COLUMNS if c.lower() in ids),
        fieldnames[1] if len(fieldnames) > 1 else fieldnames[0],
    )
    return id_col, prob_col


def find_test_csv(search_roots: list[str | Path] | None = None) -> Path | None:
    """Walk input dirs for test.csv (matches Kaggle notebook discovery)."""
    roots = search_roots or [Path("/kaggle/input"), Path("data")]
    candidates: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("test.csv"):
            if "models" in path.parts:
                continue
            candidates.append(path)
    if candidates:
        return candidates[0]
    # fallback: any non-sample csv under roots
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            name = path.name.lower()
            if "sample" in name or "submission" in name or "models" in path.parts:
                continue
            return path
    return None


def load_test_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        id_col, prob_col = _detect_columns(list(reader.fieldnames))
        rows = []
        for row in reader:
            rows.append({"id": row[id_col], "problem": row[prob_col]})
    return rows


def mock_test_rows() -> list[dict[str, Any]]:
    """Notebook mock problems when competition test.csv is unavailable."""
    return [
        {
            "id": "001",
            "problem": "What is the residue of 5^2026 modulo 13?",
        },
        {
            "id": "002",
            "problem": "If f(x) = 2x + 3, find f(f(3)).",
        },
        {
            "id": "003",
            "problem": "How many prime numbers are there between 10 and 30?",
        },
    ]


def run_submission(
    config_path: str | Path | None = None,
    *,
    test_csv: str | Path | None = None,
    out_csv: str | Path | None = None,
    model_path: str | None = None,
    limit: int | None = None,
    answer_column: str | None = None,
    allow_mock: bool = True,
) -> Path:
    cfg = load_config(config_path)
    cfg.setdefault("llm", {})
    cfg["llm"]["backend"] = cfg["llm"].get("backend") or "transformers"
    cfg["llm"]["local_files_only"] = True
    if model_path:
        cfg["llm"]["model_path"] = model_path

    paths = cfg.get("paths", {})
    ans_col = answer_column or paths.get("answer_column") or "prediction"

    test_path: Path | None
    if test_csv:
        test_path = Path(test_csv)
    elif paths.get("test_csv") and Path(paths["test_csv"]).exists():
        test_path = Path(paths["test_csv"])
    else:
        test_path = find_test_csv()

    submission_path = Path(out_csv or paths.get("submission_csv") or "submission.csv")
    ensure_dir(submission_path.parent)

    print("=" * 60)
    print("ALPHA-MATH · Kaggle submission")
    print(f"  model     : {cfg['llm'].get('model_path') or cfg['llm'].get('model')}")
    print(f"  backend   : {cfg['llm'].get('backend')}")
    print(f"  local_only: {cfg['llm'].get('local_files_only')}")
    print(f"  test      : {test_path}")
    print(f"  out       : {submission_path}")
    print(f"  column    : id,{ans_col}")
    print("=" * 60)

    agent = build_agent_from_config(cfg)

    if test_path and test_path.exists():
        rows = load_test_csv(test_path)
        mock_mode = False
    elif allow_mock:
        print("WARNING: test.csv not found — using mock problems (pipeline dry-run).")
        rows = mock_test_rows()
        mock_mode = True
    else:
        raise FileNotFoundError(
            "No test.csv found. Attach competition data or pass --test."
        )

    if limit is None and mock_mode:
        limit = 3
    if limit:
        rows = rows[:limit]

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        t_item = time.perf_counter()
        result = agent.solve(row["problem"])
        ans = result.answer if result.answer is not None else 0
        elapsed = time.perf_counter() - t_item
        print(f"[{i:03d}/{len(rows)}] id={row['id']} {ans_col}={ans}  ({elapsed:.1f}s)")
        results.append({"id": row["id"], ans_col: int(ans)})

    with open(submission_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", ans_col])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done in {time.perf_counter() - t0:.1f}s → {submission_path}")
    return submission_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ALPHA-MATH Kaggle AIMO submission")
    parser.add_argument("--config", default=None)
    parser.add_argument("--test", default=None, help="Path to test.csv")
    parser.add_argument("--out", default=None, help="Path to submission.csv")
    parser.add_argument("--model-path", default=None, help="Local math-model weights dir")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--answer-column",
        default=None,
        help="Submission value column (prediction|answer). Default from config.",
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Fail if test.csv is missing instead of using mock problems",
    )
    args = parser.parse_args(argv)

    run_submission(
        config_path=args.config,
        test_csv=args.test,
        out_csv=args.out,
        model_path=args.model_path,
        limit=args.limit,
        answer_column=args.answer_column,
        allow_mock=not args.no_mock,
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(project_root()))
    raise SystemExit(main())
