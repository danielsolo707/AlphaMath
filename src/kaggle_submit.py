"""Kaggle AIMO submission runner — offline DeepSeek-Math on competition GPU.

Usage (inside a Kaggle notebook with internet OFF after deps/model attached):

    import sys
    sys.path.append("/kaggle/working/AlphaMath")  # or wherever the repo lives
    from src.kaggle_submit import run_submission
    run_submission(config_path="/kaggle/working/AlphaMath/configs/kaggle.yaml")

Expects a test CSV with columns including an id and a problem statement.
Writes submission.csv with id,answer.
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


def run_submission(
    config_path: str | Path | None = None,
    *,
    test_csv: str | Path | None = None,
    out_csv: str | Path | None = None,
    model_path: str | None = None,
    limit: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    # Force offline transformers path for competition
    cfg.setdefault("llm", {})
    cfg["llm"]["backend"] = cfg["llm"].get("backend") or "transformers"
    cfg["llm"]["local_files_only"] = True
    if model_path:
        cfg["llm"]["model_path"] = model_path

    paths = cfg.get("paths", {})
    test_path = Path(test_csv or paths.get("test_csv") or "test.csv")
    submission_path = Path(out_csv or paths.get("submission_csv") or "submission.csv")
    ensure_dir(submission_path.parent)

    print("=" * 60)
    print("ALPHA-MATH · Kaggle submission")
    print(f"  model     : {cfg['llm'].get('model_path') or cfg['llm'].get('model')}")
    print(f"  backend   : {cfg['llm'].get('backend')}")
    print(f"  local_only: {cfg['llm'].get('local_files_only')}")
    print(f"  test      : {test_path}")
    print(f"  out       : {submission_path}")
    print("=" * 60)

    agent = build_agent_from_config(cfg)
    rows = load_test_csv(test_path)
    if limit:
        rows = rows[:limit]

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        t_item = time.perf_counter()
        result = agent.solve(row["problem"])
        # Competition-safe default if model fails entirely
        ans = result.answer if result.answer is not None else 0
        elapsed = time.perf_counter() - t_item
        print(f"[{i:03d}/{len(rows)}] id={row['id']} answer={ans}  ({elapsed:.1f}s)")
        results.append({"id": row["id"], "answer": int(ans)})

    with open(submission_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done in {time.perf_counter() - t0:.1f}s → {submission_path}")
    return submission_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ALPHA-MATH Kaggle AIMO submission")
    parser.add_argument("--config", default=None)
    parser.add_argument("--test", default=None, help="Path to test.csv")
    parser.add_argument("--out", default=None, help="Path to submission.csv")
    parser.add_argument("--model-path", default=None, help="Local DeepSeek-Math weights dir")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    run_submission(
        config_path=args.config,
        test_csv=args.test,
        out_csv=args.out,
        model_path=args.model_path,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(project_root()))
    raise SystemExit(main())
