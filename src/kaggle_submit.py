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
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.evaluate import build_agent_from_config
from src.agent import MathAgent
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


def detect_answer_column(test_path: Path | None, configured: str) -> str:
    """Prefer the competition sample-submission schema when it is available."""
    if test_path is None:
        return configured
    candidates = list(test_path.parent.glob("sample_submission*.csv"))
    candidates.extend(test_path.parent.parent.glob("sample_submission*.csv"))
    for candidate in candidates:
        try:
            with candidate.open(encoding="utf-8", newline="") as handle:
                fieldnames = csv.DictReader(handle).fieldnames or []
            values = [name for name in fieldnames if name.lower() not in {"id", "row_id", "problem_id"}]
            if len(values) == 1:
                return values[0]
        except Exception:
            continue
    return configured


def run_submission(
    config_path: str | Path | None = None,
    *,
    test_csv: str | Path | None = None,
    out_csv: str | Path | None = None,
    model_path: str | None = None,
    limit: int | None = None,
    answer_column: str | None = None,
    allow_mock: bool = True,
    agent: MathAgent | None = None,
    trace_json: str | Path | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
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

    if answer_column is None:
        ans_col = detect_answer_column(test_path, ans_col)

    submission_path = Path(out_csv or paths.get("submission_csv") or "submission.csv")
    ensure_dir(submission_path.parent)

    print("=" * 60)
    print("ALPHA-MATH - Kaggle submission")
    print(f"  model     : {cfg['llm'].get('model_path') or cfg['llm'].get('model')}")
    print(f"  backend   : {cfg['llm'].get('backend')}")
    print(f"  local_only: {cfg['llm'].get('local_files_only')}")
    print(f"  test      : {test_path}")
    print(f"  out       : {submission_path}")
    print(f"  column    : id,{ans_col}")
    print("=" * 60)

    if agent is None:
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

    result_by_id: dict[str, dict[str, Any]] = {}
    trace_by_id: dict[str, dict[str, Any]] = {}
    if resume and submission_path.exists():
        try:
            with submission_path.open(encoding="utf-8", newline="") as handle:
                for existing in csv.DictReader(handle):
                    if "id" in existing and ans_col in existing:
                        result_by_id[str(existing["id"])] = {
                            "id": existing["id"],
                            ans_col: int(existing[ans_col]),
                        }
            if trace_json and Path(trace_json).exists():
                payload = json.loads(Path(trace_json).read_text(encoding="utf-8"))
                trace_by_id = {str(item["id"]): item for item in payload.get("rows", [])}
            if result_by_id:
                print(f"Resume: found {len(result_by_id)} completed predictions.")
        except Exception as exc:
            print(f"Resume checkpoint ignored ({type(exc).__name__}: {exc}).")
            result_by_id.clear()
            trace_by_id.clear()
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        row_id = str(row["id"])
        if row_id in result_by_id:
            print(f"[{i:03d}/{len(rows)}] id={row_id} resumed")
            continue
        t_item = time.perf_counter()
        result = agent.solve(row["problem"])
        ans = result.answer if result.answer is not None else 0
        elapsed = time.perf_counter() - t_item
        print(f"[{i:03d}/{len(rows)}] id={row['id']} {ans_col}={ans}  ({elapsed:.1f}s)")
        result_by_id[row_id] = {"id": row["id"], ans_col: int(ans)}
        trace_by_id[row_id] = {
                "id": row["id"],
                "prediction": int(ans),
                "elapsed_s": round(elapsed, 3),
                "success": result.success,
                "meta": result.meta,
                "attempts": result.to_dict()["attempts"],
            }
        if checkpoint_every > 0 and i % checkpoint_every == 0:
            completed_results = [result_by_id[str(item["id"])] for item in rows if str(item["id"]) in result_by_id]
            completed_traces = [trace_by_id[str(item["id"])] for item in rows if str(item["id"]) in trace_by_id]
            _write_submission_csv(submission_path, completed_results, ans_col)
            if trace_json:
                ensure_dir(Path(trace_json).parent)
                Path(trace_json).write_text(
                    json.dumps({"completed": len(completed_results), "total": len(rows), "rows": completed_traces}, indent=2),
                    encoding="utf-8",
                )

    results = [result_by_id[str(item["id"])] for item in rows if str(item["id"]) in result_by_id]
    traces = [trace_by_id[str(item["id"])] for item in rows if str(item["id"]) in trace_by_id]
    _write_submission_csv(submission_path, results, ans_col)
    if trace_json:
        Path(trace_json).write_text(
            json.dumps({"completed": len(rows), "total": len(rows), "rows": traces}, indent=2),
            encoding="utf-8",
        )

    print(f"Done in {time.perf_counter() - t0:.1f}s -> {submission_path}")
    return submission_path


def _write_submission_csv(path: Path, rows: list[dict[str, Any]], answer_column: str) -> None:
    """Atomically checkpoint a valid submission after completed rows."""
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", answer_column])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
