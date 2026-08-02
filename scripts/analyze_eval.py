"""Summarize an evaluation.json for portfolio write-ups and failure triage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ALPHA-MATH evaluation artifacts")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to evaluation.json or a run folder containing it",
    )
    args = parser.parse_args()
    path = args.path
    if path.is_dir():
        matches = list(path.rglob("evaluation.json"))
        if not matches:
            raise SystemExit(f"No evaluation.json under {path}")
        path = matches[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        payload.get("per_problem")
        or payload.get("rows")
        or payload.get("results")
        or []
    )
    metrics = payload.get("metrics") or {}
    total = int(payload.get("total") or len(rows) or 0)
    correct = int(payload.get("correct") or sum(1 for row in rows if row.get("correct")))
    defaulted = sum(1 for row in rows if row.get("defaulted"))
    timeouts = sum(int(row.get("sandbox_timeouts") or 0) for row in rows)
    exec_ok = sum(1 for row in rows if row.get("execution_success"))
    miss_ids = [row.get("id") for row in rows if not row.get("correct")]
    fail_types: Counter[str] = Counter()
    for row in rows:
        for item in row.get("failure_types") or []:
            fail_types[str(item)] += 1
    latencies = [float(row.get("elapsed_s") or 0) for row in rows]
    avg_lat = sum(latencies) / len(latencies) if latencies else float(metrics.get("avg_latency_s") or 0)

    summary = {
        "path": str(path),
        "dataset_tier": payload.get("dataset_tier"),
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else float(payload.get("accuracy") or 0.0),
        "metrics_block": metrics,
        "execution_success_rate": (exec_ok / total) if total and rows else metrics.get("execution_success_rate"),
        "defaulted_count": defaulted,
        "sandbox_timeout_events": timeouts,
        "avg_latency_s": round(avg_lat, 3),
        "top_failure_types": fail_types.most_common(10),
        "miss_ids_head": miss_ids[:20],
        "miss_count": len(miss_ids),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
