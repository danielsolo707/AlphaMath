"""Batch evaluation harness on labeled sample problems."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.agent import MathAgent
from src.llm import build_llm
from src.utils import ensure_dir, load_config, load_problems, project_root


def evaluate(
    problems: list[dict],
    agent: MathAgent,
    *,
    verbose: bool = True,
) -> dict:
    rows = []
    correct = 0
    t0 = time.perf_counter()

    for i, item in enumerate(problems, 1):
        pid = item.get("id", f"p{i}")
        problem = item["problem"]
        gold = item.get("answer")
        t_item = time.perf_counter()
        result = agent.solve(problem)
        elapsed = time.perf_counter() - t_item
        pred = result.answer
        is_correct = gold is not None and pred is not None and int(pred) == int(gold)
        if is_correct:
            correct += 1
        row = {
            "id": pid,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "success": result.success,
            "attempts": len(result.attempts),
            "elapsed_s": round(elapsed, 3),
            "model": result.model,
            "backend": result.backend,
            "tags": item.get("tags", []),
            "difficulty": item.get("difficulty"),
        }
        rows.append(row)
        if verbose:
            mark = "OK" if is_correct else "MISS"
            print(f"[{i:02d}/{len(problems)}] {mark}  {pid}: pred={pred} gold={gold}  ({elapsed:.2f}s)")

    total = len(problems)
    acc = correct / total if total else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "backend": rows[0]["backend"] if rows else None,
        "model": rows[0]["model"] if rows else None,
        "per_problem": rows,
        "notes": (
            "Default backend is mock (offline demo templates). "
            "This measures the System-2 pipeline, not open-ended LLM olympiad skill. "
            "Set llm.backend to openai / anthropic for real model evaluation."
        ),
    }
    return summary


def build_agent_from_config(cfg: dict) -> MathAgent:
    llm = build_llm(cfg)
    agent_cfg = cfg.get("agent", {})
    sb = cfg.get("sandbox", {})
    return MathAgent(
        llm,
        max_attempts=int(agent_cfg.get("max_attempts", 3)),
        temperature=float(agent_cfg.get("temperature", 0.2)),
        sandbox_timeout=float(sb.get("timeout_seconds", 5)),
        allowed_modules=list(sb.get("allowed_modules") or []),
        answer_min=int(cfg.get("answer_min", 0)),
        answer_max=int(cfg.get("answer_max", 999)),
        clamp_answer=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ALPHA-MATH on sample problems")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--problems", default=None, help="Path to problems JSON")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N")
    parser.add_argument("--out", default=None, help="Write summary JSON here")
    parser.add_argument("--backend", default=None, help="Override llm.backend (mock|openai|anthropic)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.setdefault("llm", {})["backend"] = args.backend

    problems_path = args.problems or cfg["paths"]["sample_problems"]
    problems = load_problems(problems_path)
    if args.limit:
        problems = problems[: args.limit]

    agent = build_agent_from_config(cfg)
    summary = evaluate(problems, agent, verbose=not args.quiet)

    out = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "sample_eval.json"
    ensure_dir(out.parent)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']:.1%}")
    print(f"Wrote:    {out}")
    print(f"Backend:  {summary['backend']} / {summary['model']}")
    return 0 if summary["correct"] == summary["total"] else 1


if __name__ == "__main__":
    # Allow `python -m src.evaluate` from repo root
    sys.path.insert(0, str(project_root()))
    raise SystemExit(main())
