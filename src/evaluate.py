"""Batch evaluation harness on labeled sample problems."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from src.agent import MathAgent
from src.llm import build_llm
from src.dataio import load_labeled_problems
from src.preflight import assert_preflight, print_preflight, run_preflight
from src.reporting import write_evaluation_artifacts, zip_artifacts
from src.utils import ensure_dir, load_config, project_root


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
        # When agent clamps to 0..999, also clamp gold for fair compare
        gold_cmp = gold
        if gold is not None and agent.clamp_answer and agent.answer_max == 999 and gold >= 0:
            gold_cmp = int(gold) % 1000
        t_item = time.perf_counter()
        result = agent.solve(problem)
        elapsed = time.perf_counter() - t_item
        pred = result.answer
        is_correct = gold_cmp is not None and pred is not None and int(pred) == int(gold_cmp)
        # Default-answer fallback (0) should not count as success on unlabeled wrongs
        if result.meta.get("defaulted"):
            is_correct = False
        if is_correct:
            correct += 1
        sandbox_attempts = [a.sandbox for a in result.attempts if a.sandbox is not None]
        failure_types = [s.error_type for s in sandbox_attempts if not s.ok and s.error_type]
        row = {
            "id": pid,
            "problem": problem,
            "gold": gold,
            "gold_cmp": gold_cmp,
            "pred": pred,
            "correct": is_correct,
            "success": result.success,
            "attempts": len(result.attempts),
            "elapsed_s": round(elapsed, 3),
            "model": result.model,
            "backend": result.backend,
            "votes": result.meta.get("votes"),
            "vote_counts": result.meta.get("vote_counts"),
            "vote_agreement": result.meta.get("vote_agreement", 0.0),
            "vote_tied": result.meta.get("vote_tied", False),
            "defaulted": result.meta.get("defaulted", False),
            "budget_exhausted": result.meta.get("budget_exhausted", False),
            "execution_success": any(s.ok and s.answer is not None for s in sandbox_attempts),
            "sandbox_failures": sum(not s.ok for s in sandbox_attempts),
            "sandbox_timeouts": sum(s.timed_out for s in sandbox_attempts),
            "failure_types": failure_types,
            "tags": item.get("tags", []),
            "difficulty": item.get("difficulty"),
            "source": item.get("source"),
            "trace": result.to_dict()["attempts"],
        }
        rows.append(row)
        if verbose:
            mark = "OK" if is_correct else "MISS"
            print(f"[{i:02d}/{len(problems)}] {mark}  {pid}: pred={pred} gold={gold_cmp}  ({elapsed:.2f}s)")

    total = len(problems)
    acc = correct / total if total else 0.0
    latencies = [float(row["elapsed_s"]) for row in rows]
    attempt_counts = [int(row["attempts"]) for row in rows]
    agreements = [float(row["vote_agreement"]) for row in rows if row["votes"]]

    def _breakdown(key: str) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            values = row.get(key)
            if not isinstance(values, list):
                values = [values or "unknown"]
            for value in values:
                groups.setdefault(str(value), []).append(row)
        return {
            name: {
                "total": len(items),
                "correct": sum(bool(item["correct"]) for item in items),
                "accuracy": round(sum(bool(item["correct"]) for item in items) / len(items), 4),
            }
            for name, items in sorted(groups.items())
        }

    summary = {
        "schema_version": "2.0",
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "backend": rows[0]["backend"] if rows else None,
        "model": rows[0]["model"] if rows else None,
        "metrics": {
            "accuracy": round(acc, 4),
            "solved_rate": round(sum(bool(row["success"]) for row in rows) / total, 4) if total else 0.0,
            "execution_success_rate": round(sum(bool(row["execution_success"]) for row in rows) / total, 4) if total else 0.0,
            "default_rate": round(sum(bool(row["defaulted"]) for row in rows) / total, 4) if total else 0.0,
            "avg_attempts": round(statistics.mean(attempt_counts), 3) if attempt_counts else 0.0,
            "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "median_latency_s": round(statistics.median(latencies), 3) if latencies else 0.0,
            "max_latency_s": round(max(latencies), 3) if latencies else 0.0,
            "mean_vote_agreement": round(statistics.mean(agreements), 4) if agreements else 0.0,
            "sandbox_timeouts": sum(int(row["sandbox_timeouts"]) for row in rows),
        },
        "breakdown": {"difficulty": _breakdown("difficulty"), "tags": _breakdown("tags")},
        "per_problem": rows,
        "notes": (
            "Competition path uses open-weight Qwen2.5-Math-7B via transformers "
            "(offline, Kaggle GPU, no external API) — aligned with kernel "
            "danielsolo1770/alpha-math. "
            "Use configs/smoke_mock.yaml for CPU pipeline tests without downloading weights."
        ),
    }
    return summary


def build_agent_from_config(cfg: dict, llm=None) -> MathAgent:
    llm = llm or build_llm(cfg)
    agent_cfg = cfg.get("agent", {})
    sb = cfg.get("sandbox", {})
    default_on_fail = agent_cfg.get("default_answer_on_fail", 0)
    if default_on_fail is False or default_on_fail == "null":
        default_on_fail = None
    elif default_on_fail is not None:
        default_on_fail = int(default_on_fail)
    return MathAgent(
        llm,
        max_corrections=int(agent_cfg.get("max_corrections", agent_cfg.get("max_attempts", 2))),
        temperature=float(agent_cfg.get("temperature", 0.7)),
        top_p=float(agent_cfg.get("top_p", 0.9)),
        sandbox_timeout=float(sb.get("timeout_seconds", 5)),
        allowed_modules=list(sb.get("allowed_modules") or []),
        answer_min=int(cfg.get("answer_min", 0)),
        answer_max=int(cfg.get("answer_max", 999)),
        clamp_answer=bool(agent_cfg.get("clamp_answer", True)),
        majority_vote_k=int(agent_cfg.get("majority_vote_k", 3)),
        verbose_prompts=bool(agent_cfg.get("verbose_prompts", False)),
        default_answer_on_fail=default_on_fail,
        max_output_chars=int(sb.get("max_output_chars", 8000)),
        max_source_chars=int(sb.get("max_source_chars", 50000)),
        memory_limit_mb=sb.get("memory_limit_mb", 1536),
        time_budget_seconds=agent_cfg.get("time_budget_seconds", agent_cfg.get("timeout_seconds")),
        base_seed=int(agent_cfg.get("seed", 2026)),
        early_stop_majority=bool(agent_cfg.get("early_stop_majority", True)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ALPHA-MATH on sample problems")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--problems", default=None, help="Path to problems JSON")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N")
    parser.add_argument("--out", default=None, help="Write summary JSON here")
    parser.add_argument("--artifacts-dir", default=None, help="Write JSON/CSV/Markdown report bundle")
    parser.add_argument("--zip-artifacts", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Fail early on missing runtime requirements")
    parser.add_argument(
        "--backend",
        default=None,
        help="Override llm.backend (transformers|mock|openai|anthropic)",
    )
    parser.add_argument("--model-path", default=None, help="Override llm.model_path (local weights)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.setdefault("llm", {})["backend"] = args.backend
    if args.model_path:
        cfg.setdefault("llm", {})["model_path"] = args.model_path
        cfg["llm"]["local_files_only"] = True

    problems_path = args.problems or cfg["paths"]["sample_problems"]
    problems = load_labeled_problems(problems_path)
    if args.limit:
        problems = problems[: args.limit]

    preflight_report = None
    if args.preflight:
        preflight_report = run_preflight(cfg)
        print_preflight(preflight_report)
        assert_preflight(preflight_report)

    agent = build_agent_from_config(cfg)
    summary = evaluate(problems, agent, verbose=not args.quiet)

    out = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "sample_eval.json"
    ensure_dir(out.parent)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.artifacts_dir:
        artifacts = write_evaluation_artifacts(
            summary,
            cfg,
            args.artifacts_dir,
            dataset_path=problems_path,
            preflight=preflight_report,
        )
        print(f"Report:   {artifacts['markdown']}")
        if args.zip_artifacts:
            print(f"Archive:  {zip_artifacts(args.artifacts_dir)}")

    print()
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']:.1%}")
    print(f"Wrote:    {out}")
    print(f"Backend:  {summary['backend']} / {summary['model']}")
    return 0 if summary["correct"] == summary["total"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(project_root()))
    raise SystemExit(main())
