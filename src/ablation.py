"""Ablation study isolating the value of repair and self-consistency."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any

from src.evaluate import build_agent_from_config, evaluate
from src.llm import BaseLLM
from src.reporting import write_evaluation_artifacts
from src.utils import ensure_dir


VARIANTS = (
    ("tool_pass1", {"max_corrections": 0, "majority_vote_k": 1, "early_stop_majority": False}),
    ("tool_repair", {"max_corrections": 2, "majority_vote_k": 1, "early_stop_majority": False}),
)


def run_ablation_study(
    cfg: dict[str, Any],
    llm: BaseLLM,
    problems: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    dataset_path: str | Path,
    full_summary: dict[str, Any],
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = ensure_dir(out_dir)
    summaries: dict[str, dict[str, Any]] = {}
    for name, overrides in VARIANTS:
        variant_cfg = deepcopy(cfg)
        variant_cfg.setdefault("agent", {}).update(overrides)
        print(f"\nABLATION: {name} -> {overrides}")
        agent = build_agent_from_config(variant_cfg, llm=llm)
        summary = evaluate(problems, agent, verbose=True)
        summaries[name] = summary
        write_evaluation_artifacts(
            summary,
            variant_cfg,
            output / name,
            dataset_path=dataset_path,
            preflight=preflight,
        )

    summaries["tool_repair_vote"] = full_summary
    baseline_accuracy = summaries["tool_pass1"]["metrics"]["accuracy"]
    rows = []
    for name, summary in summaries.items():
        metrics = summary["metrics"]
        rows.append(
            {
                "variant": name,
                "accuracy": metrics["accuracy"],
                "delta_vs_pass1": round(metrics["accuracy"] - baseline_accuracy, 4),
                "execution_success_rate": metrics["execution_success_rate"],
                "avg_attempts": metrics["avg_attempts"],
                "avg_latency_s": metrics["avg_latency_s"],
                "mean_vote_agreement": metrics["mean_vote_agreement"],
            }
        )
    csv_path = output / "ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "# ALPHA-MATH Ablation Study",
        "",
        "This controlled comparison uses the same model, dataset, seed policy, and sandbox.",
        "",
        "| Variant | Accuracy | Delta vs pass1 | Exec success | Avg attempts | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['variant']} | {row['accuracy']:.2%} | {row['delta_vs_pass1']:+.2%} | "
            f"{row['execution_success_rate']:.2%} | {row['avg_attempts']} | {row['avg_latency_s']} |"
        )
    md_lines.extend(
        [
            "",
            "`tool_pass1` measures one generated program. `tool_repair` adds stateful execution feedback. "
            "`tool_repair_vote` adds independent sampling and strict-majority aggregation.",
            "",
        ]
    )
    markdown_path = output / "ABLATION.md"
    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")
    payload = {"schema_version": "1.0", "rows": rows}
    (output / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"rows": rows, "markdown": markdown_path, "csv": csv_path}
