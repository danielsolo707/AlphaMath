"""CLI: solve a single problem from string or file."""

from __future__ import annotations

import argparse
import json
import sys

from src.evaluate import build_agent_from_config
from src.utils import load_config, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve one math problem with ALPHA-MATH")
    parser.add_argument("--problem", "-p", default=None, help="Problem text")
    parser.add_argument("--file", "-f", default=None, help="Read problem text from file")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default=None, help="Override llm.backend")
    parser.add_argument("--model-path", default=None, help="Local DeepSeek-Math weights dir")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    args = parser.parse_args(argv)

    if not args.problem and not args.file:
        parser.error("Provide --problem or --file")

    text = args.problem
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()

    cfg = load_config(args.config)
    if args.backend:
        cfg.setdefault("llm", {})["backend"] = args.backend
    if args.model_path:
        cfg.setdefault("llm", {})["model_path"] = args.model_path
        cfg["llm"]["local_files_only"] = True
        cfg["llm"]["backend"] = cfg["llm"].get("backend") or "transformers"

    agent = build_agent_from_config(cfg)
    result = agent.solve(text.strip())

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("=" * 60)
        print("ALPHA-MATH")
        print("=" * 60)
        print(f"Backend : {result.backend} / {result.model}")
        print(f"Attempts: {len(result.attempts)}")
        print(f"Success : {result.success}")
        print(f"ANSWER  : {result.answer}")
        if result.attempts and result.attempts[-1].code:
            print("-" * 60)
            print("Last code:")
            print(result.attempts[-1].code)
        if result.attempts and result.attempts[-1].sandbox and result.attempts[-1].sandbox.error:
            print("-" * 60)
            print("Error:", result.attempts[-1].sandbox.error)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.path.insert(0, str(project_root()))
    raise SystemExit(main())
