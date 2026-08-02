from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent import MathAgent
from src.evaluate import evaluate
from src.llm import MockLLM


class EvaluateCheckpointTests(unittest.TestCase):
    def test_checkpoint_skips_completed(self) -> None:
        problems = [
            {"id": "a", "problem": "1+1", "answer": 2},
            {"id": "b", "problem": "2+2", "answer": 4},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint.json"
            # Seed a completed first problem
            ckpt.write_text(
                json.dumps(
                    {
                        "per_problem": [
                            {
                                "id": "a",
                                "problem": "1+1",
                                "gold": 2,
                                "gold_cmp": 2,
                                "pred": 2,
                                "correct": True,
                                "success": True,
                                "attempts": 1,
                                "elapsed_s": 0.1,
                                "model": "mock",
                                "backend": "mock",
                                "votes": [2],
                                "vote_counts": {"2": 1},
                                "vote_agreement": 1.0,
                                "vote_tied": False,
                                "defaulted": False,
                                "budget_exhausted": False,
                                "execution_success": True,
                                "sandbox_failures": 0,
                                "sandbox_timeouts": 0,
                                "failure_types": [],
                                "tags": [],
                                "difficulty": None,
                                "source": "t",
                                "trace": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            agent = MathAgent(MockLLM(), majority_vote_k=1, max_corrections=0, sandbox_timeout=2)
            summary = evaluate(problems, agent, verbose=False, checkpoint_path=ckpt)
            self.assertEqual(summary["total"], 2)
            ids = [row["id"] for row in summary["per_problem"]]
            self.assertEqual(ids, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
