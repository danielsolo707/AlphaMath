from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.agent import MathAgent
from src.kaggle_submit import run_submission
from src.llm import BaseLLM, LLMResponse
from src.reporting import file_fingerprint, write_evaluation_artifacts


class CountingLLM(BaseLLM):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse("```python\nANSWER = 9\nprint(ANSWER)\n```", "counting", "test")


class ReportingAndSubmissionTests(unittest.TestCase):
    def test_dataset_fingerprint_and_report_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "bench.json"
            dataset.write_text('[{"problem":"x","answer":1}]', encoding="utf-8")
            fingerprint = file_fingerprint(dataset)
            self.assertEqual(len(fingerprint["sha256"]), 64)
            summary = {
                "backend": "mock",
                "model": "mock",
                "total": 0,
                "accuracy": 0.0,
                "metrics": {},
                "per_problem": [],
            }
            paths = write_evaluation_artifacts(summary, {"llm": {"backend": "mock"}}, root / "out", dataset_path=dataset)
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertIn("MOCK PIPELINE TEST", paths["markdown"].read_text(encoding="utf-8"))

    def test_submission_schema_detection_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_csv = root / "test.csv"
            test_csv.write_text("id,problem\na,anything\nb,anything else\n", encoding="utf-8")
            (root / "sample_submission.csv").write_text("id,answer\na,0\nb,0\n", encoding="utf-8")
            output = root / "submission.csv"
            trace = root / "trace.json"
            llm = CountingLLM()
            agent = MathAgent(
                llm,
                majority_vote_k=1,
                max_corrections=0,
                sandbox_timeout=2,
                memory_limit_mb=None,
                clamp_answer=False,
            )
            run_submission(
                "configs/smoke_mock.yaml",
                test_csv=test_csv,
                out_csv=output,
                agent=agent,
                allow_mock=False,
                trace_json=trace,
            )
            self.assertEqual(llm.calls, 2)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(handle.closed, False)
            self.assertEqual(list(rows[0]), ["id", "answer"])
            self.assertEqual(json.loads(trace.read_text(encoding="utf-8"))["completed"], 2)

            run_submission(
                "configs/smoke_mock.yaml",
                test_csv=test_csv,
                out_csv=output,
                agent=agent,
                allow_mock=False,
                trace_json=trace,
                resume=True,
            )
            self.assertEqual(llm.calls, 2, "resume should not regenerate completed rows")


if __name__ == "__main__":
    unittest.main()
