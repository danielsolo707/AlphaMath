from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dataio import load_labeled_problems
from src.utils import extract_integer_answer, validate_config


class DataAndParsingTests(unittest.TestCase):
    def test_answer_marker_beats_other_numbers(self) -> None:
        self.assertEqual(extract_integer_answer("tried 12 then FINAL_ANSWER: 7"), 7)

    def test_boxed_answer(self) -> None:
        self.assertEqual(extract_integer_answer(r"Therefore \\boxed{314}."), 314)

    def test_negative_rejected_when_clamping(self) -> None:
        self.assertIsNone(extract_integer_answer("ANSWER: -2", clamp=True))

    def test_json_and_csv_benchmark_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "bench.json"
            json_path.write_text(
                json.dumps([{"id": "a", "problem": "2+2", "answer": 4}]),
                encoding="utf-8",
            )
            csv_path = root / "bench.csv"
            csv_path.write_text("id,question,gold\nb,3+3,6\n", encoding="utf-8")
            self.assertEqual(load_labeled_problems(json_path)[0]["answer"], 4)
            self.assertEqual(load_labeled_problems(csv_path)[0]["problem"], "3+3")

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "majority_vote_k"):
            validate_config(
                {
                    "agent": {"majority_vote_k": 0},
                    "sandbox": {"timeout_seconds": 1, "max_output_chars": 1000, "max_source_chars": 1000},
                }
            )


if __name__ == "__main__":
    unittest.main()
