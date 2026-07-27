from __future__ import annotations

import importlib.util
import time
import unittest

from src.sandbox import run_code


class SandboxTests(unittest.TestCase):
    def test_executes_integer_math(self) -> None:
        result = run_code("ANSWER = sum(range(11))\nprint(ANSWER)", timeout_seconds=2)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.answer, 55)

    def test_hard_timeout_kills_worker(self) -> None:
        started = time.perf_counter()
        result = run_code("while True:\n    pass", timeout_seconds=0.2)
        wall = time.perf_counter() - started
        self.assertTrue(result.timed_out)
        self.assertEqual(result.error_type, "TimeoutError")
        self.assertLess(wall, 1.5)
        self.assertTrue(result.meta["worker_killed"])

    def test_output_limit(self) -> None:
        result = run_code(
            "for i in range(1000):\n    print(i)",
            timeout_seconds=2,
            max_output_chars=80,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "OutputLimitError")
        self.assertTrue(result.output_truncated)

    def test_blocks_dunder_escape(self) -> None:
        result = run_code("print((1).__class__)", timeout_seconds=2)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "SecurityError")

    def test_allowed_import_is_rewritten(self) -> None:
        result = run_code("import math\nANSWER = math.gcd(21, 14)\nprint(ANSWER)", timeout_seconds=2)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.answer, 7)
        self.assertTrue(result.meta["sanitized"])

    def test_optional_dependency_is_actionable(self) -> None:
        result = run_code("ANSWER = int(sp.totient(10))\nprint(ANSWER)", timeout_seconds=4)
        if importlib.util.find_spec("sympy") is None:
            self.assertEqual(result.error_type, "DependencyError")
            self.assertIn("requirements", result.error or "")
        else:
            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.answer, 4)


if __name__ == "__main__":
    unittest.main()
