from __future__ import annotations

import unittest

from src.agent import MathAgent
from src.llm import BaseLLM, LLMResponse


class ScriptedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        self.calls.append((messages, kwargs))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return LLMResponse(self.responses[index], "scripted", "test")


class AgentTests(unittest.TestCase):
    def make_agent(self, llm: BaseLLM, **kwargs) -> MathAgent:
        return MathAgent(
            llm,
            majority_vote_k=1,
            max_corrections=2,
            sandbox_timeout=2,
            memory_limit_mb=None,
            clamp_answer=False,
            default_answer_on_fail=None,
            **kwargs,
        )

    def test_repair_preserves_problem_and_previous_response(self) -> None:
        llm = ScriptedLLM(
            [
                "```python\nprint(undefined_name)\n```",
                "```python\nANSWER = 42\nprint(ANSWER)\n```",
            ]
        )
        result = self.make_agent(llm).solve("Compute six times seven.")
        self.assertTrue(result.success)
        self.assertEqual(result.answer, 42)
        self.assertEqual(len(llm.calls), 2)
        correction_messages = llm.calls[1][0]
        self.assertEqual([m["role"] for m in correction_messages], ["system", "user", "assistant", "user"])
        self.assertIn("Compute six times seven", correction_messages[1]["content"])
        self.assertIn("undefined_name", correction_messages[2]["content"])
        self.assertIn("NameError", correction_messages[3]["content"])

    def test_two_corrections_means_three_total_attempts(self) -> None:
        llm = ScriptedLLM(["```python\nprint(missing)\n```"])
        result = self.make_agent(llm).solve("A deliberately failing problem")
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual([a.correction_index for a in result.attempts], [0, 1, 2])

    def test_strict_majority_stops_early(self) -> None:
        llm = ScriptedLLM(["```python\nANSWER = 7\nprint(ANSWER)\n```"])
        agent = MathAgent(
            llm,
            majority_vote_k=3,
            max_corrections=0,
            sandbox_timeout=2,
            memory_limit_mb=None,
            clamp_answer=False,
            early_stop_majority=True,
        )
        result = agent.solve("Return seven")
        self.assertEqual(result.answer, 7)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(result.meta["stopped_early"])
        self.assertEqual(result.meta["vote_agreement"], 1.0)

    def test_seeds_are_deterministic_and_distinct(self) -> None:
        llm = ScriptedLLM(["```python\nANSWER = 3\nprint(ANSWER)\n```"])
        agent = MathAgent(
            llm,
            majority_vote_k=2,
            max_corrections=0,
            sandbox_timeout=2,
            memory_limit_mb=None,
            clamp_answer=False,
            early_stop_majority=False,
            base_seed=100,
        )
        agent.solve("Return three")
        self.assertEqual([call[1]["seed"] for call in llm.calls], [100, 1100])


if __name__ == "__main__":
    unittest.main()
