"""ALPHA-MATH agent: math LLM plans → sandbox executes → verify → retry / majority vote.

Behavior mirrors the published Kaggle kernel (danielsolo1770/alpha-math):
  - num_generations independent samples (majority_vote_k)
  - max_correction_attempts sandbox self-repairs per sample (max_attempts)
  - temperature / top_p sampling for diversity
  - integer answer extraction from executed code output
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.llm import BaseLLM, LLMResponse
from src.prompts import build_messages, extract_code_block
from src.sandbox import SandboxResult, run_code
from src.utils import extract_integer_answer


@dataclass
class Attempt:
    index: int
    llm_text: str
    code: str | None
    sandbox: SandboxResult | None
    answer: int | None
    feedback: str | None = None
    vote_round: int = 0


@dataclass
class AgentResult:
    problem: str
    answer: int | None
    success: bool
    attempts: list[Attempt] = field(default_factory=list)
    model: str = ""
    backend: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "answer": self.answer,
            "success": self.success,
            "model": self.model,
            "backend": self.backend,
            "num_attempts": len(self.attempts),
            "attempts": [
                {
                    "index": a.index,
                    "vote_round": a.vote_round,
                    "code": a.code,
                    "answer": a.answer,
                    "sandbox_ok": a.sandbox.ok if a.sandbox else False,
                    "sandbox_error": a.sandbox.error if a.sandbox else None,
                    "stdout": a.sandbox.stdout if a.sandbox else "",
                    "feedback": a.feedback,
                }
                for a in self.attempts
            ],
            "meta": self.meta,
        }


class MathAgent:
    def __init__(
        self,
        llm: BaseLLM,
        *,
        max_attempts: int = 2,
        temperature: float = 0.7,
        top_p: float = 0.9,
        sandbox_timeout: float = 5.0,
        allowed_modules: list[str] | None = None,
        answer_min: int = 0,
        answer_max: int = 999,
        clamp_answer: bool = True,
        majority_vote_k: int = 3,
        verbose_prompts: bool = False,
        default_answer_on_fail: int | None = 0,
    ) -> None:
        self.llm = llm
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.top_p = top_p
        self.sandbox_timeout = sandbox_timeout
        self.allowed_modules = allowed_modules
        self.answer_min = answer_min
        self.answer_max = answer_max
        self.clamp_answer = clamp_answer
        self.majority_vote_k = max(1, int(majority_vote_k))
        self.verbose_prompts = verbose_prompts
        # Kaggle notebook returns 0 when every sample fails (format-safe).
        self.default_answer_on_fail = default_answer_on_fail

    def _normalize(self, answer: int | None) -> int | None:
        if answer is None:
            return None
        if self.clamp_answer:
            if answer < 0:
                return None
            # AIME / many AIMO problems: report last three digits
            if self.answer_max == 999:
                return int(answer) % 1000
            return max(self.answer_min, min(self.answer_max, int(answer)))
        return int(answer)

    def _one_pass(
        self, problem: str, feedback: str | None, vote_round: int, index: int
    ) -> tuple[Attempt, LLMResponse]:
        messages = build_messages(
            problem, feedback, verbose_system=self.verbose_prompts
        )
        resp: LLMResponse = self.llm.complete(
            messages,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        code = extract_code_block(resp.text)
        sandbox: SandboxResult | None = None
        answer: int | None = None
        fb: str | None = None

        if not code:
            fb = (
                "Your response did not contain a valid code block. "
                "Remember, you MUST wrap your code in ```python and ```."
            )
        else:
            sandbox = run_code(
                code,
                timeout_seconds=self.sandbox_timeout,
                allowed_modules=self.allowed_modules,
            )
            if sandbox.ok and sandbox.answer is not None:
                answer = self._normalize(sandbox.answer)
            elif sandbox.ok:
                answer = self._normalize(
                    extract_integer_answer(
                        sandbox.stdout or resp.text,
                        self.answer_min,
                        self.answer_max,
                        clamp=self.clamp_answer,
                    )
                )
                if answer is None:
                    fb = (
                        "Code ran but no integer result was produced. "
                        "Print only the final integer."
                    )
            else:
                err = sandbox.error or "Unknown execution error"
                fb = err

        attempt = Attempt(
            index=index,
            llm_text=resp.text,
            code=code,
            sandbox=sandbox,
            answer=answer,
            feedback=fb,
            vote_round=vote_round,
        )
        return attempt, resp

    def solve(self, problem: str) -> AgentResult:
        attempts: list[Attempt] = []
        votes: list[int] = []
        last_model = ""
        last_backend = ""
        idx = 0

        for vote_round in range(1, self.majority_vote_k + 1):
            feedback: str | None = None
            answer: int | None = None

            for _ in range(self.max_attempts):
                idx += 1
                attempt, resp = self._one_pass(problem, feedback, vote_round, idx)
                attempts.append(attempt)
                last_model = resp.model or last_model
                last_backend = resp.backend or last_backend
                if attempt.answer is not None and attempt.sandbox and attempt.sandbox.ok:
                    answer = attempt.answer
                    break
                feedback = attempt.feedback or "Unknown failure; revise the solution."

            if answer is not None:
                votes.append(answer)

        final: int | None = None
        success = False
        if votes:
            final, _count = Counter(votes).most_common(1)[0]
            success = True
        elif self.default_answer_on_fail is not None:
            final = int(self.default_answer_on_fail)

        return AgentResult(
            problem=problem,
            answer=final,
            success=success,
            attempts=attempts,
            model=str(last_model),
            backend=str(last_backend),
            meta={
                "votes": votes,
                "majority_vote_k": self.majority_vote_k,
                "vote_counts": dict(Counter(votes)),
                "defaulted": bool(not votes and self.default_answer_on_fail is not None),
            },
        )
