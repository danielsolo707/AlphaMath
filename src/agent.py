"""ALPHA-MATH agent: generate code → execute in sandbox → verify → retry."""

from __future__ import annotations

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
        max_attempts: int = 3,
        temperature: float = 0.2,
        sandbox_timeout: float = 5.0,
        allowed_modules: list[str] | None = None,
        answer_min: int = 0,
        answer_max: int = 999,
        clamp_answer: bool = False,
    ) -> None:
        self.llm = llm
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.sandbox_timeout = sandbox_timeout
        self.allowed_modules = allowed_modules
        self.answer_min = answer_min
        self.answer_max = answer_max
        self.clamp_answer = clamp_answer

    def solve(self, problem: str) -> AgentResult:
        attempts: list[Attempt] = []
        feedback: str | None = None
        last_model = ""
        last_backend = ""

        for i in range(1, self.max_attempts + 1):
            messages = build_messages(problem, feedback)
            resp: LLMResponse = self.llm.complete(
                messages, temperature=self.temperature
            )
            last_model, last_backend = resp.model, resp.backend
            code = extract_code_block(resp.text)
            sandbox: SandboxResult | None = None
            answer: int | None = None
            fb: str | None = None

            if not code:
                fb = "No Python code block found. Emit CODE in a ```python fenced block and set ANSWER."
            else:
                sandbox = run_code(
                    code,
                    timeout_seconds=self.sandbox_timeout,
                    allowed_modules=self.allowed_modules,
                )
                if sandbox.ok and sandbox.answer is not None:
                    answer = sandbox.answer
                    if self.clamp_answer:
                        parsed = extract_integer_answer(
                            str(answer),
                            self.answer_min,
                            self.answer_max,
                            clamp=True,
                        )
                        answer = parsed
                elif sandbox.ok:
                    # Try parse from stdout / free text
                    answer = extract_integer_answer(
                        sandbox.stdout or resp.text,
                        self.answer_min,
                        self.answer_max,
                        clamp=self.clamp_answer,
                    )
                    if answer is None:
                        fb = (
                            "Code ran but no integer ANSWER was produced. "
                            "Set ANSWER = <int> and print(ANSWER)."
                        )
                else:
                    fb = f"Execution failed: {sandbox.error}"

            attempt = Attempt(
                index=i,
                llm_text=resp.text,
                code=code,
                sandbox=sandbox,
                answer=answer,
                feedback=fb,
            )
            attempts.append(attempt)

            if answer is not None and (not self.clamp_answer or self.answer_min <= answer <= self.answer_max or True):
                # Accept any integer for demo; competition mode can tighten later
                if sandbox and sandbox.ok and answer is not None:
                    return AgentResult(
                        problem=problem,
                        answer=answer,
                        success=True,
                        attempts=attempts,
                        model=last_model,
                        backend=last_backend,
                    )

            feedback = fb or "Unknown failure; revise the solution."

        final_answer = next((a.answer for a in reversed(attempts) if a.answer is not None), None)
        return AgentResult(
            problem=problem,
            answer=final_answer,
            success=final_answer is not None and attempts[-1].sandbox is not None and attempts[-1].sandbox.ok,
            attempts=attempts,
            model=last_model,
            backend=last_backend,
            meta={"exhausted_attempts": True},
        )
