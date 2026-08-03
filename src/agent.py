"""ALPHA-MATH agent: math LLM plans → sandbox executes → verify → retry / majority vote.

Behavior mirrors the published Kaggle kernel (danielsolo1770/alpha-math):
  - num_generations independent samples (majority_vote_k)
  - one initial generation plus max_corrections stateful repairs per sample
  - temperature / top_p sampling for diversity
  - integer answer extraction from executed code output
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import time
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
    correction_index: int = 0
    seed: int | None = None


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
                    "correction_index": a.correction_index,
                    "seed": a.seed,
                    "llm_text": a.llm_text,
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
        max_corrections: int = 2,
        max_attempts: int | None = None,
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
        max_output_chars: int = 8000,
        max_source_chars: int = 50_000,
        memory_limit_mb: int | None = 1536,
        time_budget_seconds: float | None = None,
        base_seed: int = 2026,
        early_stop_majority: bool = True,
    ) -> None:
        self.llm = llm
        # ``max_attempts`` is accepted for backward compatibility. Historically
        # it was documented as corrections but implemented as total attempts.
        if max_attempts is not None:
            max_corrections = max_attempts
        self.max_corrections = max(0, int(max_corrections))
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
        self.max_output_chars = max(256, int(max_output_chars))
        self.max_source_chars = max(1000, int(max_source_chars))
        self.memory_limit_mb = memory_limit_mb
        self.time_budget_seconds = time_budget_seconds
        self.base_seed = int(base_seed)
        self.early_stop_majority = bool(early_stop_majority)

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
        self,
        problem: str,
        feedback: str | None,
        previous_response: str | None,
        previous_code: str | None,
        vote_round: int,
        correction_index: int,
        index: int,
    ) -> tuple[Attempt, LLMResponse]:
        messages = build_messages(
            problem,
            feedback,
            previous_response=previous_response,
            previous_code=previous_code,
            verbose_system=self.verbose_prompts,
        )
        seed = self.base_seed + (vote_round - 1) * 1000 + correction_index
        try:
            resp: LLMResponse = self.llm.complete(
                messages,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=seed,
            )
        except Exception as exc:  # surface generation failures in the audit trace
            resp = LLMResponse(
                text="",
                model=type(self.llm).__name__,
                backend="generation_error",
                raw=None,
            )
            attempt = Attempt(
                index=index,
                llm_text="",
                code=None,
                sandbox=None,
                answer=None,
                feedback=f"LLMGenerationError: {type(exc).__name__}: {exc}",
                vote_round=vote_round,
                correction_index=correction_index,
                seed=seed,
            )
            return attempt, resp
        code = extract_code_block(resp.text)
        sandbox: SandboxResult | None = None
        answer: int | None = None
        fb: str | None = None

        if not code:
            fb = (
                "Your response did not contain a valid ```python``` code block. "
                "Do NOT write LaTeX or natural-language math. "
                "Output only one fenced Python script that prints the final integer."
            )
        else:
            # Cheap syntax gate before paying sandbox process overhead.
            try:
                compile(code, "<model_code>", "exec")
            except SyntaxError as syn_exc:
                fb = (
                    f"SyntaxError before execution: {syn_exc}\n"
                    "Rewrite as pure Python (no LaTeX). Use ```python``` fences only."
                )
                attempt = Attempt(
                    index=index,
                    llm_text=resp.text,
                    code=code,
                    sandbox=None,
                    answer=None,
                    feedback=fb,
                    vote_round=vote_round,
                    correction_index=correction_index,
                    seed=seed,
                )
                return attempt, resp
            sandbox = run_code(
                code,
                timeout_seconds=self.sandbox_timeout,
                allowed_modules=self.allowed_modules,
                max_output_chars=self.max_output_chars,
                max_source_chars=self.max_source_chars,
                memory_limit_mb=self.memory_limit_mb,
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
                if sandbox.stdout:
                    err += f"\nCAPTURED_STDOUT:\n{sandbox.stdout[-2000:]}"
                fb = err

        attempt = Attempt(
            index=index,
            llm_text=resp.text,
            code=code,
            sandbox=sandbox,
            answer=answer,
            feedback=fb,
            vote_round=vote_round,
            correction_index=correction_index,
            seed=seed,
        )
        return attempt, resp

    def solve(self, problem: str) -> AgentResult:
        attempts: list[Attempt] = []
        votes: list[int] = []
        last_model = ""
        last_backend = ""
        idx = 0
        started = time.perf_counter()
        stopped_early = False
        budget_exhausted = False

        for vote_round in range(1, self.majority_vote_k + 1):
            feedback: str | None = None
            answer: int | None = None
            previous_response: str | None = None
            previous_code: str | None = None

            for correction_index in range(self.max_corrections + 1):
                if (
                    self.time_budget_seconds is not None
                    and time.perf_counter() - started >= self.time_budget_seconds
                ):
                    budget_exhausted = True
                    break
                idx += 1
                attempt, resp = self._one_pass(
                    problem,
                    feedback,
                    previous_response,
                    previous_code,
                    vote_round,
                    correction_index,
                    idx,
                )
                attempts.append(attempt)
                last_model = resp.model or last_model
                last_backend = resp.backend or last_backend
                if attempt.answer is not None and attempt.sandbox and attempt.sandbox.ok:
                    answer = attempt.answer
                    break
                feedback = attempt.feedback or "Unknown failure; revise the solution."
                previous_response = resp.text
                previous_code = attempt.code

            if answer is not None:
                votes.append(answer)
                current_count = Counter(votes).most_common(1)[0][1]
                if self.early_stop_majority and current_count > self.majority_vote_k // 2:
                    stopped_early = True
                    break
            if budget_exhausted:
                break

        final: int | None = None
        success = False
        vote_counts = Counter(votes)
        tied = False
        agreement = 0.0
        if votes:
            final, top_count = vote_counts.most_common(1)[0]
            tied = sum(1 for count in vote_counts.values() if count == top_count) > 1
            agreement = top_count / len(votes)
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
                "vote_counts": dict(vote_counts),
                "vote_agreement": round(agreement, 4),
                "vote_tied": tied,
                "strict_majority": bool(votes and max(vote_counts.values()) > self.majority_vote_k // 2),
                "rounds_completed": len(votes),
                "stopped_early": stopped_early,
                "budget_exhausted": budget_exhausted,
                "elapsed_s": round(time.perf_counter() - started, 4),
                "base_seed": self.base_seed,
                "max_corrections": self.max_corrections,
                "defaulted": bool(not votes and self.default_answer_on_fail is not None),
            },
        )
