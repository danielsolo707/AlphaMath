"""Prompt templates for mathematical reasoning + Python tool use."""

from __future__ import annotations

SYSTEM_PROMPT = """You are ALPHA-MATH, a careful mathematical reasoning agent for olympiad-style problems.

Rules:
1. Think step by step, but the final computational work MUST be done in Python.
2. Use only: math, sympy (as sympy or sp), itertools, functools, collections, fractions, decimal, numpy (as numpy or np).
3. Do NOT import modules. The sandbox already provides the allowed libraries.
4. Assign the final integer answer to a variable named ANSWER.
5. Also print(ANSWER) so the verifier can capture stdout.
6. Answers for competition problems are non-negative integers (AIME-style often in 000–999).
7. Prefer exact sympy arithmetic over floating point.
8. If you are unsure, still produce best-effort executable code.

Output format (exactly):
REASONING:
<short natural-language plan>

CODE:
```python
# your code
ANSWER = ...
print(ANSWER)
```
"""


def build_user_prompt(problem: str, feedback: str | None = None) -> str:
    parts = [
        "Solve the following problem. Produce REASONING + executable CODE as specified.",
        "",
        "PROBLEM:",
        problem.strip(),
    ]
    if feedback:
        parts.extend(
            [
                "",
                "PREVIOUS ATTEMPT FAILED. Feedback from the verifier:",
                feedback.strip(),
                "Revise your approach and try a different method if needed.",
            ]
        )
    return "\n".join(parts)


def build_messages(problem: str, feedback: str | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(problem, feedback)},
    ]


def extract_code_block(text: str) -> str | None:
    """Extract the first fenced python code block, or a heuristic fallback."""
    import re

    if not text:
        return None
    fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # Heuristic: from first assignment-looking line to end
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(("ANSWER", "import", "from ", "#", "print", "n =", "x =")) or "=" in line:
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()
    return None
