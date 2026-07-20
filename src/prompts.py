"""Prompt templates for DeepSeek-Math tool-integrated reasoning + SymPy code."""

from __future__ import annotations

SYSTEM_PROMPT = """You are ALPHA-MATH, built on a math-specialized open-weight language model
(DeepSeek-Math style tool-integrated reasoning).

Your job: solve olympiad / AIME-style problems where the final answer is an integer.

Method (required):
1. Write a short plan.
2. Implement the plan as Python that the sandbox will execute.
3. Use exact arithmetic (prefer sympy / integers). Avoid floats unless necessary.
4. Libraries already available — do NOT import anything:
   math, sympy (as sympy or sp), itertools, functools, collections, fractions, decimal, numpy (np)
5. Assign the final integer to ANSWER and print it.

Competition rule:
- Final answers are non-negative integers. When a problem asks for the last three digits
  or an AIME answer, report a value in 0..999.

Output format (exactly):
REASONING:
<brief plan>

CODE:
```python
# computation
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
                "PREVIOUS ATTEMPT FAILED. Feedback from the Python verifier:",
                feedback.strip(),
                "Fix the code or try a different correct method. Still set ANSWER and print it.",
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
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("ANSWER", "#", "print", "n =", "x =", "def ", "for ", "while ")):
            start = i
            break
        if "=" in stripped and not stripped.lower().startswith("reasoning"):
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()
    return None
