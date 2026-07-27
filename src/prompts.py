"""Prompt templates aligned with the published Kaggle ALPHA-MATH notebook.

Kaggle kernel (danielsolo1770/alpha-math) uses a strict "Python generator" style:
emit a single sympy script, no free-form natural-language solution dump.
"""

from __future__ import annotations

# Matches the Kaggle notebook system prompt (tool-integrated math solving).
SYSTEM_PROMPT = """You are a strict Mathematical Python Generator. Your ONLY job is to write a Python script using sympy to solve the user's problem.
CRITICAL RULES:
1. Do NOT explain the math in natural language.
2. You MUST enclose your full Python code strictly inside a single ```python and ``` block.
3. The script MUST end by printing only the final integer result.
4. Prefer exact arithmetic (integers / sympy). Avoid floats unless necessary.
5. Libraries already available in the sandbox — do NOT import os/sys/subprocess:
   math, sympy (as sympy or sp), itertools, functools, collections, fractions, decimal, numpy (np)
6. Optionally set ANSWER = <int> before printing it.
"""

# Slightly richer prompt for portfolio / local demos (same tool loop).
SYSTEM_PROMPT_VERBOSE = """You are ALPHA-MATH, a math-specialized open-weight model with a Python sandbox.

Solve olympiad / AIME-style problems where the final answer is an integer.

Method:
1. Short plan (optional).
2. Implement as Python that the sandbox will execute.
3. Exact arithmetic (sympy / integers). Libraries preloaded — do not import os/sys.
4. Assign ANSWER = <int> and print it.
5. Competition answers are non-negative integers; AIME-style often wants last three digits (0..999).

Output format:
REASONING:
<brief plan>

CODE:
```python
# computation
ANSWER = ...
print(ANSWER)
```
"""


def build_user_prompt(problem: str) -> str:
    """Build the original problem turn.

    The problem is intentionally kept as its own message so correction turns can
    preserve the complete conversation instead of asking the model to repair an
    unknown script.
    """
    return f"Write a Python script using sympy to solve this problem: {problem.strip()}"


def build_messages(
    problem: str,
    feedback: str | None = None,
    *,
    previous_response: str | None = None,
    previous_code: str | None = None,
    verbose_system: bool = False,
) -> list[dict[str, str]]:
    """Build a first-attempt or stateful correction conversation."""
    system = SYSTEM_PROMPT_VERBOSE if verbose_system else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(problem)},
    ]
    if not feedback:
        return messages

    prior = previous_response
    if not prior and previous_code:
        prior = f"```python\n{previous_code.strip()}\n```"
    messages.append(
        {
            "role": "assistant",
            "content": prior or "```python\n# Previous response was not executable.\n```",
        }
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "The previous script failed during restricted execution.\n\n"
                f"EXECUTION_FEEDBACK:\n{feedback.strip()}\n\n"
                "Diagnose the failure, preserve the original problem requirements, "
                "and output the complete corrected script inside exactly one "
                "```python ... ``` block. Print only the final integer."
            ),
        }
    )
    return messages


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
