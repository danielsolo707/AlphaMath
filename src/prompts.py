"""Prompt templates aligned with the published Kaggle ALPHA-MATH notebook.

Kaggle kernel (danielsolo1770/alpha-math) uses a strict "Python generator" style:
emit a single sympy script, no free-form natural-language solution dump.
"""

from __future__ import annotations

# Matches the Kaggle notebook system prompt (tool-integrated math solving).
SYSTEM_PROMPT = """You are a strict Mathematical Python Generator. Your ONLY job is to write a Python script using sympy to solve the user's problem.
CRITICAL RULES:
1. Do NOT explain the math in natural language. Do NOT write LaTeX.
2. Forbidden in the response: \\[ \\] \\( \\) $...$ \\frac \\text prose paragraphs.
3. You MUST enclose your full Python code strictly inside a single ```python and ``` block.
4. The script MUST end by printing only the final integer result.
5. Prefer exact arithmetic (integers / sympy). Avoid floats unless necessary.
6. Libraries already available in the sandbox — do NOT import os/sys/subprocess:
   math, sympy (as sympy or sp), itertools, functools, collections, fractions, decimal, numpy (np)
7. Optionally set ANSWER = <int> before printing it.
8. For AIME-style contests the official answer is an integer in 0..999. If the
   natural answer is larger, print answer % 1000 (last three digits) unless the
   problem already asks for a residue or remainder.
9. Keep scripts self-contained: define helpers before use, avoid undefined names,
   and never leave incomplete blocks or trailing prose after the code fence.
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
    extra = ""
    fb_lower = (feedback or "").lower()
    if "indentationerror" in fb_lower or "indent" in fb_lower:
        extra = (
            "\nIndentation tip: use only 4-space indents, never mix tabs, and do not "
            "leave residual outer indentation on every line of the fence body.\n"
        )
    elif "syntaxerror" in fb_lower:
        extra = (
            "\nSyntax tip: emit a complete, runnable script — balanced parentheses, "
            "no trailing prose inside the fence, define names before use.\n"
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "The previous script failed during restricted execution.\n\n"
                f"EXECUTION_FEEDBACK:\n{feedback.strip()}\n"
                f"{extra}\n"
                "Diagnose the failure, preserve the original problem requirements, "
                "and output the complete corrected script inside exactly one "
                "```python ... ``` block. Print only the final integer."
            ),
        }
    )
    return messages


def sanitize_python_code(code: str) -> str:
    """Normalize model-emitted code to reduce IndentationError / fence noise."""
    import re
    import textwrap

    cleaned = code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    # Drop accidental fence leftovers without strip() — strip() would destroy
    # relative indentation by removing only the first line's leading spaces.
    cleaned = re.sub(r"^```(?:python)?[ \t]*\n?", "", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```[ \t]*$", "", cleaned, count=1)
    cleaned = textwrap.dedent(cleaned)
    # Trim blank lines only; keep internal structure.
    lines = cleaned.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _looks_like_latex_or_prose(text: str) -> bool:
    """Reject LaTeX / natural-language dumps that are not executable Python."""
    import re

    sample = text.strip()
    if not sample:
        return True
    latex_hits = len(re.findall(r"\\\[|\\\]|\\\(|\\\)|\\frac|\\text|\\begin\{|\\end\{", sample))
    if latex_hits >= 1:
        return True
    # Dollar-math heavy blobs without Python keywords.
    if sample.count("$") >= 2 and not re.search(r"\b(print|import|def|for|while|if)\b", sample):
        return True
    return False


def extract_code_block(text: str) -> str | None:
    """Extract a fenced python code block.

    Heuristic (unfenced) extraction is intentionally conservative: AIME-style
    model dumps often contain LaTeX equations with ``=`` that must NOT be treated
    as Python (that was a major SyntaxError source on the 90-problem run).
    """
    import re

    if not text:
        return None
    # Prefer explicit python fences.
    fence = re.search(r"```python[ \t]*\r?\n([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        code = sanitize_python_code(fence.group(1))
        if code and not _looks_like_latex_or_prose(code):
            return code
    # Generic fence only if body looks like python, not latex.
    fence_any = re.search(r"```[ \t]*\r?\n([\s\S]*?)```", text)
    if fence_any:
        code = sanitize_python_code(fence_any.group(1))
        if code and not _looks_like_latex_or_prose(code):
            # Require at least one python-ish token.
            if re.search(r"\b(print|import|def|for|while|ANSWER)\b|^\s*[A-Za-z_][\w]*\s*=", code, re.M):
                return code
    # Same-line fence: ```python code```
    fence_inline = re.search(r"```(?:python)?[ \t]+([\s\S]*?)```", text, re.IGNORECASE)
    if fence_inline:
        code = sanitize_python_code(fence_inline.group(1))
        if code and not _looks_like_latex_or_prose(code):
            return code
    # Conservative unfenced fallback: only pure-looking short scripts.
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "print(", "ANSWER", "#")):
            start = i
            break
    if start is None:
        return None
    code = sanitize_python_code("\n".join(lines[start:]))
    if not code or _looks_like_latex_or_prose(code):
        return None
    return code
