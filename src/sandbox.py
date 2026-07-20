"""Restricted Python + SymPy execution sandbox for System-2 verification.

Not a full security boundary (no OS-level isolation). Intended for:
  - local portfolio demos
  - trusted model-generated math code
  - unit evaluation

For production / untrusted multi-tenant use, run inside a container or
Firecracker microVM with no network and a hard cgroup memory limit.
"""

from __future__ import annotations

import ast
import math
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any

# Optional heavy deps imported lazily-safe
try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    # Attribute access is allowed (sympy/math), but we block dunder tricks via name check
)

FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "pickle",
    "requests",
    "httpx",
    "ctypes",
    "multiprocessing",
    "threading",
}


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    value: Any = None
    answer: int | None = None
    error: str | None = None
    timed_out: bool = False
    elapsed_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class _SafePrinter:
    """Capture print() into a buffer."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        sep = kwargs.get("sep", " ")
        self.lines.append(sep.join(str(a) for a in args))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise SecurityError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SecurityError(f"Disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SecurityError(f"Disallowed dunder attribute: {node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                raise SecurityError(f"Disallowed call: {func.id}")


class SecurityError(Exception):
    pass


def _build_globals(allowed_modules: list[str] | None = None) -> dict[str, Any]:
    allowed = set(allowed_modules or ["math", "sympy", "itertools", "functools", "collections"])
    g: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "pow": pow,
            "print": None,  # replaced per-run
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }
    }
    if "math" in allowed:
        g["math"] = math
    if "sympy" in allowed and sp is not None:
        g["sympy"] = sp
        g["sp"] = sp
    if "numpy" in allowed and np is not None:
        g["numpy"] = np
        g["np"] = np
    if "itertools" in allowed:
        import itertools

        g["itertools"] = itertools
    if "functools" in allowed:
        import functools

        g["functools"] = functools
    if "collections" in allowed:
        import collections

        g["collections"] = collections
    if "fractions" in allowed:
        import fractions

        g["fractions"] = fractions
    if "decimal" in allowed:
        import decimal

        g["decimal"] = decimal
    return g


def _execute(code: str, allowed_modules: list[str] | None) -> SandboxResult:
    printer = _SafePrinter()
    try:
        tree = ast.parse(code, mode="exec")
        _validate_ast(tree)
    except SecurityError as e:
        return SandboxResult(ok=False, error=f"SecurityError: {e}")
    except SyntaxError as e:
        return SandboxResult(ok=False, error=f"SyntaxError: {e}")

    g = _build_globals(allowed_modules)
    g["__builtins__"]["print"] = printer  # type: ignore[index]
    local: dict[str, Any] = {}

    try:
        compiled = compile(tree, filename="<sandbox>", mode="exec")
        exec(compiled, g, local)  # noqa: S102 — intentional, AST-gated sandbox
    except Exception as e:  # noqa: BLE001 — surface model errors to agent
        return SandboxResult(
            ok=False,
            stdout=printer.text,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )

    # Prefer explicit ANSWER / answer / result bindings
    value = None
    for key in ("ANSWER", "answer", "result", "RESULT", "final", "FINAL"):
        if key in local:
            value = local[key]
            break
    if value is None and printer.lines:
        value = printer.lines[-1]

    answer = _coerce_int(value)
    if answer is None and printer.text:
        # try parse from stdout
        from src.utils import extract_integer_answer

        answer = extract_integer_answer(printer.text, clamp=False)

    return SandboxResult(
        ok=True,
        stdout=printer.text,
        value=value,
        answer=answer,
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if sp is not None and isinstance(value, sp.Integer):
        return int(value)
    if sp is not None and hasattr(value, "is_integer") and value.is_integer:
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            return None
    try:
        s = str(value).strip()
        if re_fullmatch_int(s):
            return int(s)
    except Exception:  # noqa: BLE001
        return None
    return None


def re_fullmatch_int(s: str) -> bool:
    import re

    return bool(re.fullmatch(r"-?\d+", s))


def run_code(
    code: str,
    *,
    timeout_seconds: float = 5.0,
    allowed_modules: list[str] | None = None,
) -> SandboxResult:
    """Execute model-generated code with a wall-clock timeout."""
    import time

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_execute, code, allowed_modules)
        try:
            result = fut.result(timeout=timeout_seconds)
        except FuturesTimeout:
            return SandboxResult(
                ok=False,
                timed_out=True,
                error=f"Timeout after {timeout_seconds}s",
                elapsed_s=time.perf_counter() - t0,
            )
    result.elapsed_s = time.perf_counter() - t0
    return result
