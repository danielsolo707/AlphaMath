"""Restricted, process-isolated Python execution for math-tool verification.

The AST gate reduces accidental access to dangerous Python primitives. A fresh
worker process supplies the hard wall-clock timeout: timed-out code is killed,
not merely abandoned in a live thread. This is still defense-in-depth for
trusted model-generated math code, not a multi-tenant security boundary.
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any


FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "exit", "quit", "os", "sys", "subprocess", "socket", "pathlib", "shutil",
    "pickle", "requests", "httpx", "ctypes", "multiprocessing", "threading",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "memoryview", "help", "license", "credits",
}
OPTIONAL_ALIASES = {
    "sympy": {"sympy", "sp"},
    "numpy": {"numpy", "np"},
}


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    value: Any = None
    answer: int | None = None
    error: str | None = None
    error_type: str | None = None
    timed_out: bool = False
    output_truncated: bool = False
    elapsed_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class SecurityError(Exception):
    pass


class DependencyError(Exception):
    pass


class OutputLimitError(Exception):
    pass


class _SafePrinter:
    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.lines: list[str] = []
        self.char_count = 0
        self.truncated = False

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        sep = str(kwargs.get("sep", " "))
        line = sep.join(str(a) for a in args)
        remaining = self.max_chars - self.char_count
        if remaining <= 0:
            self.truncated = True
            raise OutputLimitError(f"stdout exceeded {self.max_chars} characters")
        if len(line) + 1 > remaining:
            self.lines.append(line[: max(0, remaining)])
            self.char_count = self.max_chars
            self.truncated = True
            raise OutputLimitError(f"stdout exceeded {self.max_chars} characters")
        self.lines.append(line)
        self.char_count += len(line) + 1

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _validate_ast(tree: ast.AST) -> set[str]:
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise SecurityError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            referenced.add(node.id)
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise SecurityError(f"Disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SecurityError(f"Disallowed dunder attribute: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                raise SecurityError(f"Disallowed call: {node.func.id}")
    return referenced


def _build_globals(
    allowed_modules: list[str] | None,
    referenced_names: set[str],
) -> tuple[dict[str, Any], list[str]]:
    allowed = set(
        allowed_modules
        or ["math", "sympy", "itertools", "functools", "collections", "fractions", "decimal"]
    )
    g: dict[str, Any] = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "float": float, "int": int, "len": len,
            "list": list, "max": max, "min": min, "pow": pow, "print": None,
            "range": range, "round": round, "set": set, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "zip": zip,
            "True": True, "False": False, "None": None,
        }
    }
    loaded: list[str] = []
    for module_name in sorted(allowed):
        aliases = OPTIONAL_ALIASES.get(module_name, {module_name})
        is_optional = module_name in OPTIONAL_ALIASES
        if is_optional and not aliases.intersection(referenced_names):
            continue
        try:
            module = math if module_name == "math" else importlib.import_module(module_name)
        except ImportError as exc:
            if aliases.intersection(referenced_names):
                raise DependencyError(
                    f"Generated code requires '{module_name}', but it is not installed. "
                    "Install the project requirements before evaluation."
                ) from exc
            continue
        g[module_name] = module
        if module_name == "sympy":
            g["sp"] = module
        elif module_name == "numpy":
            g["np"] = module
        loaded.append(module_name)
    return g, loaded


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if hasattr(value, "is_integer") and value.is_integer:
        try:
            return int(value)
        except Exception:
            return None
    try:
        text = str(value).strip()
        return int(text) if re.fullmatch(r"-?\d+", text) else None
    except Exception:
        return None


def sanitize_model_code(code: str) -> str:
    """Rewrite common allow-listed import lines; block all other imports."""
    out_lines: list[str] = []
    allowed = {name: name for name in (
        "math", "sympy", "numpy", "itertools", "functools", "collections", "fractions", "decimal"
    )}
    for line in code.splitlines():
        raw, stripped = line, line.strip()
        match = re.match(r"^import\s+([a-zA-Z0-9_]+)(?:\s+as\s+([a-zA-Z0-9_]+))?\s*$", stripped)
        if match:
            module_name, alias = match.group(1), match.group(2)
            if module_name in allowed:
                name = alias or module_name
                if name not in {module_name, "sp", "np"}:
                    out_lines.append(f"{name} = {module_name}")
                elif alias == "sp":
                    out_lines.append("sp = sympy")
                elif alias == "np":
                    out_lines.append("np = numpy")
                continue
            out_lines.append(f"# blocked import: {stripped}")
            continue
        match = re.match(r"^from\s+([a-zA-Z0-9_]+)\s+import\s+(.+)$", stripped)
        if match:
            module_name, names = match.group(1), match.group(2)
            if module_name in allowed:
                for part in names.split(","):
                    part = part.strip()
                    if not part or part == "*":
                        continue
                    if " as " in part:
                        source, destination = [p.strip() for p in part.split(" as ", 1)]
                        out_lines.append(f"{destination} = {module_name}.{source}")
                    else:
                        out_lines.append(f"{part} = {module_name}.{part}")
                continue
            out_lines.append(f"# blocked import: {stripped}")
            continue
        out_lines.append(raw)
    return "\n".join(out_lines)


def _execute(
    code: str,
    allowed_modules: list[str] | None,
    max_output_chars: int,
) -> SandboxResult:
    printer = _SafePrinter(max_output_chars)
    try:
        tree = ast.parse(code, mode="exec")
        referenced = _validate_ast(tree)
        globals_dict, loaded_modules = _build_globals(allowed_modules, referenced)
    except (SecurityError, DependencyError, SyntaxError) as exc:
        return SandboxResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )

    globals_dict["__builtins__"]["print"] = printer
    local: dict[str, Any] = {}
    try:
        compiled = compile(tree, filename="<sandbox>", mode="exec")
        exec(compiled, globals_dict, local)  # noqa: S102 - AST-gated worker process
    except Exception as exc:
        return SandboxResult(
            ok=False,
            stdout=printer.text,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
            error_type=type(exc).__name__,
            output_truncated=printer.truncated,
            meta={"loaded_modules": loaded_modules},
        )

    value = None
    for key in ("ANSWER", "answer", "result", "RESULT", "final", "FINAL"):
        if key in local:
            value = local[key]
            break
    if value is None and printer.lines:
        value = printer.lines[-1]
    answer = _coerce_int(value)
    if answer is None and printer.text:
        from src.utils import extract_integer_answer

        answer = extract_integer_answer(printer.text, clamp=False)
    safe_value = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
    return SandboxResult(
        ok=True,
        stdout=printer.text,
        value=safe_value,
        answer=answer,
        output_truncated=printer.truncated,
        meta={"loaded_modules": loaded_modules},
    )


def _apply_memory_limit(memory_limit_mb: int | None) -> bool:
    if not memory_limit_mb or os.name == "nt":
        return False
    try:
        import resource

        limit = int(memory_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return True
    except Exception:
        return False


def _worker_main() -> int:
    request = json.load(sys.stdin)
    memory_applied = _apply_memory_limit(request.get("memory_limit_mb"))
    result = _execute(
        request["code"],
        request.get("allowed_modules"),
        int(request.get("max_output_chars", 8000)),
    )
    result.meta["memory_limit_applied"] = memory_applied
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


def run_code(
    code: str,
    *,
    timeout_seconds: float = 5.0,
    allowed_modules: list[str] | None = None,
    max_output_chars: int = 8000,
    max_source_chars: int = 50_000,
    memory_limit_mb: int | None = 1536,
) -> SandboxResult:
    """Execute model-generated code in a killable worker process."""
    started = time.perf_counter()
    if len(code) > max_source_chars:
        return SandboxResult(
            ok=False,
            error=f"SourceLimitError: code exceeded {max_source_chars} characters",
            error_type="SourceLimitError",
        )
    cleaned = sanitize_model_code(code)
    request = {
        "code": cleaned,
        "allowed_modules": allowed_modules,
        "max_output_chars": max_output_chars,
        "memory_limit_mb": memory_limit_mb,
    }
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [sys.executable, "-m", "src.sandbox", "--worker"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )
    try:
        stdout, stderr = process.communicate(
            json.dumps(request, ensure_ascii=False),
            timeout=max(0.001, float(timeout_seconds)),
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return SandboxResult(
            ok=False,
            timed_out=True,
            error=f"TimeoutError: worker killed after {timeout_seconds}s",
            error_type="TimeoutError",
            elapsed_s=time.perf_counter() - started,
            meta={"worker_killed": True, "sanitized": cleaned != code},
        )

    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        return SandboxResult(
            ok=False,
            error=f"WorkerError: {stderr.strip() or 'sandbox worker exited unexpectedly'}",
            error_type="WorkerError",
            elapsed_s=elapsed,
            meta={"returncode": process.returncode, "sanitized": cleaned != code},
        )
    try:
        payload = json.loads(stdout)
        result = SandboxResult(**payload)
    except Exception as exc:
        return SandboxResult(
            ok=False,
            error=f"ProtocolError: {exc}; worker_stdout={stdout[:500]!r}",
            error_type="ProtocolError",
            elapsed_s=elapsed,
        )
    result.elapsed_s = elapsed
    result.meta["sanitized"] = cleaned != code
    result.meta["worker_pid"] = process.pid
    return result


if __name__ == "__main__":
    if "--worker" not in sys.argv:
        raise SystemExit("src.sandbox is an internal worker module")
    raise SystemExit(_worker_main())
