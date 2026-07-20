"""Pluggable LLM backends for code generation.

Backends:
  - transformers / deepseek_math / local  → open-weight math model (Kaggle / offline GPU)
  - mock                                  → deterministic templates for CPU pipeline tests
  - openai / openai_compatible            → optional cloud or local OpenAI-API servers
  - anthropic                             → optional Claude API
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    raw: Any = None


class BaseLLM(ABC):
    @abstractmethod
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Offline solver templates keyed by problem fingerprint.

    CPU-only pipeline smoke tests. Not used for Kaggle competition inference.
    """

    def __init__(self) -> None:
        # NOTE: sandbox forbids import statements — use preloaded math / sp / sympy.
        self._templates: list[tuple[list[str], str]] = [
            (
                ["2^10", "divided by 7", "remainder"],
                "ANSWER = pow(2, 10, 7)\nprint(ANSWER)",
            ),
            (
                ["sum of the first 50"],
                "n = 50\nANSWER = n * (n + 1) // 2\nprint(ANSWER)",
            ),
            (
                ["sum of the first 40"],
                "n = 40\nANSWER = n * (n + 1) // 2\nprint(ANSWER)",
            ),
            (
                ["positive divisors", "360"],
                "def num_divisors(n):\n"
                "    c = 0\n    for i in range(1, int(n**0.5) + 1):\n"
                "        if n % i == 0:\n            c += 1 if i * i == n else 2\n"
                "    return c\nANSWER = num_divisors(360)\nprint(ANSWER)",
            ),
            (
                ["last three digits", "7^5"],
                "ANSWER = pow(7, 5, 1000)\nprint(ANSWER)",
            ),
            (
                ["5 red", "3 blue", "3 marbles"],
                "ANSWER = math.comb(8, 3)\nprint(ANSWER)",
            ),
            (
                ["n^2 + n - 56", "positive integer"],
                "n = sp.symbols('n', integer=True, positive=True)\n"
                "sols = sp.solve(n**2 + n - 56, n)\nANSWER = int(sols[0])\nprint(ANSWER)",
            ),
            (
                ["gcd(252", "105"],
                "ANSWER = math.gcd(252, 105)\nprint(ANSWER)",
            ),
            (
                ["coprime to 100", "phi(100)", "euler"],
                "ANSWER = int(sp.totient(100))\nprint(ANSWER)",
            ),
            (
                ["sum_{k=1}^{10}", "k^2"],
                "ANSWER = sum(k*k for k in range(1, 11))\nprint(ANSWER)",
            ),
            (
                ["trailing zeros", "100!"],
                "def trailing_zeros(n):\n    z = 0\n    while n:\n        n //= 5\n        z += n\n"
                "    return z\nANSWER = trailing_zeros(100)\nprint(ANSWER)",
            ),
        ]

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        problem = user.lower()
        for keys, code in self._templates:
            if all(k.lower() in problem for k in keys):
                text = (
                    "REASONING:\nMatched offline demo template for portfolio evaluation.\n\n"
                    f"CODE:\n```python\n{code}\n```"
                )
                return LLMResponse(text=text, model="mock-templates", backend="mock")

        text = (
            "REASONING:\nNo offline template matched; emitting a failing stub so the "
            "verifier can request a retry (use the DeepSeek-Math backend for open problems).\n\n"
            "CODE:\n```python\nANSWER = None\nprint('NO_TEMPLATE')\n```"
        )
        return LLMResponse(text=text, model="mock-templates", backend="mock")


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai package required for this backend. pip install openai"
            ) from e

        key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not key and not base_url:
            raise RuntimeError(
                "OPENAI_API_KEY (or LLM_API_KEY) not set. "
                "For local servers set llm.base_url and a dummy key."
            )
        self.client = OpenAI(api_key=key or "not-needed", base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        temp = kwargs.get("temperature", 0.2)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, model=self.model, backend="openai", raw=resp)


class AnthropicLLM(BaseLLM):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("anthropic package required. pip install anthropic") from e
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", 0.2),
            system=system,
            messages=user_msgs,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(text=text, model=self.model, backend="anthropic", raw=resp)


def build_llm(cfg: dict[str, Any]) -> BaseLLM:
    llm_cfg = dict(cfg.get("llm", {}) or {})
    backend = (llm_cfg.get("backend") or "transformers").lower()
    model = llm_cfg.get("model") or "deepseek-ai/deepseek-math-7b-instruct"
    max_tokens = int(llm_cfg.get("max_tokens") or llm_cfg.get("max_new_tokens") or 1024)
    base_url = llm_cfg.get("base_url")
    key_env = llm_cfg.get("api_key_env") or "OPENAI_API_KEY"
    api_key = os.getenv(key_env) if key_env else None
    allow_mock_fallback = bool(llm_cfg.get("allow_mock_fallback", True))

    if backend in {"mock", "demo"}:
        return MockLLM()
    if backend in {
        "transformers",
        "huggingface",
        "hf",
        "local",
        "deepseek",
        "deepseek_math",
        "deepseek-math",
    }:
        try:
            from src.local_model import build_transformers_llm

            return build_transformers_llm(llm_cfg)
        except Exception as e:
            # Kaggle / competition configs must fail hard (no silent mock).
            if llm_cfg.get("local_files_only") or not allow_mock_fallback:
                raise
            import warnings

            warnings.warn(
                f"DeepSeek-Math backend unavailable ({type(e).__name__}: {e}). "
                "Falling back to mock templates for CPU smoke tests. "
                "Download weights (scripts/download_deepseek_math.py) or set "
                "--model-path for real math-model inference.",
                RuntimeWarning,
                stacklevel=2,
            )
            return MockLLM()
    if backend in {"openai", "openai_compatible", "ollama", "vllm"}:
        return OpenAICompatibleLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )
    if backend == "anthropic":
        return AnthropicLLM(
            model=model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown llm.backend: {backend}")
