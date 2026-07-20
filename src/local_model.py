"""Offline open-weight math LLM via Hugging Face Transformers.

Designed for AIMO / Kaggle constraints:
  - no external APIs
  - no internet at inference time (local_files_only=True)
  - single GPU (T4 / P100) with optional 4-bit quantization

Primary model: deepseek-ai/deepseek-math-7b-instruct
Also works with other chat-tuned math models (Qwen2.5-Math, etc.) when
tokenizer.apply_chat_template is available.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.llm import BaseLLM, LLMResponse

log = logging.getLogger("alphamath.local_model")

# Default open-source math specialist used by ALPHA-MATH
DEFAULT_MATH_MODEL = "deepseek-ai/deepseek-math-7b-instruct"


def resolve_model_path(model: str, model_path: str | None = None) -> str:
    """Prefer an explicit local directory (Kaggle input) over a Hub id."""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return str(p)
        raise FileNotFoundError(
            f"llm.model_path does not exist: {model_path}\n"
            "On Kaggle, attach the model as a Dataset/Model and point model_path "
            "at /kaggle/input/<name>/..."
        )
    return model


def _pick_dtype(name: str | None):
    import torch

    name = (name or "auto").lower()
    if name in {"auto", "none", ""}:
        if torch.cuda.is_available():
            # bf16 on Ampere+ if available, else fp16
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unknown torch_dtype: {name}")
    return mapping[name]


class TransformersMathLLM(BaseLLM):
    """Causal LM chat backend for math-specialized open weights."""

    def __init__(
        self,
        model: str = DEFAULT_MATH_MODEL,
        *,
        model_path: str | None = None,
        local_files_only: bool = False,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        torch_dtype: str = "auto",
        device_map: str | dict | None = "auto",
        max_new_tokens: int = 1024,
        trust_remote_code: bool = True,
        revision: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers (+ torch) required for the local math model backend.\n"
                "  pip install -r requirements-gpu.txt"
            ) from e

        self.model_id = resolve_model_path(model, model_path)
        self.display_name = model_path or model
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self._torch = torch

        log.info(
            "Loading math model %s (4bit=%s, local_only=%s, cuda=%s)",
            self.model_id,
            load_in_4bit,
            local_files_only,
            torch.cuda.is_available(),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            revision=revision,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        dtype = _pick_dtype(torch_dtype)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "device_map": device_map,
            "revision": revision,
        }

        if load_in_4bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = dtype
            if not torch.cuda.is_available():
                model_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

        if not torch.cuda.is_available() and device_map is None:
            self.model.to("cpu")

        self.device = next(self.model.parameters()).device
        log.info("Math model ready on %s", self.device)

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Build a single prompt string; prefer tokenizer chat template."""
        tok = self.tokenizer
        # HF chat templates (Qwen, Llama-3 style, many DeepSeek variants)
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("apply_chat_template failed (%s); using DeepSeek-style fallback", e)

        # DeepSeek-Math instruct style fallback
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_parts = [m["content"] for m in messages if m["role"] == "user"]
        assistant_parts = [m["content"] for m in messages if m["role"] == "assistant"]
        user = "\n\n".join(user_parts)
        if system:
            user = f"{system.strip()}\n\n{user}"
        # Multi-turn (retries): append previous assistant if present
        if assistant_parts:
            # Unusual for our agent (we resend full user with feedback), but keep safe
            hist = ""
            for m in messages:
                if m["role"] == "user":
                    hist += f"User: {m['content']}\n\n"
                elif m["role"] == "assistant":
                    hist += f"Assistant: {m['content']}\n\n"
            return hist + "Assistant:"
        return f"User: {user}\n\nAssistant:"

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        torch = self._torch
        prompt = self._format_prompt(messages)
        temperature = float(kwargs.get("temperature", 0.2))
        max_new = int(kwargs.get("max_tokens", self.max_new_tokens))

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=float(kwargs.get("top_p", 0.95)),
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return LLMResponse(
            text=text,
            model=str(self.display_name),
            backend="transformers",
            raw={"prompt_chars": len(prompt), "new_tokens": int(new_tokens.shape[0])},
        )


def build_transformers_llm(llm_cfg: dict[str, Any]) -> TransformersMathLLM:
    return TransformersMathLLM(
        model=llm_cfg.get("model") or DEFAULT_MATH_MODEL,
        model_path=llm_cfg.get("model_path"),
        local_files_only=bool(llm_cfg.get("local_files_only", False)),
        load_in_4bit=bool(llm_cfg.get("load_in_4bit", True)),
        load_in_8bit=bool(llm_cfg.get("load_in_8bit", False)),
        torch_dtype=str(llm_cfg.get("torch_dtype") or "auto"),
        device_map=llm_cfg.get("device_map", "auto"),
        max_new_tokens=int(llm_cfg.get("max_new_tokens") or llm_cfg.get("max_tokens") or 1024),
        trust_remote_code=bool(llm_cfg.get("trust_remote_code", True)),
        revision=llm_cfg.get("revision"),
    )
