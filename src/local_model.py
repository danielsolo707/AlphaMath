"""Offline open-weight math LLM via Hugging Face Transformers.

Designed for AIMO / Kaggle constraints:
  - no external APIs
  - no internet at inference time (local_files_only=True)
  - single GPU (T4 / P100) with optional 4-bit quantization

Published Kaggle kernel (danielsolo1770/alpha-math) loads:
  Qwen2.5-Math-7B from a Kaggle Model mirror (float16, device_map=auto).

Also works with DeepSeek-Math and other chat-tuned math 7B checkpoints when
tokenizer.apply_chat_template is available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.llm import BaseLLM, LLMResponse

log = logging.getLogger("alphamath.local_model")

# Primary open-source math model used on the published Kaggle run
DEFAULT_MATH_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
# Alternate Hub ids accepted via config
ALTERNATE_MATH_MODELS = (
    "Qwen/Qwen2.5-Math-7B",
    "deepseek-ai/deepseek-math-7b-instruct",
)


def resolve_model_path(model: str, model_path: str | None = None) -> str:
    """Prefer an explicit local directory (Kaggle input) over a Hub id."""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return str(p)
        raise FileNotFoundError(
            f"llm.model_path does not exist: {model_path}\n"
            "On Kaggle, attach the model as a Dataset/Model and point model_path "
            "at the folder containing config.json (e.g. "
            "/kaggle/input/models/urvishp80/qwen-2.5-math-7b/transformers/default/1)."
        )
    return model


def _pick_dtype(name: str | None):
    import torch

    name = (name or "auto").lower()
    if name in {"auto", "none", ""}:
        if torch.cuda.is_available():
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
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "float16",
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
                "  pip install -r requirements/gpu.txt"
            ) from e

        self.model_id = resolve_model_path(model, model_path)
        self.display_name = model_path or model
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self._torch = torch
        self.generation_count = 0

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

        dtype = _pick_dtype(torch_dtype)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "device_map": device_map,
            "revision": revision,
        }

        # bitsandbytes 4/8-bit needs modern GPU arches (typically sm_70+).
        # Kaggle sometimes assigns Tesla P100 (sm_60): current PyTorch + bnb can
        # hard-crash (ops.cu "named symbol not found" → segfault). Fall back to fp16.
        use_4bit = bool(load_in_4bit and torch.cuda.is_available())
        use_8bit = bool(load_in_8bit and torch.cuda.is_available() and not use_4bit)
        if torch.cuda.is_available() and (use_4bit or use_8bit):
            major, minor = torch.cuda.get_device_capability(0)
            gpu_name = torch.cuda.get_device_name(0)
            if major < 7:
                log.warning(
                    "GPU %s (sm_%d%d) is too old for bitsandbytes on this PyTorch build; "
                    "disabling 4/8-bit and loading float16 instead.",
                    gpu_name,
                    major,
                    minor,
                )
                print(
                    f"[WARN] {gpu_name} sm_{major}{minor}: bitsandbytes disabled → float16 load",
                    flush=True,
                )
                use_4bit = False
                use_8bit = False

        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif use_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = dtype
            if not torch.cuda.is_available():
                model_kwargs["device_map"] = None
            elif torch.cuda.is_available() and device_map == "auto":
                # Leave headroom on 16GB cards (P100/T4) when not quantizing.
                try:
                    free_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if free_gb <= 17:
                        model_kwargs["max_memory"] = {0: f"{max(int(free_gb - 1.5), 10)}GiB"}
                except Exception:  # pragma: no cover
                    pass

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

        if not torch.cuda.is_available() and device_map is None:
            self.model.to("cpu")

        self.device = next(self.model.parameters()).device
        # Fail fast if accelerate parked weights on meta/disk because VRAM/arch is broken.
        try:
            devices = {str(p.device) for p in self.model.parameters()}
        except Exception:  # pragma: no cover
            devices = {str(self.device)}
        if any(d.startswith("meta") for d in devices) or any(
            "disk" in d for d in devices
        ):
            raise RuntimeError(
                f"Model parameters on unusable devices {sorted(devices)}. "
                "This usually means GPU offload to disk on an unsupported GPU "
                "(e.g. P100 with modern PyTorch). Re-run on Tesla T4 (sm_75)."
            )
        log.info("Math model ready on %s (param devices=%s)", self.device, sorted(devices))
        print(f"Math model ready on {self.device} (param devices={sorted(devices)})", flush=True)

        # One tiny generate to prove CUDA kernels work before a 90-problem loop.
        if torch.cuda.is_available():
            try:
                probe = self.tokenizer("1+1=", return_tensors="pt")
                probe = {k: v.to(self.device) for k, v in probe.items()}
                with torch.inference_mode():
                    _ = self.model.generate(**probe, max_new_tokens=4, do_sample=False)
                print("CUDA generate probe: OK", flush=True)
            except Exception as exc:
                raise RuntimeError(
                    "CUDA generate probe failed after load. This GPU/PyTorch pair cannot "
                    f"run inference ({type(exc).__name__}: {exc}). "
                    "Cancel and re-queue until Kaggle assigns Tesla T4 (sm_70+)."
                ) from exc

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Build a single prompt string; prefer tokenizer chat template (Qwen etc.)."""
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("apply_chat_template failed (%s); using role fallback", e)

        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_parts = [m["content"] for m in messages if m["role"] == "user"]
        assistant_parts = [m["content"] for m in messages if m["role"] == "assistant"]
        user = "\n\n".join(user_parts)
        if system:
            user = f"{system.strip()}\n\n{user}"
        if assistant_parts:
            hist = f"System: {system}\n\n" if system else ""
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
        temperature = float(kwargs.get("temperature", 0.7))
        max_new = int(kwargs.get("max_tokens", self.max_new_tokens))
        top_p = float(kwargs.get("top_p", 0.9))
        seed = int(kwargs.get("seed", 2026))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
                top_p=top_p,
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        self.generation_count += 1

        new_tokens = out[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return LLMResponse(
            text=text,
            model=str(self.display_name),
            backend="transformers",
            raw={
                "prompt_chars": len(prompt),
                "new_tokens": int(new_tokens.shape[0]),
                "seed": seed,
                "generation_index": self.generation_count,
                "device": str(self.device),
            },
        )


def build_transformers_llm(llm_cfg: dict[str, Any]) -> TransformersMathLLM:
    return TransformersMathLLM(
        model=llm_cfg.get("model") or DEFAULT_MATH_MODEL,
        model_path=llm_cfg.get("model_path"),
        local_files_only=bool(llm_cfg.get("local_files_only", False)),
        load_in_4bit=bool(llm_cfg.get("load_in_4bit", False)),
        load_in_8bit=bool(llm_cfg.get("load_in_8bit", False)),
        torch_dtype=str(llm_cfg.get("torch_dtype") or "float16"),
        device_map=llm_cfg.get("device_map", "auto"),
        max_new_tokens=int(llm_cfg.get("max_new_tokens") or llm_cfg.get("max_tokens") or 1024),
        trust_remote_code=bool(llm_cfg.get("trust_remote_code", False)),
        revision=llm_cfg.get("revision"),
    )
