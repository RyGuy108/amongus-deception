"""Lazy, process-local Hugging Face backend for reproducible open-weight runs."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCAL_PREFIX = "local:"


def is_local_model(model: str) -> bool:
    return str(model).startswith(LOCAL_PREFIX)


def _model_path(model: str) -> str:
    value = str(model)[len(LOCAL_PREFIX) :].strip()
    if not value:
        raise ValueError("local model identifiers must be formatted as local:/path/to/model")
    return value


class LocalTransformersRuntime:
    """One lazily loaded causal LM per model path, serialized for safe MPS use."""

    _instances: dict[str, "LocalTransformersRuntime"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, model: str):
        path = _model_path(model)
        key = str(Path(path).expanduser().resolve()) if Path(path).exists() else path
        with cls._instances_lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = super().__new__(cls)
                instance._runtime_key = key
                instance._initialized = False
                cls._instances[key] = instance
            return instance

    def __init__(self, model: str):
        if self._initialized:
            return
        self.model_path = self._runtime_key
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"
        self._generation_lock = threading.Lock()
        self._initialized = True

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.backends.mps.is_available():
            self._device = "mps"
            dtype = torch.float16
        else:
            self._device = "cpu"
            dtype = torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self._device)
        self._model.eval()

    def _generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        import torch

        with self._generation_lock, torch.inference_mode():
            self._ensure_loaded()
            prompt = self._tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
            encoded = self._tokenizer(prompt, return_tensors="pt")
            encoded = {name: value.to(self._device) for name, value in encoded.items()}
            do_sample = temperature > 0
            generation_args = {
                **encoded,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": do_sample,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if do_sample:
                generation_args.update(
                    {"temperature": max(float(temperature), 1e-5), "top_p": 0.95}
                )
            output = self._model.generate(**generation_args)
            generated = output[0, encoded["input_ids"].shape[1] :]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_new_tokens: int | None = None,
    ) -> str:
        token_limit = max_new_tokens or int(
            os.getenv("AMONGUS_LOCAL_MAX_NEW_TOKENS", "192")
        )
        return await asyncio.to_thread(
            self._generate,
            messages,
            temperature=float(temperature),
            max_new_tokens=token_limit,
        )

    def _generate_batch(
        self,
        message_batches: Sequence[Sequence[Mapping[str, str]]],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        import torch

        if not message_batches:
            return []
        with self._generation_lock, torch.inference_mode():
            self._ensure_loaded()
            prompts = [
                self._tokenizer.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True
                )
                for messages in message_batches
            ]
            original_padding_side = self._tokenizer.padding_side
            self._tokenizer.padding_side = "left"
            try:
                encoded = self._tokenizer(prompts, return_tensors="pt", padding=True)
            finally:
                self._tokenizer.padding_side = original_padding_side
            encoded = {name: value.to(self._device) for name, value in encoded.items()}
            do_sample = temperature > 0
            generation_args = {
                **encoded,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": do_sample,
                "pad_token_id": (
                    self._tokenizer.pad_token_id
                    if self._tokenizer.pad_token_id is not None
                    else self._tokenizer.eos_token_id
                ),
            }
            if do_sample:
                generation_args.update(
                    {"temperature": max(float(temperature), 1e-5), "top_p": 0.95}
                )
            output = self._model.generate(**generation_args)
            generated = output[:, encoded["input_ids"].shape[1] :]
            return [
                text.strip()
                for text in self._tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
            ]

    async def generate_batch(
        self,
        message_batches: Sequence[Sequence[Mapping[str, str]]],
        *,
        temperature: float,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate equal-setting requests together without changing their prompts."""
        token_limit = max_new_tokens or int(
            os.getenv("AMONGUS_LOCAL_MAX_NEW_TOKENS", "192")
        )
        return await asyncio.to_thread(
            self._generate_batch,
            message_batches,
            temperature=float(temperature),
            max_new_tokens=token_limit,
        )

    def _count_chat_tokens(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool
    ) -> int:
        with self._generation_lock:
            self._ensure_loaded()
            prompt = self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            return int(self._tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])

    async def count_chat_tokens(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        """Count local tokenizer input tokens for monitor cost accounting."""
        return await asyncio.to_thread(
            self._count_chat_tokens,
            messages,
            add_generation_prompt=add_generation_prompt,
        )

    def _count_text_tokens(self, text: str) -> int:
        with self._generation_lock:
            self._ensure_loaded()
            return int(self._tokenizer(str(text), return_tensors="pt")["input_ids"].shape[1])

    async def count_text_tokens(self, text: str) -> int:
        """Count local tokenizer tokens in generated monitor output."""
        return await asyncio.to_thread(self._count_text_tokens, text)

    def _pooled_activations(
        self,
        messages: Sequence[Mapping[str, str]],
        continuation: str,
    ) -> dict[str, Any]:
        """Return prompt-final and response-mean vectors for every hidden layer."""
        import numpy as np
        import torch

        with self._generation_lock, torch.inference_mode():
            self._ensure_loaded()
            prompt = self._tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"]
            full_ids = self._tokenizer(
                prompt + continuation, return_tensors="pt"
            )["input_ids"]
            inputs = full_ids.to(self._device)
            output = self._model(input_ids=inputs, output_hidden_states=True)
            prompt_index = min(prompt_ids.shape[1] - 1, full_ids.shape[1] - 1)
            response_start = min(prompt_ids.shape[1], full_ids.shape[1] - 1)
            prompt_final = []
            response_mean = []
            for hidden in output.hidden_states:
                prompt_final.append(hidden[0, prompt_index].float().cpu().numpy())
                response_slice = hidden[0, response_start:]
                response_mean.append(response_slice.mean(dim=0).float().cpu().numpy())
            return {
                "prompt_final": np.stack(prompt_final),
                "response_mean": np.stack(response_mean),
                "prompt_tokens": int(prompt_ids.shape[1]),
                "response_tokens": int(full_ids.shape[1] - prompt_ids.shape[1]),
                "device": self._device,
            }

    async def pooled_activations(
        self,
        messages: Sequence[Mapping[str, str]],
        continuation: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._pooled_activations, messages, continuation
        )
