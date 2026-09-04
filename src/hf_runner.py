from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Generation:
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass
class CandidateScores:
    candidates: list[str]
    log_scores: list[float]
    probabilities: list[float]
    latency_ms: float

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.candidates, self.probabilities))


class HFLocalModel:
    """Small Hugging Face causal-LM runner designed for Colab/Kaggle GPUs.

    It supports deterministic generation and candidate continuation likelihoods.
    The latter is preferred for classification because it avoids asking the model
    to fabricate confidence values.
    """

    def __init__(
        self,
        model_id: str,
        *,
        short_name: str | None = None,
        revision: str | None = None,
        load_in_4bit: bool = True,
        max_context_tokens: int = 4096,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.short_name = short_name or model_id.split("/")[-1]
        self.requested_revision = revision
        self.max_context_tokens = max_context_tokens
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer_kwargs = {"trust_remote_code": False}
        if revision:
            tokenizer_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"

        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        if revision:
            model_kwargs["revision"] = revision

        self.quantized_4bit = bool(load_in_4bit and torch.cuda.is_available())
        if self.quantized_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    def metadata(self) -> dict:
        import torch

        return {
            "model_id": self.model_id,
            "short_name": self.short_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "quantized_4bit": self.quantized_4bit,
            "device_type": self.device_type,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "max_context_tokens": self.max_context_tokens,
        }

    def _prompt_text(self, system: str, prompt: str) -> str:
        messages = []
        if system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Some chat templates do not define a separate system role.
                combined = f"INSTRUCTIONS:\n{system.strip()}\n\nTASK:\n{prompt.strip()}" if system.strip() else prompt.strip()
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": combined}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        # Generic fallback for older/simpler causal LMs.
        if system.strip():
            return f"System: {system.strip()}\nUser: {prompt.strip()}\nAssistant:"
        return f"User: {prompt.strip()}\nAssistant:"

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool | None = None,
        seed: int = 42,
    ) -> Generation:
        import torch

        if do_sample is None:
            do_sample = temperature > 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rendered = self._prompt_text(system, prompt)
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update({"temperature": max(float(temperature), 1e-5), "top_p": float(top_p)})

        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = output[0, prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return Generation(
            text=text,
            latency_ms=latency_ms,
            input_tokens=prompt_len,
            output_tokens=int(generated_ids.numel()),
        )

    def score_candidates(
        self,
        *,
        system: str,
        prompt: str,
        candidates: Iterable[str],
        length_normalize: bool = True,
    ) -> CandidateScores:
        import torch
        import torch.nn.functional as F

        candidates = list(candidates)
        if not candidates:
            raise ValueError("At least one candidate is required")

        prefix_text = self._prompt_text(system, prompt)
        prefix_ids = self.tokenizer(
            prefix_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_context_tokens - 16,
        )["input_ids"]
        if not prefix_ids:
            raise ValueError("Prompt tokenized to an empty sequence")

        full_sequences: list[list[int]] = []
        candidate_token_ids: list[list[int]] = []
        for candidate in candidates:
            ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"Candidate tokenized to empty sequence: {candidate!r}")
            candidate_token_ids.append(ids)
            full_sequences.append(prefix_ids + ids)

        max_len = max(len(x) for x in full_sequences)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(full_sequences), max_len), pad_id, dtype=torch.long)
        attention = torch.zeros((len(full_sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(full_sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention[i, : len(seq)] = 1

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention = attention.to(device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits
            log_probs = F.log_softmax(logits.float(), dim=-1)
        latency_ms = (time.perf_counter() - t0) * 1000

        scores: list[float] = []
        prefix_len = len(prefix_ids)
        for i, cand_ids in enumerate(candidate_token_ids):
            token_scores = []
            for j, token_id in enumerate(cand_ids):
                position = prefix_len + j
                # token at `position` is predicted by logits at `position - 1`
                token_scores.append(float(log_probs[i, position - 1, token_id].item()))
            score = float(np.mean(token_scores) if length_normalize else np.sum(token_scores))
            scores.append(score)

        arr = np.asarray(scores, dtype=np.float64)
        arr = arr - np.max(arr)
        probs = np.exp(arr)
        probs /= probs.sum()
        return CandidateScores(candidates=candidates, log_scores=scores, probabilities=probs.tolist(), latency_ms=latency_ms)

    def unload(self) -> None:
        try:
            del self.model
            del self.tokenizer
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
