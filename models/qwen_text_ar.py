"""Qwen3-8B text-numeric autoregressive wrapper.

This module imports Transformers lazily so the small-baseline environment does
not require an LLM runtime. It is intended for the second, language-pretraining
factor of the study rather than the controlled first-milestone comparison.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch


class QwenTextARForecaster:
    """Fine-tunable Qwen text forecaster with prompt-only label masking."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen experiments require transformers; use the wavellm environment"
            ) from exc

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device.index or 0} if self.device.type == "cuda" else None,
            attn_implementation="sdpa",
        )
        self._uses_device_map = self.device.type == "cuda"
        if self.device.type != "cuda":
            self.model.to(self.device)

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise RuntimeError("Qwen LoRA experiments require peft") from exc
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
        if not self._uses_device_map:
            self.model.to(self.device)
        self.model.config.use_cache = False

    @staticmethod
    def _format_values(values: torch.Tensor) -> str:
        clipped = values.detach().float().clamp(-9.99, 9.99).cpu().tolist()
        return " ".join(f"{float(value):+.2f}" for value in clipped)

    def _prompt_and_answer(
        self, context: torch.Tensor, future: torch.Tensor
    ) -> tuple[list[str], list[str]]:
        prompts = [
            "Forecast the next values of this normalized time series.\n"
            "History: "
            + self._format_values(row)
            + "\nForecast:"
            for row in context
        ]
        answers = [" " + self._format_values(row) for row in future]
        return prompts, answers

    def encode_batch(self, context: torch.Tensor, future: torch.Tensor) -> dict[str, torch.Tensor]:
        prompts, answers = self._prompt_and_answer(context, future)
        prompt_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answers, add_special_tokens=False)["input_ids"]
        sequences = [prompt + answer for prompt, answer in zip(prompt_ids, answer_ids)]
        padded = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100
        for row_index, prompt in enumerate(prompt_ids):
            labels[row_index, : len(prompt)] = -100
        return {key: value.to(self.device) for key, value in {**padded, "labels": labels}.items()}

    def train_step(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model.train()
        batch = self.encode_batch(context, future)
        output = self.model(**batch)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return float(output.loss.detach().cpu())

    @torch.no_grad()
    def generate_forecast(
        self,
        context: torch.Tensor,
        horizon: int,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[list[list[float]], str]:
        self.model.eval()
        prompt = (
            "Forecast the next values of this normalized time series.\n"
            "History: "
            + self._format_values(context[0])
            + "\nForecast:"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens or max(32, horizon * 8),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = generated[0, encoded["input_ids"].size(1) :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        numbers = [float(match) for match in re.findall(r"[+-]?\d\.\d\d", text)]
        return [numbers[:horizon]], text

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())
