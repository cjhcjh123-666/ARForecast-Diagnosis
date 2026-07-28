"""Qwen3-8B text-numeric autoregressive wrapper.

This module imports Transformers lazily so the small-baseline environment does
not require an LLM runtime. It is intended for the second, language-pretraining
factor of the study rather than the controlled first-milestone comparison.
"""

from __future__ import annotations

import re
from typing import Optional

import torch


class QwenTextARForecaster:
    """Fine-tunable Qwen text forecaster with prompt-only label masking."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        random_init: bool = False,
        adapter_path: Optional[str] = None,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen experiments require transformers; use the wavellm environment"
            ) from exc

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        if random_init and adapter_path is not None:
            raise ValueError("random_init cannot load an adapter without the random base weights")
        if random_init:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_config(config, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": self.device.index or 0} if self.device.type == "cuda" else None,
                **model_kwargs,
            )
        self._uses_device_map = self.device.type == "cuda" and not random_init
        if self.device.type != "cuda":
            self.model.to(self.device)

        if use_lora:
            try:
                from peft import LoraConfig, PeftModel, get_peft_model
            except ImportError as exc:
                raise RuntimeError("Qwen LoRA experiments require peft") from exc
            if adapter_path is not None:
                self.model = PeftModel.from_pretrained(
                    self.model, adapter_path, is_trainable=False
                )
            else:
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
    def generate_forecast_batch(
        self,
        context: torch.Tensor,
        horizon: int,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[list[list[float]], list[str]]:
        self.model.eval()
        prompts = [
            "Forecast the next values of this normalized time series.\n"
            "History: "
            + self._format_values(row)
            + "\nForecast:"
            for row in context
        ]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens or max(32, horizon * 8),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        input_length = encoded["input_ids"].size(1)
        texts = []
        forecasts = []
        for row_index in range(generated.size(0)):
            new_tokens = generated[row_index, input_length:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            numbers = [float(match) for match in re.findall(r"[+-]?\d\.\d\d", text)]
            forecasts.append(numbers[:horizon])
            texts.append(text)
        return forecasts, texts

    @torch.no_grad()
    def generate_forecast(
        self,
        context: torch.Tensor,
        horizon: int,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[list[float], str]:
        forecasts, texts = self.generate_forecast_batch(context[:1], horizon, max_new_tokens)
        return forecasts[0], texts[0]

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())
