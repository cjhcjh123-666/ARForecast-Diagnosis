"""Random-initialized text AR model using the Qwen BPE tokenizer only."""

from __future__ import annotations

import re
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from models.backbone import CausalTransformerBackbone


class RandomBPETextARModel(nn.Module):
    """Small random Transformer that matches Qwen's text tokenization protocol."""

    def __init__(
        self,
        tokenizer_path: str,
        device: str = "cuda:0",
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The tokenizer-matched baseline requires transformers; use the wavellm environment"
            ) from exc
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.backbone = CausalTransformerBackbone(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=0.1,
            max_seq_len=max_seq_len,
        )
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.to(self.device)

    @staticmethod
    def _format_values(values: torch.Tensor) -> str:
        clipped = values.detach().float().clamp(-9.99, 9.99).cpu().tolist()
        return " ".join(f"{float(value):+.2f}" for value in clipped)

    def _prompts_answers(
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

    def _encode_teacher_forcing(
        self, context: torch.Tensor, future: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        prompts, answers = self._prompts_answers(context, future)
        prompt_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answers, add_special_tokens=False)["input_ids"]
        sequences = [prompt + answer for prompt, answer in zip(prompt_ids, answer_ids)]
        padded = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100
        for row_index, prompt in enumerate(prompt_ids):
            labels[row_index, : len(prompt)] = -100
        return {key: value.to(self.device) for key, value in {**padded, "labels": labels}.items()}

    def training_loss(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        batch = self._encode_teacher_forcing(context, future)
        hidden = self.backbone(self.token_embedding(batch["input_ids"]))
        logits = self.lm_head(hidden)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            batch["labels"].reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate_forecast_batch(
        self,
        context: torch.Tensor,
        horizon: int,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[list[list[float]], list[str]]:
        self.eval()
        prompts, _ = self._prompts_answers(context, torch.zeros(context.size(0), horizon))
        forecasts, texts = [], []
        for prompt in prompts:
            generated = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            for _ in range(max_new_tokens or max(32, horizon * 8)):
                hidden = self.backbone(self.token_embedding(generated))
                next_token = self.lm_head(hidden[:, -1]).argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            text = self.tokenizer.decode(generated[0, -max(32, horizon * 8) :], skip_special_tokens=True)
            numbers = [float(match) for match in re.findall(r"(?<![0-9])[+-]?\d+\.\d{2}", text)]
            forecasts.append(numbers[:horizon])
            texts.append(text)
        return forecasts, texts

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

