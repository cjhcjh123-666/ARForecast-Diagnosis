"""Three forecasting mechanisms with a common causal Transformer family."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from models.backbone import CausalTransformerBackbone
from tokenizers.numeric import FixedWidthTextNumberTokenizer


def _backbone(
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    max_seq_len: int,
) -> CausalTransformerBackbone:
    return CausalTransformerBackbone(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )


class DirectForecastModel(nn.Module):
    """Predict all future values from one encoded context."""

    def __init__(
        self,
        context_len: int,
        horizon: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.value_embedding = nn.Linear(1, d_model)
        self.backbone = _backbone(d_model, n_heads, n_layers, dropout, context_len)
        self.forecast_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(self.value_embedding(context.unsqueeze(-1)))
        return self.forecast_head(hidden[:, -1])

    def training_loss(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(context), future)

    @torch.no_grad()
    def predict(self, context: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        prediction = self(context)
        requested = self.horizon if horizon is None else horizon
        return prediction[:, :requested]


class ContinuousARModel(nn.Module):
    """Teacher-forced continuous scalar autoregression with free rollout."""

    def __init__(
        self,
        context_len: int,
        horizon: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.horizon = horizon
        self.value_embedding = nn.Linear(1, d_model)
        self.backbone = _backbone(d_model, n_heads, n_layers, dropout, context_len + horizon - 1)
        self.next_value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def _teacher_forced_prediction(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([context, future[:, :-1]], dim=1).unsqueeze(-1)
        hidden = self.backbone(self.value_embedding(inputs))
        start = self.context_len - 1
        selected = hidden[:, start : start + self.horizon]
        return self.next_value_head(selected).squeeze(-1)

    def training_loss(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self._teacher_forced_prediction(context, future), future)

    def forward(self, context: torch.Tensor, future: Optional[torch.Tensor] = None) -> torch.Tensor:
        if future is not None:
            return self._teacher_forced_prediction(context, future)
        return self.predict(context)

    @torch.no_grad()
    def predict(self, context: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        requested = self.horizon if horizon is None else horizon
        history = context
        predictions = []
        for _ in range(requested):
            hidden = self.backbone(self.value_embedding(history.unsqueeze(-1)))
            next_value = self.next_value_head(hidden[:, -1]).squeeze(-1)
            predictions.append(next_value)
            history = torch.cat([history, next_value.unsqueeze(-1)], dim=1)
        return torch.stack(predictions, dim=1)


class TextARModel(nn.Module):
    """Generate fixed-width textual numeric tokens one token at a time."""

    def __init__(
        self,
        context_len: int,
        horizon: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_abs: float = 9.99,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.horizon = horizon
        self.tokenizer = FixedWidthTextNumberTokenizer(max_abs=max_abs)
        token_length = self.tokenizer.value_width * (context_len + horizon)
        self.token_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        self.backbone = _backbone(d_model, n_heads, n_layers, dropout, token_length)
        self.lm_head = nn.Linear(d_model, self.tokenizer.vocab_size)

    def _teacher_forced_logits(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        values = torch.cat([context, future], dim=1)
        tokens = self.tokenizer.encode(values)
        hidden = self.backbone(self.token_embedding(tokens))
        first_future_position = self.context_len * self.tokenizer.value_width - 1
        token_count = self.horizon * self.tokenizer.value_width
        return self.lm_head(hidden[:, first_future_position : first_future_position + token_count])

    def training_loss(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        logits = self._teacher_forced_logits(context, future)
        targets = self.tokenizer.encode(future)
        return F.cross_entropy(logits.reshape(-1, self.tokenizer.vocab_size), targets.reshape(-1))

    def forward(self, context: torch.Tensor, future: Optional[torch.Tensor] = None) -> torch.Tensor:
        if future is not None:
            return self._teacher_forced_logits(context, future)
        return self.predict(context)

    @torch.no_grad()
    def predict(self, context: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        requested = self.horizon if horizon is None else horizon
        generated = self.tokenizer.encode(context)
        for _ in range(requested * self.tokenizer.value_width):
            hidden = self.backbone(self.token_embedding(generated))
            next_token = self.lm_head(hidden[:, -1]).argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)
        generated_future = generated[:, -requested * self.tokenizer.value_width :]
        return self.tokenizer.decode(generated_future)
