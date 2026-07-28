"""Shared causal Transformer backbone."""

from __future__ import annotations

import torch
from torch import nn


class CausalTransformerBackbone(nn.Module):
    """A compact decoder-only backbone implemented with causal encoder layers."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(inputs.shape)}")
        sequence_length = inputs.size(1)
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_seq_len={self.max_seq_len}"
            )
        positions = self.position_embedding[:, :sequence_length].to(inputs.dtype)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.norm(self.encoder(inputs + positions, mask=causal_mask))
