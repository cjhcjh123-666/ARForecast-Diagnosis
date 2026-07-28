"""A deterministic fixed-width character-like tokenizer for normalized scalars."""

from __future__ import annotations

import torch


class FixedWidthTextNumberTokenizer:
    """Encode values as five tokens representing ``+d.dd`` or ``-d.dd``.

    Values are clipped before formatting. The compact fixed width keeps the
    token-level objective aligned with each scalar while retaining a textual
    number representation and a real generation loop.
    """

    value_width = 5
    vocab_size = 13
    plus_id = 0
    minus_id = 1
    digit_offset = 2
    dot_id = 12

    def __init__(self, max_abs: float = 9.99, decimals: int = 2) -> None:
        if decimals != 2:
            raise ValueError("the first experiment uses exactly two decimal places")
        self.max_abs = float(max_abs)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 0:
            values = values.reshape(1)
        clipped = values.detach().clamp(-self.max_abs, self.max_abs)
        scaled = torch.round(clipped.abs() * 100.0).to(torch.long)
        integer = (scaled // 100).clamp(0, 9)
        fractional = scaled % 100
        tens = fractional // 10
        ones = fractional % 10
        sign = torch.where(clipped < 0, self.minus_id, self.plus_id).to(torch.long)
        dot = torch.full_like(sign, self.dot_id)
        encoded = torch.stack(
            [
                sign,
                integer + self.digit_offset,
                dot,
                tens + self.digit_offset,
                ones + self.digit_offset,
            ],
            dim=-1,
        )
        return encoded.reshape(*values.shape[:-1], values.shape[-1] * self.value_width)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(-1) % self.value_width != 0:
            raise ValueError("token sequence length must be divisible by value_width")
        groups = tokens.reshape(*tokens.shape[:-1], -1, self.value_width)
        sign = torch.where(groups[..., 0] == self.minus_id, -1.0, 1.0)
        integer = (groups[..., 1] - self.digit_offset).clamp(0, 9).float()
        tens = (groups[..., 3] - self.digit_offset).clamp(0, 9).float()
        ones = (groups[..., 4] - self.digit_offset).clamp(0, 9).float()
        return sign * (integer + 0.1 * tens + 0.01 * ones)
