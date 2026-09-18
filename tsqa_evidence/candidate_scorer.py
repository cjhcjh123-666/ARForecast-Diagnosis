"""Shared question-conditioned candidate scorer with explicit evidence interaction."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn


def _projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)
    return torch.from_numpy(matrix)


class QuestionEvidenceScorer(nn.Module):
    """Score each candidate using a nonlinear function of question and evidence.

    The fixed projections are buffers rather than fitted parameters.  Two P/R branches created
    with the same dimensions and seed therefore receive byte-identical projections, while their
    trainable scorers can be initialized identically and trained independently.
    """

    def __init__(self, temporal_dim: int, question_dim: int, projection_dim: int = 128,
                 hidden_dim: int = 64, seed: int = 7):
        super().__init__()
        self.register_buffer("temporal_projection", _projection(temporal_dim, projection_dim, seed + 101))
        self.register_buffer("question_projection", _projection(question_dim, projection_dim, seed + 211))
        # Parameters are initialized on CPU before ``model.to(device)``.  Keep
        # the seeded fork CPU-only so a small diagnostic head never initializes
        # RNG state on unrelated visible GPUs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.scorer = nn.Sequential(
                nn.Linear(2 * projection_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            )

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, (x.shape[-1],))

    def forward(self, temporal: torch.Tensor, question_options: torch.Tensor) -> torch.Tensor:
        if question_options.ndim != 3 or temporal.ndim != 2:
            raise ValueError("expected temporal[B,Dx], question_options[B,K,Dq]")
        ux = self._normalize(temporal @ self.temporal_projection)
        uq = self._normalize(question_options @ self.question_projection)
        ux = ux[:, None, :].expand(-1, uq.shape[1], -1)
        return self.scorer(torch.cat([ux, uq], dim=-1)).squeeze(-1)

    def projection_hashes(self) -> dict[str, str]:
        def digest(tensor: torch.Tensor) -> str:
            return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return {"temporal_projection": digest(self.temporal_projection),
                "question_projection": digest(self.question_projection)}


def tensor_sha256(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()
