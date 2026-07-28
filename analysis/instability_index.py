"""Local rollout sensitivity diagnostics."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def rollout_instability_index(model, context: torch.Tensor, epsilon: float = 1e-3) -> np.ndarray:
    """Measure forecast divergence after a small perturbation to the last value."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    model.eval()
    baseline = model.predict(context)
    perturbed_context = context.clone()
    perturbed_context[:, -1] += epsilon
    perturbed = model.predict(perturbed_context)
    divergence = torch.sqrt(torch.mean((perturbed - baseline) ** 2, dim=1)) / epsilon
    return divergence.detach().cpu().numpy()


def instability_error_correlation(instability: np.ndarray, error: np.ndarray) -> float:
    instability = np.asarray(instability, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    if len(instability) != len(error):
        raise ValueError("instability and error must have the same number of samples")
    if len(instability) < 2 or np.std(instability) < 1e-12 or np.std(error) < 1e-12:
        return 0.0
    return float(np.corrcoef(instability, error)[0, 1])
