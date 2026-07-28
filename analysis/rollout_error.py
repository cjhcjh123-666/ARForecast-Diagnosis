"""Error propagation summaries over the forecast horizon."""

from __future__ import annotations

import numpy as np


def rollout_error_by_horizon(prediction: np.ndarray, target: np.ndarray) -> dict[str, list[float]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    return {
        "step": list(range(1, error.shape[1] + 1)),
        "mae": np.mean(np.abs(error), axis=0).tolist(),
        "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
    }
