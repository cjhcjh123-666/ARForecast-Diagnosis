"""Forecasting metrics with no third-party scientific dependencies."""

from __future__ import annotations

import numpy as np


def _as_numpy(values: np.ndarray) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def forecast_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    prediction = _as_numpy(prediction)
    target = _as_numpy(target)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} != target shape {target.shape}")
    error = prediction - target
    per_horizon_rmse = np.sqrt(np.mean(error**2, axis=0))
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "per_horizon_rmse": per_horizon_rmse.tolist(),
    }
