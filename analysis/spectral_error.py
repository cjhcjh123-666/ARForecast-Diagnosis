"""Simple frequency-domain diagnostics for forecast windows."""

from __future__ import annotations

import numpy as np


def spectral_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred_amplitude = np.abs(np.fft.rfft(prediction, axis=1))
    target_amplitude = np.abs(np.fft.rfft(target, axis=1))
    amplitude_mae = float(np.mean(np.abs(pred_amplitude - target_amplitude)))
    if prediction.shape[1] > 1:
        pred_dominant = np.argmax(pred_amplitude[:, 1:], axis=1) + 1
        target_dominant = np.argmax(target_amplitude[:, 1:], axis=1) + 1
    else:
        pred_dominant = np.zeros(len(prediction))
        target_dominant = np.zeros(len(target))
    return {
        "amplitude_mae": amplitude_mae,
        "dominant_frequency_match": float(np.mean(pred_dominant == target_dominant)),
        "dominant_frequency_abs_error": float(np.mean(np.abs(pred_dominant - target_dominant))),
    }
