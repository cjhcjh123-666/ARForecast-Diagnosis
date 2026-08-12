"""Hand-crafted temporal statistics behind the cheap (non-LLM) router floor.

The feature router is deliberately kept simple: standardized hand features fed
into a softmax regression.  Its routing accuracy is the baseline that any LLM
router must beat to justify the LLM.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "slope_norm",
    "linear_r2",
    "volatility",
    "acf1",
    "dom_period",
    "period_strength",
    "spectral_entropy",
    "peak_concentration",
    "range_ratio",
]


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    variance = float(x.var()) + 1e-12
    n = len(x)
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        out[lag] = float(np.dot(x[: n - lag], x[lag:])) / float(n - lag) / variance
    return out


def _dominant_period(x: np.ndarray, max_period: int, min_period: int = 3) -> tuple[float, float]:
    """Return (dominant period, strength) using local maxima of the autocorrelation.

    A monotonically decaying autocorrelation (e.g. AR(1)) has no local maximum,
    so it is reported as aperiodic.  A genuinely periodic signal has peaks at
    multiples of its period.
    """

    max_lag = min(max_period, len(x) - 2)
    if max_lag <= min_period:
        return 0.0, 0.0
    acf = _autocorr(x, max_lag)
    best_lag, best_val = 0.0, 0.0
    for lag in range(min_period, max_lag):
        if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1] and acf[lag] > best_val:
            best_val = float(acf[lag])
            best_lag = float(lag)
    if best_val < 0.25:
        return 0.0, 0.0
    return best_lag, best_val


def temporal_features(x: np.ndarray, max_period: int = 32) -> np.ndarray:
    """Fixed feature vector for one univariate context window."""

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    t = np.arange(n, dtype=float)
    std_x = float(x.std()) + 1e-9

    slope = float(np.polyfit(t, x, 1)[0]) if n > 2 else 0.0
    line = slope * t + (x.mean() - slope * t.mean())
    resid = x - line
    ss_tot = float(np.sum((x - x.mean()) ** 2)) + 1e-12
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot
    slope_norm = abs(slope) * n / std_x

    volatility = float(np.std(np.diff(resid))) / (std_x + 1e-9)

    max_lag = min(n // 2, max_period)
    acf = _autocorr(resid, max_lag)
    acf1 = float(acf[1]) if len(acf) > 1 else 0.0

    dom_period, period_strength = _dominant_period(resid, max_period)

    spectrum = np.abs(np.fft.rfft(resid)) ** 2
    spectrum = spectrum[1:]
    ps = spectrum / (spectrum.sum() + 1e-12)
    spectral_entropy = float(
        -np.sum(ps * np.log(ps + 1e-12)) / np.log(len(ps) + 1e-12)
    )
    top3 = np.sort(spectrum)[-3:]
    peak_concentration = float(top3.sum() / (spectrum.sum() + 1e-12))

    range_ratio = float((x.max() - x.min()) / std_x)
    return np.asarray(
        [
            slope_norm,
            r2,
            volatility,
            acf1,
            dom_period,
            period_strength,
            spectral_entropy,
            peak_concentration,
            range_ratio,
        ],
        dtype=np.float64,
    )
