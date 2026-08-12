"""Lightweight statistical forecast experts used by the router experiment."""

from __future__ import annotations

import numpy as np


class TrendExpert:
    """Polynomial extrapolation of the context window."""

    name = "trend"

    def __init__(self, degree: int = 1) -> None:
        self.degree = degree

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        t = np.arange(len(context), dtype=float)
        if len(context) < self.degree + 2:
            return np.full(horizon, context[-1])
        coef = np.polyfit(t, context, self.degree)
        t_future = np.arange(len(context), len(context) + horizon, dtype=float)
        return np.polyval(coef, t_future)


class PeriodicExpert:
    """Seasonal naive with damped residual correction.

    The dominant period is detected from local maxima of the detrended
    autocorrelation; the future repeats the last detected cycle, and a damped
    AR(1) correction is added from the residual at the window end.
    """

    name = "periodic"

    def __init__(self, max_period: int = 32, min_period: int = 3) -> None:
        self.max_period = max_period
        self.min_period = min_period

    def _dominant_period(self, x: np.ndarray) -> float:
        t = np.arange(len(x), dtype=float)
        resid = x - np.polyval(np.polyfit(t, x, 1), t)
        n = len(resid)
        max_lag = min(n // 2, self.max_period)
        if max_lag <= self.min_period:
            return 0.0
        best_lag, best_val = 0.0, 0.0
        for lag in range(self.min_period, max_lag):
            a = resid[: n - lag]
            b = resid[lag:]
            denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) + 1e-12
            value = float(np.dot(a, b)) / denom
            if value > 0.4 and value > best_val:
                best_val, best_lag = value, lag
        return best_lag

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        period = self._dominant_period(context)
        if period <= 0:
            return np.full(horizon, context[-1])
        period = int(round(period))
        cycle = context[-period:]
        reps = int(np.ceil(horizon / period))
        sequence = np.tile(cycle, reps)[:horizon].astype(np.float64)
        t = np.arange(len(context), dtype=float)
        resid = context - np.polyval(np.polyfit(t, context, 1), t)
        if len(resid) > 2:
            phi = float(
                np.dot(resid[:-1], resid[1:]) / (np.dot(resid[:-1], resid[:-1]) + 1e-12)
            )
            phi = min(max(phi, 0.0), 0.95)
            last = resid[-1]
            sequence = sequence + np.asarray(
                [last * (phi**k) for k in range(1, horizon + 1)]
            )
        return sequence


class LocalExpert:
    """Stable AR(k) least-squares fit with recursive multi-step rollout.

    The raw least-squares coefficients are damped toward the origin until the
    characteristic polynomial is stable (all roots strictly inside the unit
    circle), which prevents divergent rollouts on real-world series.
    """

    name = "local"

    def __init__(self, order: int = 5) -> None:
        self.order = order

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        k = min(self.order, len(context) - 2)
        if k < 1:
            return np.full(horizon, context[-1])
        y = context[k:]
        design = np.stack(
            [context[i : len(context) - k + i] for i in range(k)], axis=1
        )
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

        # enforce stability by damping toward zero
        for _ in range(60):
            companion = np.zeros((k, k))
            companion[0, :] = coef
            if k > 1:
                companion[1:, :-1] = np.eye(k - 1)
            roots = np.linalg.eigvals(companion)
            if np.max(np.abs(roots)) < 0.999:
                break
            coef = coef * 0.9

        history = list(context[-k:])
        out = []
        for _ in range(horizon):
            pred = float(np.dot(coef, history[-k:]))
            out.append(pred)
            history.append(pred)
        return np.asarray(out)


class CrossChannelExpert:
    """Ridge regression from all channels' recent values to the target future."""

    name = "cross"

    def __init__(self, ridge: float = 1.0, k: int = 4) -> None:
        self.ridge = ridge
        self.k = k
        self.W: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CrossChannelExpert":
        """X: [N, C, K] last-k values of every channel; Y: [N, H]."""
        num_windows, _, _ = X.shape
        Phi = X.reshape(num_windows, -1)
        Phi = np.concatenate([Phi, np.ones((num_windows, 1))], axis=1)
        eye = np.eye(Phi.shape[1])
        self.W = np.linalg.solve(Phi.T @ Phi + self.ridge * eye, Phi.T @ Y)
        return self

    def predict(self, context_channels: np.ndarray, horizon: int) -> np.ndarray:
        """context_channels: [C, context_len] -> [H]."""
        if self.W is None:
            raise RuntimeError("CrossChannelExpert must be fit before predict")
        k = min(self.k, context_channels.shape[1])
        Phi = np.concatenate([context_channels[:, -k:].reshape(-1), [1.0]])
        return Phi @ self.W
