"""Controlled, labeled temporal-dynamics windows for the router experiment.

Each window belongs to one of a small set of synthetic dynamics kinds.  The
*intended* expert label is the kind itself; the *oracle* label used for router
supervision is computed by actually running every expert and taking the lowest
forecast error, so mixtures and regime switches are labeled by outcome rather
than by construction.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

INTENDED_KINDS: list[str] = ["trend", "periodic", "local"]


def _ar1(n: int, rng: np.random.Generator, phi: float, sigma: float) -> np.ndarray:
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _standardize(x: np.ndarray, context_len: int) -> np.ndarray:
    c = x[:context_len]
    mean = float(c.mean())
    std = float(c.std()) + 1e-6
    return (x - mean) / std


def generate_window(
    kind: str,
    context_len: int,
    horizon: int,
    rng: np.random.Generator,
    params: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized (context, future) for one window of a dynamics kind."""

    total = context_len + horizon
    t = np.arange(total, dtype=float)
    p = params or {}
    if kind == "trend":
        slope = p.get("slope", rng.uniform(0.03, 0.18))
        noise = p.get("noise", 0.05)
        x = slope * t + rng.normal(0.0, noise, total)
    elif kind == "periodic":
        period = p.get("period", float(rng.choice([12, 24])))
        amp = p.get("amp", rng.uniform(0.5, 1.5))
        noise = p.get("noise", 0.05)
        phase = p.get("phase", rng.uniform(0.0, 2.0 * np.pi))
        x = amp * np.sin(2.0 * np.pi * t / period + phase) + rng.normal(0.0, noise, total)
    elif kind == "local":
        phi = p.get("phi", rng.uniform(0.7, 0.95))
        sigma = p.get("sigma", 0.15)
        x = _ar1(total, rng, phi=phi, sigma=sigma)
    elif kind == "mixture":
        slope = p.get("slope", rng.uniform(0.01, 0.06))
        period = p.get("period", float(rng.choice([12, 24])))
        amp = p.get("amp", rng.uniform(0.4, 1.2))
        x = slope * t + amp * np.sin(2.0 * np.pi * t / period) + rng.normal(0.0, 0.05, total)
    elif kind == "regime":
        switch = int(context_len * 0.6)
        x = np.empty(total)
        x[:switch] = 0.08 * t[:switch] + rng.normal(0.0, 0.05, switch)
        x[switch:] = (
            np.sin(2.0 * np.pi * t[switch:] / 12.0) + rng.normal(0.0, 0.05, total - switch)
        )
    else:
        raise ValueError(f"unknown kind {kind!r}")
    x = _standardize(x, context_len)
    return x[:context_len], x[context_len:]


def build_labeled_windows(
    kinds: Sequence[str],
    context_len: int,
    horizon: int,
    n_per_kind: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacked (contexts, futures, kind_ids).  kind_id indexes into `kinds`."""

    rng = np.random.default_rng(seed)
    contexts, futures, kind_ids = [], [], []
    for kind_id, kind in enumerate(kinds):
        for _ in range(n_per_kind):
            c, f = generate_window(kind, context_len, horizon, rng)
            contexts.append(c)
            futures.append(f)
            kind_ids.append(kind_id)
    return (
        np.stack(contexts),
        np.stack(futures),
        np.asarray(kind_ids, dtype=np.int64),
    )
