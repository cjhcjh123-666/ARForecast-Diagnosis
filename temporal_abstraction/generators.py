"""Ground-truth temporal generators for the abstraction probes.

Design rules
------------
* `data/dynamics.py` is FROZEN and is not modified. `render_legacy()` re-implements its five
  families with the *identical* parameter ranges and draw order, so "legacy"-compatible windows can
  be produced (and verified against `build_labeled_windows`) inside the new study.
* Every window records the latent parameters that generated it (`latents` dict) — this is what makes
  the probe labels ground truth instead of model-derived.
* Extra knobs that the legacy generator does not have: signed slope (trend direction), per-window
  noise level, wider period set, anomaly injection (spike / dip / flatline / level shift with a
  recorded index+width), and change points at a uniform random index (level or variance switch).

All windows are standardized with the CONTEXT mean/std only (identical to the legacy protocol).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

C_DEFAULT = 64
H_DEFAULT = 16
PERIODS = [8, 10, 12, 16, 20, 24, 32]
NOISE_LEVELS_R = [0.03, 0.08, 0.15]
ANOMALY_TYPES = ["spike", "dip", "flatline", "level_shift"]
CHANGEPOINT_TYPES = ["level", "variance"]


@dataclass
class Window:
    ctx: np.ndarray
    fut: np.ndarray
    latents: dict = field(default_factory=dict)


def _ar1(e: np.ndarray, phi: float) -> np.ndarray:
    x = np.empty(len(e))
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    return x


def render_legacy(family: str, C: int, H: int, rng: np.random.Generator,
                  overrides: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Re-implementation of data/dynamics.generate_window with the same ranges + recorded latents."""
    p = dict(overrides or {})
    total = C + H
    t = np.arange(total, dtype=float)
    L: dict = {"family": family}
    if family == "trend":
        slope = p.get("slope", rng.uniform(0.03, 0.18))
        noise = p.get("noise", 0.05)
        e = rng.normal(0.0, noise, total)
        x = slope * t + e
        L.update(slope=slope, noise=noise)
    elif family == "periodic":
        period = p.get("period", float(rng.choice([12, 24])))
        amp = p.get("amp", rng.uniform(0.5, 1.5))
        noise = p.get("noise", 0.05)
        phase = p.get("phase", rng.uniform(0.0, 2.0 * np.pi))
        e = rng.normal(0.0, noise, total)
        x = amp * np.sin(2.0 * np.pi * t / period + phase) + e
        L.update(period=period, amp=amp, noise=noise, phase=phase)
    elif family == "local":
        phi = p.get("phi", rng.uniform(0.7, 0.95))
        sigma = p.get("sigma", 0.15)
        e = rng.normal(0.0, sigma, total)
        x = _ar1(e, phi)
        L.update(phi=phi, sigma=sigma)
    elif family == "mixture":
        slope = p.get("slope", rng.uniform(0.01, 0.06))
        period = p.get("period", float(rng.choice([12, 24])))
        amp = p.get("amp", rng.uniform(0.4, 1.2))
        e = rng.normal(0.0, 0.05, total)
        x = slope * t + amp * np.sin(2.0 * np.pi * t / period) + e
        L.update(slope=slope, period=period, amp=amp, noise=0.05)
    elif family == "regime":
        switch = int(C * 0.6)
        e = np.concatenate([rng.normal(0.0, 0.05, switch), rng.normal(0.0, 0.05, total - switch)])
        x = np.empty(total)
        x[:switch] = 0.08 * t[:switch] + e[:switch]
        x[switch:] = np.sin(2.0 * np.pi * t[switch:] / 12.0) + e[switch:]
        L.update(switch=switch, slope_pre=0.08, period_post=12.0, noise=0.05)
    else:
        raise ValueError(f"unknown legacy family {family!r}")
    return x, L


def render_primitive(family: str, C: int, H: int, rng: np.random.Generator,
                     overrides: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Extended primitive families with the extra knobs required by the probe taxonomy."""
    p = dict(overrides or {})
    total = C + H
    t = np.arange(total, dtype=float)
    L: dict = {"family": family}
    if family == "trend":
        mag = p.get("slope_mag", rng.uniform(0.03, 0.18))
        sign = p.get("sign", float(rng.choice([-1.0, 1.0])))
        slope = sign * mag
        noise = p.get("noise", float(rng.choice([0.03, 0.08, 0.15])))
        x = slope * t + rng.normal(0.0, noise, total)
        L.update(slope=slope, slope_mag=mag, sign=sign, noise=noise)
    elif family == "periodic":
        period = float(p.get("period", rng.choice(PERIODS)))
        amp = p.get("amp", rng.uniform(0.3, 1.8))
        noise = p.get("noise", float(rng.choice([0.03, 0.08, 0.15])))
        phase = p.get("phase", rng.uniform(0.0, 2.0 * np.pi))
        x = amp * np.sin(2.0 * np.pi * t / period + phase) + rng.normal(0.0, noise, total)
        L.update(period=period, amp=amp, noise=noise, phase=phase, periodicity_present=1)
    elif family == "aperiodic":
        # trend + AR noise, no periodic component (control for periodicity presence)
        slope = p.get("slope", rng.uniform(-0.08, 0.08))
        phi = p.get("phi", 0.6)
        noise = p.get("noise", float(rng.choice([0.03, 0.08, 0.15])))
        x = slope * t + _ar1(rng.normal(0.0, noise, total), phi)
        L.update(slope=slope, phi=phi, noise=noise, periodicity_present=0)
    elif family == "local":
        phi = p.get("phi", rng.uniform(0.5, 0.97))
        sigma = p.get("sigma", float(rng.choice([0.05, 0.15, 0.30])))
        x = _ar1(rng.normal(0.0, sigma, total), phi)
        L.update(phi=phi, sigma=sigma, noise=sigma)
    elif family == "regime":
        # trend -> periodic switch at a UNIFORM random index inside the context (legacy uses 0.6*C)
        switch = int(p.get("switch", rng.integers(int(0.3 * C), int(0.7 * C))))
        slope_pre = p.get("slope_pre", float(rng.choice([-1.0, 1.0])) * rng.uniform(0.04, 0.16))
        period_post = float(p.get("period_post", rng.choice(PERIODS)))
        amp_post = p.get("amp_post", rng.uniform(0.4, 1.4))
        noise = p.get("noise", float(rng.choice(NOISE_LEVELS_R)))
        e = rng.normal(0.0, noise, total)
        x = np.empty(total)
        x[:switch] = slope_pre * t[:switch] + e[:switch]
        x[switch:] = amp_post * np.sin(2.0 * np.pi * t[switch:] / period_post) + e[switch:]
        L.update(switch=switch, slope_pre=slope_pre, period_post=period_post, amp_post=amp_post,
                 noise=noise, changepoint_present=1, changepoint_index=switch,
                 changepoint_type="regime", changepoint_joint_index=switch / max(1, C - 1))
    elif family == "mixture":
        slope = p.get("slope", rng.uniform(-0.06, 0.06))
        period = float(p.get("period", rng.choice(PERIODS)))
        amp = p.get("amp", rng.uniform(0.4, 1.2))
        noise = p.get("noise", 0.05)
        x = slope * t + amp * np.sin(2.0 * np.pi * t / period) + rng.normal(0.0, noise, total)
        L.update(slope=slope, period=period, amp=amp, noise=noise, periodicity_present=1)
    else:
        raise ValueError(family)
    # optional anomaly / change-point injection (recorded)
    x, extra = inject(x, C, H, rng, p)
    L.update(extra)
    return x, L


def inject(z: np.ndarray, C: int, H: int, rng: np.random.Generator, p: dict) -> tuple[np.ndarray, dict]:
    """Inject an anomaly and/or change point **in standardized (z) space**.

    Probes P7-P10 read the CONTEXT representation only, so any injected event must live inside the
    context; indices are recorded both raw and normalized. Magnitudes are in z-units, which makes
    the detection difficulty independent of the family's raw scale.
    """
    out = z.astype(float).copy()
    extra: dict = {"anomaly_present": 0, "changepoint_present": 0}
    if p.get("anomaly"):
        typ = p.get("anomaly_type") or str(rng.choice(ANOMALY_TYPES))
        width = int(p.get("anomaly_width", rng.integers(1, 4)))
        first = int(p.get("anomaly_min_index", 4))
        last = int(p.get("anomaly_max_index", max(first + 1, C - width - 2)))
        idx = int(p.get("anomaly_index", rng.integers(first, last)))
        mag = float(p.get("anomaly_mag", rng.uniform(2.5, 5.0)))
        if typ == "spike":
            out[idx:idx + width] += mag
        elif typ == "dip":
            out[idx:idx + width] -= mag
        elif typ == "flatline":
            out[idx:idx + width] = out[idx]
        elif typ == "level_shift":
            out[idx:] += mag
        else:
            raise ValueError(typ)
        extra.update(anomaly_present=1, anomaly_type=typ, anomaly_index=idx, anomaly_width=width,
                     anomaly_mag=mag, anomaly_joint_index=idx / max(1, C - 1),
                     anomaly_bin=int(np.clip(int(8 * idx / max(1, C)), 0, 7)))
    if p.get("changepoint"):
        typ = p.get("changepoint_type") or str(rng.choice(CHANGEPOINT_TYPES))
        lo = int(p.get("changepoint_min_index", int(0.2 * C)))
        hi = int(p.get("changepoint_max_index", int(0.8 * C)))
        idx = int(p.get("changepoint_index", rng.integers(lo, hi)))
        if typ == "level":
            delta = float(p.get("changepoint_delta", rng.uniform(1.5, 3.5) * rng.choice([-1.0, 1.0])))
            out[idx:] += delta
            extra.update(changepoint_delta=delta)
        elif typ == "variance":
            scale = float(p.get("changepoint_scale", rng.uniform(2.0, 3.5)))
            seg = out[idx:]
            out[idx:] = seg * scale
            extra.update(changepoint_scale=scale)
        else:
            raise ValueError(typ)
        extra.update(changepoint_present=1, changepoint_type=typ, changepoint_index=idx,
                     changepoint_joint_index=idx / max(1, C - 1),
                     changepoint_bin=int(np.clip(int(8 * idx / max(1, C)), 0, 7)))
    return out, extra


def make_window(family: str, rng: np.random.Generator, C: int = C_DEFAULT, H: int = H_DEFAULT,
                legacy: bool = False, **props) -> Window:
    """Generate one window; returns standardized context/future plus the full latent record."""
    render = render_legacy if legacy else render_primitive
    x, L = render(family, C, H, rng, {k: v for k, v in props.items()
                                      if k not in ("anomaly", "changepoint")})
    mu = float(x[:C].mean())
    sd = float(x[:C].std()) + 1e-6
    z = (x - mu) / sd
    z, extra = inject(z, C, H, rng, props)          # z-space, context-only events
    L.update(extra)
    L.update(mu=mu, sd=sd, C=C, H=H, legacy=bool(legacy))
    return Window(ctx=z[:C].astype(np.float64), fut=z[C:].astype(np.float64), latents=L)


def make_legacy_compatible(family: str, rng: np.random.Generator, C: int = C_DEFAULT,
                           H: int = H_DEFAULT) -> Window:
    return make_window(family, rng, C, H, legacy=True)
