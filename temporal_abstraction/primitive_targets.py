"""Probe-pool construction: blocks of windows with ground-truth latents + probe target definitions.

Pool blocks per seed (C=64, H=16) — see docs/temporal_abstraction/02_protocol_lock.md:

  TREND      500  family=trend,      sign balanced, |slope| 4 levels, noise 3 levels
  PERIODIC   600  family=periodic,   period in 7 levels, noise 3 levels
  APERIODIC  300  family=aperiodic,  trend+AR, no sinusoid (control for periodicity)
  LOCAL      500  family=local,      phi 4 levels, sigma 3 levels
  TP         300  trend+periodic      (SEEN composition)
  TL         300  trend+local         (UNSEEN composition)
  TPL        300  trend+periodic+local(UNSEEN composition)
  REGIME     300  trend -> periodic switch (UNSEEN temporal composition)
  ANOM_CLEAN 320  matched clean controls for the anomaly block
  ANOM       320  anomaly injected in z-space, type x index recorded
  CP_CLEAN   320  matched clean controls for the change-point block
  CP         320  change point at a uniform random context index
"""
from __future__ import annotations

import numpy as np

from .generators import Window, make_window

BLOCKS = ["TREND", "PERIODIC", "APERIODIC", "LOCAL", "TP", "TL", "TPL", "REGIME",
          "ANOM_CLEAN", "ANOM", "CP_CLEAN", "CP"]
NOISE_LEVELS = [0.03, 0.08, 0.15]
SIGMA_LEVELS = [0.05, 0.15, 0.30]
PHI_LEVELS = [0.55, 0.70, 0.85, 0.95]
BASE_FAMILIES = ["trend", "periodic", "local"]


def _block(name: str, n: int, rng: np.random.Generator, **props):
    for _ in range(n):
        if name == "TREND":
            yield make_window("trend", rng, sign=float(rng.choice([-1, 1])), noise=float(rng.choice(NOISE_LEVELS)), **props)
        elif name == "PERIODIC":
            yield make_window("periodic", rng, period=float(rng.choice([8, 10, 12, 16, 20, 24, 32])),
                              noise=float(rng.choice(NOISE_LEVELS)), **props)
        elif name == "APERIODIC":
            yield make_window("aperiodic", rng, noise=float(rng.choice(NOISE_LEVELS)), **props)
        elif name == "LOCAL":
            yield make_window("local", rng, phi=float(rng.choice(PHI_LEVELS)), sigma=float(rng.choice(SIGMA_LEVELS)), **props)
        elif name == "TP":
            yield make_window("mixture", rng, period=float(rng.choice([8, 10, 12, 16, 20, 24, 32])), **props)
        elif name in ("TL", "TPL"):
            # trend + local AR (+ periodic) built directly, with recorded components
            w = _composite(name, rng)
            yield w
        elif name == "REGIME":
            yield make_window("regime", rng, **props)
        elif name in ("ANOM_CLEAN", "CP_CLEAN", "ANOM", "CP"):
            base = str(rng.choice(BASE_FAMILIES))
            noise = float(rng.choice(NOISE_LEVELS))
            common = dict(noise=noise)
            if name == "ANOM_CLEAN":
                yield make_window(base, rng, **common, **props)
            elif name == "ANOM":
                yield make_window(base, rng, anomaly=True, anomaly_type=str(rng.choice(["spike", "dip", "flatline", "level_shift"])),
                                  **common, **props)
            elif name == "CP_CLEAN":
                yield make_window(base, rng, **common, **props)
            else:
                yield make_window(base, rng, changepoint=True, changepoint_type=str(rng.choice(["level", "variance"])),
                                  **common, **props)
        else:
            raise ValueError(name)


def _composite(name: str, rng: np.random.Generator) -> Window:
    """trend + local AR (+ periodic) composition, standardized on the context."""
    from .generators import _ar1
    C = H = None
    w = make_window("trend", rng)           # reuse ranges; we only keep its latents for slope
    C, H = w.latents["C"], w.latents["H"]
    total = C + H
    t = np.arange(total, dtype=float)
    slope = float(rng.uniform(-0.06, 0.06))
    phi = float(rng.choice([0.6, 0.75, 0.9]))
    sigma = float(rng.choice(SIGMA_LEVELS))
    noise = float(rng.choice(NOISE_LEVELS))
    x = slope * t + _ar1(rng.normal(0.0, sigma, total), phi) + rng.normal(0.0, noise, total)
    lat = dict(family="trend+local", slope=slope, phi=phi, sigma=sigma, noise=noise,
               trend_present=1, periodic_present=0, local_present=1)
    if name == "TPL":
        period = float(rng.choice([8, 10, 12, 16, 20, 24, 32]))
        amp = float(rng.uniform(0.4, 1.2))
        x = x + amp * np.sin(2.0 * np.pi * t / period)
        lat.update(family="trend+periodic+local", period=period, amp=amp, periodic_present=1)
    mu, sd = float(x[:C].mean()), float(x[:C].std()) + 1e-6
    z = (x - mu) / sd
    lat.update(mu=mu, sd=sd, C=C, H=H, legacy=False)
    return Window(ctx=z[:C], fut=z[C:], latents=lat)


def build_pool(seed: int, sizes: dict | None = None) -> list[Window]:
    sizes = sizes or {"TREND": 500, "PERIODIC": 600, "APERIODIC": 300, "LOCAL": 500, "TP": 300,
                      "TL": 300, "TPL": 300, "REGIME": 300, "ANOM_CLEAN": 320, "ANOM": 320,
                      "CP_CLEAN": 320, "CP": 320}
    rng = np.random.default_rng(seed)
    pool: list[Window] = []
    for name in BLOCKS:
        for w in _block(name, sizes[name], rng):
            w.latents["block"] = name
            pool.append(w)
    return pool


# ---------------------------------------------------------------- probe targets
def anomaly_index_map(pool):
    """indices of ANOM (1) / ANOM_CLEAN (0) / clean primitives (control) with their labels."""
    idx_anom = [i for i, w in enumerate(pool) if w.latents["block"] == "ANOM"]
    idx_clean = [i for i, w in enumerate(pool) if w.latents["block"] == "ANOM_CLEAN"]
    return idx_anom, idx_clean


def target_labels(pool: list[Window]) -> dict:
    """All probe targets as (label array, valid-index array, task type)."""
    n = len(pool)
    lat = [w.latents for w in pool]
    T = {}
    tr = np.array([i for i in range(n) if lat[i]["block"] == "TREND"])
    pe = np.array([i for i in range(n) if lat[i]["block"] == "PERIODIC"])
    ap = np.array([i for i in range(n) if lat[i]["block"] == "APERIODIC"])
    lo = np.array([i for i in range(n) if lat[i]["block"] == "LOCAL"])
    an = np.array([i for i in range(n) if lat[i]["block"] == "ANOM"])
    anc = np.array([i for i in range(n) if lat[i]["block"] == "ANOM_CLEAN"])
    cp = np.array([i for i in range(n) if lat[i]["block"] == "CP"])
    cpc = np.array([i for i in range(n) if lat[i]["block"] == "CP_CLEAN"])
    # P1 trend direction
    T["P1_trend_direction"] = (np.array([1 if lat[i]["sign"] > 0 else 0 for i in tr]), tr, "binary")
    # P2 trend strength: 4 classes by |slope| quartile (fixed cut points, protocol-locked)
    cuts = [0.06, 0.10, 0.14]
    T["P2_trend_strength"] = (np.digitize([abs(lat[i]["slope"]) for i in tr], cuts), tr, "ordinal4")
    T["P2b_trend_strength_reg"] = (np.array([abs(lat[i]["slope"]) for i in tr]), tr, "regression")
    # P3 periodicity presence: periodic vs aperiodic
    keep = np.concatenate([pe, ap])
    T["P3_periodicity_present"] = (np.array([1] * len(pe) + [0] * len(ap)), keep, "binary")
    # P4 dominant period (7 classes)
    per = sorted({lat[i]["period"] for i in pe})
    T["P4_dominant_period"] = (np.array([per.index(lat[i]["period"]) for i in pe]), pe, "multiclass7")
    T["P4b_dominant_period_reg"] = (np.array([lat[i]["period"] for i in pe]), pe, "regression")
    # P5 local dependence (phi levels)
    phis = sorted({lat[i]["phi"] for i in lo})
    T["P5_local_dependence"] = (np.array([phis.index(lat[i]["phi"]) for i in lo]), lo, "multiclass4")
    T["P5b_local_dependence_reg"] = (np.array([lat[i]["phi"] for i in lo]), lo, "regression")
    # P6 noise level (3 classes) over trend+periodic+local blocks
    nz = np.concatenate([tr, pe, lo])
    T["P6_noise_level"] = (np.array([NOISE_LEVELS.index(lat[i]["noise"]) if lat[i]["noise"] in NOISE_LEVELS
                                     else 0 for i in nz]), nz, "ordinal3")
    # P7 anomaly presence / P8 anomaly location
    keep_a = np.concatenate([an, anc])
    T["P7_anomaly_present"] = (np.array([1] * len(an) + [0] * len(anc)), keep_a, "binary")
    T["P8_anomaly_location"] = (np.array([lat[i]["anomaly_bin"] for i in an]), an, "multiclass8")
    # P9 change-point presence / P10 change-point location
    keep_c = np.concatenate([cp, cpc])
    T["P9_changepoint_present"] = (np.array([1] * len(cp) + [0] * len(cpc)), keep_c, "binary")
    T["P10_changepoint_location"] = (np.array([lat[i]["changepoint_bin"] for i in cp]), cp, "multiclass8")
    return T


COMPOSITION_GROUP = {"TREND": "T", "PERIODIC": "P", "LOCAL": "L", "TP": "TP", "TL": "TL", "TPL": "TPL"}
SEEN_COMPOSITIONS = ["T", "P", "L", "TP"]          # signature probe trains on these
UNSEEN_COMPOSITIONS = ["TL", "TPL"]                 # presence signatures never seen in training
# REGIME shares the (1,1,0) presence signature with TP, so it is NOT usable for the signature probe.
# It is used for the temporal-ARRANGEMENT probe C3 (simultaneous TP vs sequential REGIME), which asks
# whether the representation encodes *how* primitives are arranged, not just which ones are present.
ARRANGEMENT_CLASSES = ["TP", "REGIME"]
PRESENCE_SIGNATURE = {"TREND": (1, 0, 0), "PERIODIC": (0, 1, 0), "LOCAL": (0, 0, 1),
                      "TP": (1, 1, 0), "TL": (1, 0, 1), "TPL": (1, 1, 1), "REGIME": (1, 1, 0)}
