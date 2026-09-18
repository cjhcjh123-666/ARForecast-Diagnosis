"""Deterministic evidence-family construction from observable time-series facts.

The generator deliberately computes every gold label from the final, rounded values that a
model sees.  Latent generation parameters are retained for auditing but never rendered into a
model input.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import DATASET_VERSION, PROTOCOL_VERSION

N_STEPS = 48
CHANNELS = ("sensor_A", "sensor_B", "sensor_C", "sensor_D")
TASK_IDS = (
    "A1_interval_trend", "A2_dominant_interval",
    "B1_interval_mean", "B2_interval_mean_difference",
    "C1_local_deviation_interval", "C2_change_point_interval",
    "D1_repeat_interval_comparison", "D2_channel_lead_lag",
    "E1_repeat_and_event", "E2_before_after_structure",
    "F1_conditional_extremum", "F2_event_order",
)
LABELS = ("A", "B", "C", "D")
CONTROL_TASK = {task: "D2_channel_lead_lag" for task in TASK_IDS}
CONTROL_TASK["D2_channel_lead_lag"] = "A2_dominant_interval"
CONTROL_TASK["F2_event_order"] = "A2_dominant_interval"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    category: str
    answer_type: str
    question: str
    target_program: str
    evidence_channels: tuple[str, ...]
    metric: str = "accuracy"


TASK_SPECS = {
    "A1_interval_trend": TaskSpec(
        "A1_interval_trend", "A_global_structure", "class",
        "From t=0 through t=23 inclusive, what is the trend direction of sensor_A? "
        "Use the least-squares slope: above +0.025 units/step is rising, below -0.025 is falling, "
        "and otherwise it is stable.",
        "ols_slope(sensor_A[0:24]); thresholds=(-0.025,+0.025)", ("sensor_A",)),
    "A2_dominant_interval": TaskSpec(
        "A2_dominant_interval", "A_global_structure", "integer",
        "After removing a linear trend from all 48 values of sensor_A, which candidate lag has "
        "the largest Pearson autocorrelation?",
        "argmax autocorr(detrend(sensor_A), lag in {6,8,12})", ("sensor_A",)),
    "B1_interval_mean": TaskSpec(
        "B1_interval_mean", "B_numeric_information", "scalar",
        "What is the mean of sensor_A from t=0 through t=23 inclusive, rounded to the nearest "
        "0.5 unit?",
        "round_half(mean(sensor_A[0:24]))", ("sensor_A",), "mae"),
    "B2_interval_mean_difference": TaskSpec(
        "B2_interval_mean_difference", "B_numeric_information", "scalar",
        "What is mean(sensor_A, t=0..23) minus mean(sensor_A, t=24..47), rounded to the nearest "
        "0.5 unit?",
        "round_half(mean(sensor_A[0:24])-mean(sensor_A[24:48]))", ("sensor_A",), "mae"),
    "C1_local_deviation_interval": TaskSpec(
        "C1_local_deviation_interval", "C_time_localization", "interval",
        "Split sensor_A into four 12-step intervals. In which interval is the single value with "
        "the largest absolute deviation from that interval's median?",
        "argmax_quarter max(abs(sensor_A-median(quarter)))", ("sensor_A",)),
    "C2_change_point_interval": TaskSpec(
        "C2_change_point_interval", "C_time_localization", "interval",
        "At which candidate boundary (t=12, 24, or 36) is the absolute difference between the "
        "four values immediately before and after the boundary largest?",
        "argmax boundary abs(mean(pre4)-mean(post4))", ("sensor_A",)),
    "D1_repeat_interval_comparison": TaskSpec(
        "D1_repeat_interval_comparison", "D_temporal_relation", "class",
        "Using the detrended autocorrelation rule over lags 6, 8, and 12, compare the dominant "
        "repeat intervals of sensor_A and sensor_B.",
        "compare(period(sensor_A),period(sensor_B))", ("sensor_A", "sensor_B")),
    "D2_channel_lead_lag": TaskSpec(
        "D2_channel_lead_lag", "D_temporal_relation", "lag",
        "After centering and clipping each channel to its 10th--90th percentile range, which lag "
        "from -4 through +4 maximizes correlation between sensor_C and sensor_D? Positive k means "
        "sensor_D lags sensor_C by k steps; negative k means it leads.",
        "argmax crosscorr(robust(sensor_C),robust(sensor_D),lag=-4..4)",
        ("sensor_C", "sensor_D")),
    "E1_repeat_and_event": TaskSpec(
        "E1_repeat_and_event", "E_multi_information", "combination",
        "Which option jointly gives sensor_A's dominant detrended repeat lag (6, 8, or 12) and "
        "the 12-step interval containing its strongest local deviation?",
        "(period(sensor_A),event_quarter(sensor_A))", ("sensor_A",)),
    "E2_before_after_structure": TaskSpec(
        "E2_before_after_structure", "E_multi_information", "class",
        "Estimate sensor_A's dominant lag separately on t=0..23 and t=24..47 using candidate "
        "lags 6, 8, and 12. Is the repeat interval shorter after t=24, longer after, or the same?",
        "compare(period(sensor_A[24:48]),period(sensor_A[0:24]))", ("sensor_A",)),
    "F1_conditional_extremum": TaskSpec(
        "F1_conditional_extremum", "F_conditional_reasoning", "interval",
        "First select the half (t=0..23 or t=24..47) with the higher mean of sensor_B. Within "
        "that selected half, which 12-step interval contains the maximum value of sensor_A?",
        "select_half(argmax mean(sensor_B)); locate argmax(sensor_A)",
        ("sensor_A", "sensor_B")),
    "F2_event_order": TaskSpec(
        "F2_event_order", "F_conditional_reasoning", "class",
        "For sensor_C and sensor_D, find each channel's 12-step interval with the strongest "
        "absolute deviation from its own interval median. Which event interval occurs first?",
        "compare(event_quarter(sensor_C),event_quarter(sensor_D))",
        ("sensor_C", "sensor_D")),
}


def stable_seed(*parts: object) -> int:
    raw = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _round_half(value: float) -> float:
    rounded = float(np.round(value * 2.0) / 2.0)
    return 0.0 if rounded == 0.0 else rounded


def _detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    return x - np.polyval(np.polyfit(t, x, 1), t)


def dominant_period(x: np.ndarray) -> tuple[int, float]:
    z = _detrend(np.asarray(x, dtype=float))
    scores = []
    for lag in (6, 8, 12):
        left, right = z[:-lag], z[lag:]
        denom = float(left.std() * right.std())
        score = -1.0 if denom < 1e-12 else float(np.corrcoef(left, right)[0, 1])
        scores.append(score)
    order = np.argsort(scores)[::-1]
    margin = float(scores[int(order[0])] - scores[int(order[1])])
    return (6, 8, 12)[int(order[0])], margin


def event_quarter(x: np.ndarray) -> tuple[int, float]:
    scores = []
    for q in range(4):
        segment = np.asarray(x[q * 12:(q + 1) * 12], dtype=float)
        scores.append(float(np.max(np.abs(segment - np.median(segment)))))
    order = np.argsort(scores)[::-1]
    return int(order[0]), float(scores[int(order[0])] - scores[int(order[1])])


def change_boundary(x: np.ndarray) -> tuple[int, float]:
    scores = []
    for boundary in (12, 24, 36):
        scores.append(abs(float(np.mean(x[boundary - 4:boundary]) - np.mean(x[boundary:boundary + 4]))))
    order = np.argsort(scores)[::-1]
    return (12, 24, 36)[int(order[0])], float(scores[int(order[0])] - scores[int(order[1])])


def robust_lag(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    def prep(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        lo, hi = np.quantile(a, [0.1, 0.9])
        a = np.clip(a, lo, hi)
        return a - a.mean()
    x, y = prep(x), prep(y)
    scores = []
    lags = list(range(-4, 5))
    for lag in lags:
        if lag > 0:
            a, b = x[:-lag], y[lag:]
        elif lag < 0:
            a, b = x[-lag:], y[:lag]
        else:
            a, b = x, y
        denom = float(a.std() * b.std())
        scores.append(-1.0 if denom < 1e-12 else float(np.corrcoef(a, b)[0, 1]))
    order = np.argsort(scores)[::-1]
    return lags[int(order[0])], float(scores[int(order[0])] - scores[int(order[1])])


def interval_name(q: int) -> str:
    return f"t={q * 12}-{q * 12 + 11}"


def lag_name(lag: int) -> str:
    if lag > 0:
        return f"sensor_D lags sensor_C by {lag} steps"
    if lag < 0:
        return f"sensor_D leads sensor_C by {-lag} steps"
    return "sensor_C and sensor_D are aligned"


def gold(task_id: str, values: np.ndarray) -> str:
    a, b, c, d = (values[:, i] for i in range(4))
    if task_id == "A1_interval_trend":
        slope = float(np.polyfit(np.arange(24), a[:24], 1)[0])
        return "rising" if slope > 0.025 else "falling" if slope < -0.025 else "stable"
    if task_id == "A2_dominant_interval":
        return f"{dominant_period(a)[0]} steps"
    if task_id == "B1_interval_mean":
        return f"{_round_half(float(a[:24].mean())):.1f} units"
    if task_id == "B2_interval_mean_difference":
        return f"{_round_half(float(a[:24].mean() - a[24:].mean())):.1f} units"
    if task_id == "C1_local_deviation_interval":
        return interval_name(event_quarter(a)[0])
    if task_id == "C2_change_point_interval":
        return f"boundary t={change_boundary(a)[0]}"
    if task_id == "D1_repeat_interval_comparison":
        pa, pb = dominant_period(a)[0], dominant_period(b)[0]
        return "sensor_A is shorter" if pa < pb else "sensor_B is shorter" if pb < pa else "same interval"
    if task_id == "D2_channel_lead_lag":
        return lag_name(robust_lag(c, d)[0])
    if task_id == "E1_repeat_and_event":
        return f"lag {dominant_period(a)[0]}; event {interval_name(event_quarter(a)[0])}"
    if task_id == "E2_before_after_structure":
        p0, p1 = dominant_period(a[:24])[0], dominant_period(a[24:])[0]
        return "shorter after t=24" if p1 < p0 else "longer after t=24" if p1 > p0 else "same interval before and after"
    if task_id == "F1_conditional_extremum":
        half = 0 if b[:24].mean() >= b[24:].mean() else 1
        start = half * 24
        q = (start + int(np.argmax(a[start:start + 24]))) // 12
        return interval_name(q)
    if task_id == "F2_event_order":
        qc, qd = event_quarter(c)[0], event_quarter(d)[0]
        return "sensor_C event is earlier" if qc < qd else "sensor_D event is earlier" if qd < qc else "events are in the same interval"
    raise KeyError(task_id)


def serialize_series(values: np.ndarray) -> str:
    rows = ["time unit=step; value unit=arbitrary_units; time_origin=0; interval=1"]
    rows.append("t," + ",".join(CHANNELS))
    for t, row in enumerate(values):
        rows.append(str(t) + "," + ",".join(f"{float(v):+.3f}" for v in row))
    return "\n".join(rows)


def parse_series(text: str) -> np.ndarray:
    lines = text.splitlines()
    if len(lines) != N_STEPS + 2 or lines[1].split(",")[1:] != list(CHANNELS):
        raise ValueError("invalid visible series header")
    out = []
    for expected_t, line in enumerate(lines[2:]):
        fields = line.split(",")
        if int(fields[0]) != expected_t:
            raise ValueError("time index changed during round trip")
        out.append([float(x) for x in fields[1:]])
    return np.asarray(out, dtype=np.float64)


def visible_hash(values: np.ndarray) -> str:
    return sha256_bytes(serialize_series(values).encode("utf-8"))


def _ar1(rng: np.random.Generator, scale: float = 0.12, phi: float = 0.65) -> np.ndarray:
    e = rng.normal(0.0, scale, N_STEPS)
    z = np.empty(N_STEPS)
    z[0] = e[0]
    for i in range(1, N_STEPS):
        z[i] = phi * z[i - 1] + e[i]
    return z


def _sine(period: int, phase: float = 0.0, amplitude: float = 1.0, length: int = N_STEPS) -> np.ndarray:
    t = np.arange(length, dtype=float)
    return amplitude * np.sin(2.0 * np.pi * t / period + phase)


def make_base(family_id: str, split: str, dataset_seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(stable_seed(DATASET_VERSION, dataset_seed, family_id))
    train_compositions = ("T", "P", "L", "TP")
    ood_compositions = ("TL", "PL", "TPL")
    choices = ood_compositions if split == "composition_ood_test" else train_compositions
    composition = choices[stable_seed(family_id, "composition") % len(choices)]
    periods = (6, 8, 12)
    p0 = int(rng.choice(periods)); p1 = int(rng.choice(periods)); pb = int(rng.choice(periods))
    phase = float(rng.uniform(0, 2 * np.pi))
    a = np.concatenate([_sine(p0, phase, 1.3, 24), _sine(p1, phase / 2, 1.3, 24)])
    b = _sine(pb, float(rng.uniform(0, 2 * np.pi)), 1.1)
    if "T" in composition:
        a += float(rng.choice((-0.055, 0.055))) * np.arange(N_STEPS)
        b += float(rng.choice((-0.035, 0.035))) * np.arange(N_STEPS)
    if "L" in composition:
        a += _ar1(rng); b += _ar1(rng)
    a += rng.normal(0, 0.04, N_STEPS); b += rng.normal(0, 0.04, N_STEPS)
    # Observed local event and change boundary; labels are later recomputed from rounded values.
    q_a = int(rng.integers(0, 4)); q_b = int(rng.integers(0, 4))
    a[q_a * 12 + int(rng.integers(3, 9))] += float(rng.choice((-1, 1))) * 5.0
    b[q_b * 12 + int(rng.integers(3, 9))] += float(rng.choice((-1, 1))) * 4.5
    cp = int(rng.choice((12, 24, 36)))
    a[cp:] += float(rng.choice((-1.0, 1.0))) * 1.4
    selected_first = bool(rng.integers(0, 2))
    b[:24] += 1.5 if selected_first else -0.5
    b[24:] += -0.5 if selected_first else 1.5

    common = _sine(int(rng.choice(periods)), float(rng.uniform(0, 2 * np.pi)), 1.0) + _ar1(rng, 0.05)
    lag = int(rng.choice((-4, 0, 4)))
    c = common + rng.normal(0, 0.025, N_STEPS)
    d = np.roll(common, lag) + rng.normal(0, 0.025, N_STEPS)
    qc, qd = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    c[qc * 12 + 6] += 4.0
    d[qd * 12 + 6] -= 4.0
    values = np.column_stack([a, b, c, d])
    # This quantization is the actual visible precision.  All downstream gold uses these values.
    values = np.round(values, 3).astype(np.float64)
    metadata = {
        "composition": composition, "difficulty": "high" if "L" in composition else "standard",
        "generation_period_A_pre": p0, "generation_period_A_post": p1,
        "generation_period_B": pb, "generation_lag_CD": lag,
        "generation_event_quarter_A": q_a, "generation_change_boundary_A": cp,
    }
    return values, metadata


def _replace_period(x: np.ndarray, period: int, length: int = N_STEPS) -> np.ndarray:
    return float(np.mean(x)) + _sine(period, 0.37, 3.0, length)


def make_relevant(base: np.ndarray, task_id: str) -> np.ndarray:
    original = gold(task_id, base)
    candidates: list[np.ndarray] = []
    if task_id == "A1_interval_trend":
        for direction in (-1.0, 1.0):
            z = base.copy(); z[:24, 0] = base[0, 0] + direction * 0.35 * np.arange(24)
            candidates.append(z)
    elif task_id == "A2_dominant_interval":
        for period in (6, 8, 12):
            z = base.copy(); z[:, 0] = _replace_period(base[:, 0], period)
            candidates.append(z)
    elif task_id in ("B1_interval_mean", "B2_interval_mean_difference"):
        for delta in (-3.0, 3.0):
            z = base.copy(); z[:24, 0] += delta; candidates.append(z)
    elif task_id in ("C1_local_deviation_interval", "E1_repeat_and_event"):
        old_q = event_quarter(base[:, 0])[0]
        for q in range(4):
            if q == old_q: continue
            z = base.copy(); z[q * 12 + 5, 0] = float(np.max(np.abs(base[:, 0]))) + 25.0
            candidates.append(z)
    elif task_id == "C2_change_point_interval":
        old = change_boundary(base[:, 0])[0]
        for boundary in (12, 24, 36):
            if boundary == old: continue
            z = base.copy(); z[:, 0] = _sine(8, 0.2, 0.2)
            z[boundary:, 0] += 12.0; candidates.append(z)
    elif task_id == "D1_repeat_interval_comparison":
        z = base.copy(); z[:, 1] = base[:, 0]; candidates.append(z)
        for period in (6, 8, 12):
            z = base.copy(); z[:, 1] = _replace_period(base[:, 1], period); candidates.append(z)
    elif task_id == "D2_channel_lead_lag":
        current = robust_lag(base[:, 2], base[:, 3])[0]
        for lag in (-4, 0, 4):
            if lag == current: continue
            z = base.copy(); z[:, 3] = np.roll(base[:, 2], lag); candidates.append(z)
    elif task_id == "E2_before_after_structure":
        for p0, p1 in ((6, 8), (8, 12), (12, 8), (8, 8), (6, 12), (12, 6)):
            z = base.copy()
            z[:24, 0] = _replace_period(base[:24, 0], p0, 24)
            z[24:, 0] = _replace_period(base[24:, 0], p1, 24)
            candidates.append(z)
    elif task_id == "F1_conditional_extremum":
        first = bool(base[:24, 1].mean() >= base[24:, 1].mean())
        z = base.copy()
        z[:24, 1] += -8.0 if first else 8.0
        z[24:, 1] += 8.0 if first else -8.0
        candidates.append(z)
    elif task_id == "F2_event_order":
        for qc, qd in ((0, 3), (3, 0), (1, 1)):
            z = base.copy()
            z[:, 2] = _sine(8, 0.1, 0.3); z[:, 3] = _sine(8, 0.4, 0.3)
            z[qc * 12 + 5, 2] += 20.0; z[qd * 12 + 5, 3] -= 20.0
            candidates.append(z)
    else:
        raise KeyError(task_id)
    control = CONTROL_TASK[task_id]
    for candidate in candidates:
        candidate = np.round(candidate, 3)
        if gold(task_id, candidate) != original and gold(control, candidate) == gold(control, base):
            return candidate
    raise RuntimeError(f"could not construct verified relevant edit for {task_id}: {original}")


def make_irrelevant(base: np.ndarray, task_id: str) -> np.ndarray:
    z = base.copy()
    # A/B questions get an affine edit to C.  C/D questions get an affine edit to B.
    # These edits are visible but provably outside the corresponding target programs.
    channel = 1 if task_id in ("D2_channel_lead_lag", "F2_event_order") else 2
    z[:, channel] += 1.234
    z = np.round(z, 3)
    if gold(task_id, z) != gold(task_id, base):
        raise RuntimeError(f"irrelevant edit changed gold for {task_id}")
    return z


def _numeric_options(base_answer: str, relevant_answer: str) -> list[str]:
    def number(s: str) -> float: return float(s.split()[0])
    vals = [number(base_answer), number(relevant_answer)]
    for offset in (0.5, -0.5, 1.0, -1.0, 1.5, -1.5):
        candidate = vals[0] + offset
        if candidate not in vals: vals.append(candidate)
        if len(vals) == 4: break
    return [f"{x:.1f} units" for x in vals]


def semantic_options(task_id: str, base_answer: str, relevant_answer: str) -> list[str]:
    static = {
        "A1_interval_trend": ["rising", "falling", "stable", "not uniquely determined"],
        "A2_dominant_interval": ["6 steps", "8 steps", "12 steps", "no candidate lag"],
        "C1_local_deviation_interval": [interval_name(i) for i in range(4)],
        "C2_change_point_interval": ["boundary t=12", "boundary t=24", "boundary t=36", "no candidate boundary"],
        "D1_repeat_interval_comparison": ["sensor_A is shorter", "sensor_B is shorter", "same interval", "not uniquely determined"],
        "E2_before_after_structure": ["shorter after t=24", "longer after t=24", "same interval before and after", "not uniquely determined"],
        "F1_conditional_extremum": [interval_name(i) for i in range(4)],
        "F2_event_order": ["sensor_C event is earlier", "sensor_D event is earlier", "events are in the same interval", "not uniquely determined"],
    }
    if task_id in static:
        options = static[task_id]
    elif task_id in ("B1_interval_mean", "B2_interval_mean_difference"):
        options = _numeric_options(base_answer, relevant_answer)
    elif task_id == "D2_channel_lead_lag":
        options = [base_answer, relevant_answer]
        for lag in (-4, -2, 0, 2, 4):
            name = lag_name(lag)
            if name not in options: options.append(name)
            if len(options) == 4: break
    elif task_id == "E1_repeat_and_event":
        options = [base_answer, relevant_answer]
        for period in (6, 8, 12):
            for q in range(4):
                name = f"lag {period}; event {interval_name(q)}"
                if name not in options: options.append(name)
                if len(options) == 4: break
            if len(options) == 4: break
    else:
        raise KeyError(task_id)
    if base_answer not in options or relevant_answer not in options or len(set(options)) != 4:
        raise AssertionError((task_id, base_answer, relevant_answer, options))
    return list(options)


def permute_options(options: list[str], family_id: str, task_id: str) -> dict[str, str]:
    rng = np.random.default_rng(stable_seed(DATASET_VERSION, family_id, task_id, "options"))
    perm = rng.permutation(len(options))
    return {LABELS[i]: options[int(perm[i])] for i in range(4)}


def _gold_option(options: dict[str, str], semantic_gold: str) -> str:
    matches = [label for label, value in options.items() if value == semantic_gold]
    if len(matches) != 1:
        raise AssertionError((semantic_gold, options))
    return matches[0]


def split_assignment(n_families: int = 900) -> dict[str, str]:
    if n_families != 900:
        raise ValueError("protocol v1 fixes exactly 900 main families")
    ids = [f"F{i:04d}" for i in range(n_families)]
    ranked = sorted(ids, key=lambda x: stable_seed(DATASET_VERSION, "split", x))
    result = {}
    cuts = ((400, "train"), (100, "validation"), (200, "iid_test"), (200, "composition_ood_test"))
    start = 0
    for count, split in cuts:
        for family_id in ranked[start:start + count]: result[family_id] = split
        start += count
    return result


ITEM_FIELDS = [
    "dataset_version", "protocol_version", "family_id", "series_id", "series_index",
    "item_id", "question_id", "task_id", "category", "split", "variant_id", "condition",
    "contrast_group_id", "expected_relation", "target_program", "question_text", "options_json",
    "semantic_gold", "gold_option", "answer_type", "control_question_id", "difficulty",
    "composition", "visible_input_sha256",
]


def _atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_dataset(out_dir: Path, dataset_seed: int = 20260918) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments = split_assignment()
    family_ids = sorted(assignments)
    variant_ids = ["base"] + [f"relevant_{t}" for t in TASK_IDS] + [f"irrelevant_{t}" for t in TASK_IDS]
    arrays = np.empty((len(family_ids), len(variant_ids), N_STEPS, len(CHANNELS)), dtype=np.float32)
    rows: list[dict] = []
    family_rows: list[dict] = []
    for family_index, family_id in enumerate(family_ids):
        split = assignments[family_id]
        base, meta = make_base(family_id, split, dataset_seed)
        base = parse_series(serialize_series(base))
        variants = {"base": base}
        for task_id in TASK_IDS:
            variants[f"relevant_{task_id}"] = make_relevant(base, task_id)
            variants[f"irrelevant_{task_id}"] = make_irrelevant(base, task_id)
        for variant_index, variant_id in enumerate(variant_ids):
            arrays[family_index, variant_index] = variants[variant_id]
        family_rows.append({
            "dataset_version": DATASET_VERSION, "family_id": family_id, "family_index": family_index,
            "split": split, "composition": meta["composition"], "difficulty": meta["difficulty"],
            "base_visible_sha256": visible_hash(base), "latent_audit_json": json.dumps(meta, sort_keys=True),
        })
        question_ids = {task: f"{family_id}_{task}" for task in TASK_IDS}
        for task_id in TASK_IDS:
            spec = TASK_SPECS[task_id]
            relevant = variants[f"relevant_{task_id}"]
            irrelevant = variants[f"irrelevant_{task_id}"]
            base_answer, rel_answer = gold(task_id, base), gold(task_id, relevant)
            options = permute_options(semantic_options(task_id, base_answer, rel_answer), family_id, task_id)
            control_task = CONTROL_TASK[task_id]
            contrast = f"{family_id}_{task_id}_cross"
            records = [
                ("base", task_id, base, "FULL", "REFERENCE", contrast),
                (f"relevant_{task_id}", task_id, relevant, "RELEVANT_EDIT", "CHANGE", contrast),
                (f"irrelevant_{task_id}", task_id, irrelevant, "IRRELEVANT_EDIT", "INVARIANT", f"{family_id}_{task_id}_irrelevant"),
            ]
            # Explicit fourth cell: q_control on the target's relevant transform.  Its own base row
            # is linked by control_question_id and can be recovered without duplicating arrays.
            control_spec = TASK_SPECS[control_task]
            control_base = gold(control_task, base)
            control_rel = gold(control_task, relevant)
            control_changed = gold(control_task, variants[f"relevant_{control_task}"])
            control_options = permute_options(
                semantic_options(control_task, control_base, control_changed),
                family_id, control_task)
            if control_base != control_rel:
                raise AssertionError(f"cross-control changed for {family_id}/{task_id}/{control_task}")
            for variant_id, row_task, values, condition, relation, group in records:
                answer = gold(row_task, values)
                series_index = variant_ids.index(variant_id)
                rows.append({
                    "dataset_version": DATASET_VERSION, "protocol_version": PROTOCOL_VERSION,
                    "family_id": family_id, "series_id": f"{family_id}:{variant_id}",
                    "series_index": series_index, "item_id": f"{family_id}:{task_id}:{condition}",
                    "question_id": question_ids[row_task], "task_id": row_task, "category": spec.category,
                    "split": split, "variant_id": variant_id, "condition": condition,
                    "contrast_group_id": group, "expected_relation": relation,
                    "target_program": spec.target_program, "question_text": spec.question,
                    "options_json": json.dumps(options, sort_keys=True), "semantic_gold": answer,
                    "gold_option": _gold_option(options, answer), "answer_type": spec.answer_type,
                    "control_question_id": question_ids[control_task], "difficulty": meta["difficulty"],
                    "composition": meta["composition"], "visible_input_sha256": visible_hash(values),
                })
            rows.append({
                "dataset_version": DATASET_VERSION, "protocol_version": PROTOCOL_VERSION,
                "family_id": family_id, "series_id": f"{family_id}:relevant_{task_id}",
                "series_index": variant_ids.index(f"relevant_{task_id}"),
                "item_id": f"{family_id}:{task_id}:CROSS_CONTROL:{control_task}",
                "question_id": question_ids[control_task], "task_id": control_task,
                "category": control_spec.category, "split": split,
                "variant_id": f"relevant_{task_id}", "condition": "CROSS_CONTROL",
                "contrast_group_id": contrast, "expected_relation": "INVARIANT",
                "target_program": control_spec.target_program, "question_text": control_spec.question,
                "options_json": json.dumps(control_options, sort_keys=True), "semantic_gold": control_rel,
                "gold_option": _gold_option(control_options, control_rel), "answer_type": control_spec.answer_type,
                "control_question_id": question_ids[control_task], "difficulty": meta["difficulty"],
                "composition": meta["composition"], "visible_input_sha256": visible_hash(relevant),
            })
    _atomic_csv(out_dir / "item_manifest.csv", rows, ITEM_FIELDS)
    _atomic_csv(out_dir / "split_manifest.csv", family_rows, list(family_rows[0]))
    npz_tmp = out_dir / "series_arrays.tmp.npz"
    np.savez_compressed(npz_tmp, values=arrays, family_ids=np.asarray(family_ids),
                        variant_ids=np.asarray(variant_ids), timestamps=np.arange(N_STEPS),
                        channels=np.asarray(CHANNELS))
    os.replace(npz_tmp, out_dir / "series_arrays.npz")
    metadata = {
        "dataset_version": DATASET_VERSION, "protocol_version": PROTOCOL_VERSION,
        "dataset_seed": dataset_seed, "n_families": len(family_ids), "n_items": len(rows),
        "shape": list(arrays.shape), "family_counts": {s: list(assignments.values()).count(s) for s in sorted(set(assignments.values()))},
        "item_manifest_sha256": sha256_bytes((out_dir / "item_manifest.csv").read_bytes()),
        "split_manifest_sha256": sha256_bytes((out_dir / "split_manifest.csv").read_bytes()),
        "series_arrays_sha256": sha256_bytes((out_dir / "series_arrays.npz").read_bytes()),
    }
    _atomic_json(out_dir / "dataset_metadata.json", metadata)
    return metadata


def load_dataset(root: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    root = Path(root)
    with (root / "item_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    raw = np.load(root / "series_arrays.npz", allow_pickle=False)
    arrays = {key: raw[key] for key in raw.files}
    return rows, arrays


def row_values(row: dict, arrays: dict[str, np.ndarray]) -> np.ndarray:
    family_lookup = {str(x): i for i, x in enumerate(arrays["family_ids"])}
    return np.asarray(arrays["values"][family_lookup[row["family_id"]], int(row["series_index"])], dtype=float)
