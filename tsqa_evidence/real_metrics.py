"""Locked-formula candidates for the real evidence-family benchmark.

The functions keep parse failures in accuracy denominators and keep error
metrics valid-output-only while reporting their coverage.  Inputs are plain
records so every reported number can be rebuilt from prediction JSONL.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import numpy as np


CELLS = ("t0", "c0", "t1", "c1")


def four_cell_metrics(families: Iterable[dict]) -> dict:
    rows = list(families)
    if not rows:
        raise ValueError("at least one family is required")
    for row in rows:
        missing = [c for c in CELLS if c not in row]
        if missing:
            raise ValueError(f"family {row.get('family_id')} lacks cells {missing}")
    correct = {cell: np.asarray([bool(r[cell]) for r in rows], dtype=bool) for cell in CELLS}
    tpair = correct["t0"] & correct["t1"]
    cpair = correct["c0"] & correct["c1"]
    joint = tpair & cpair
    qswitch0 = correct["t0"] & correct["c0"]
    qswitch1 = correct["t1"] & correct["c1"]
    valid = np.asarray([
        all(bool(r.get(f"valid_{cell}", True)) for cell in CELLS) for r in rows
    ], dtype=bool)

    def conditional(numerator: np.ndarray, condition: np.ndarray) -> dict:
        denominator = int(condition.sum())
        return {
            "numerator": int((numerator & condition).sum()),
            "denominator": denominator,
            "value": float((numerator & condition).sum() / denominator) if denominator else None,
        }

    target_valid = np.asarray([
        bool(r.get("valid_t0", True)) and bool(r.get("valid_t1", True)) for r in rows
    ])
    control_valid = np.asarray([
        bool(r.get("valid_c0", True)) and bool(r.get("valid_c1", True)) for r in rows
    ])
    target_changed = np.asarray([
        r.get("pred_t0") != r.get("pred_t1") for r in rows
    ])
    control_changed = np.asarray([
        r.get("pred_c0") != r.get("pred_c1") for r in rows
    ])

    result = {
        "n_families": len(rows),
        **{f"acc_{cell}": float(correct[cell].mean()) for cell in CELLS},
        "fcjs": float(joint.mean()),
        "tus": float(tpair.mean()),
        "cis": float(cpair.mean()),
        "qsja": float((qswitch0.mean() + qswitch1.mean()) / 2),
        "vor": float(valid.mean()),
        "tus_given_t0_correct": conditional(correct["t1"], correct["t0"]),
        "cis_given_c0_correct": conditional(correct["c1"], correct["c0"]),
        "target_prediction_change_rate": conditional(target_changed, target_valid),
        "control_spurious_change_rate": conditional(control_changed, control_valid),
    }
    assert_metric_consistency(result)
    return result


def assert_metric_consistency(metrics: dict, atol: float = 1e-12) -> None:
    fcjs, tus, cis, qsja = (float(metrics[k]) for k in ("fcjs", "tus", "cis", "qsja"))
    if fcjs > tus + atol:
        raise AssertionError("FCJS must be <= TUS")
    if fcjs > cis + atol:
        raise AssertionError("FCJS must be <= CIS")
    if fcjs > qsja + atol:
        raise AssertionError("FCJS must be <= QSJA")
    if fcjs + atol < max(0.0, tus + cis - 1.0):
        raise AssertionError("FCJS violates the TUS/CIS intersection lower bound")


def numeric_metrics(rows: Iterable[dict], *, abs_tol: float, rel_tol: float,
                    scale_task: float) -> dict:
    if abs_tol < 0 or rel_tol < 0:
        raise ValueError("tolerances must be nonnegative")
    if not np.isfinite(scale_task) or scale_task <= 0:
        raise ValueError("scale_task must be a positive train-derived constant")
    records = list(rows)
    if not records:
        raise ValueError("at least one numeric item is required")
    valid_errors: list[float] = []
    successes = 0
    valid_count = 0
    for row in records:
        valid = bool(row.get("valid_output", False))
        try:
            pred = float(row["prediction"])
            gold = float(row["gold"])
            valid = valid and np.isfinite(pred) and np.isfinite(gold)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            continue
        valid_count += 1
        error = abs(pred - gold)
        valid_errors.append(error)
        tolerance = max(abs_tol, rel_tol * abs(gold))
        successes += int(error <= tolerance)
    return {
        "n_total": len(records), "valid_n": valid_count,
        "vor": valid_count / len(records),
        "tolerance_success_full_denominator": successes / len(records),
        "tolerance_success_numerator": successes,
        "mae_valid_only": float(np.mean(valid_errors)) if valid_errors else None,
        "nmae_valid_only": float(np.mean(valid_errors) / scale_task) if valid_errors else None,
        "scale_task": float(scale_task), "abs_tol": float(abs_tol), "rel_tol": float(rel_tol),
    }


def aligned_improvement(metric_name: str, pretrained: float, random_init: float) -> dict:
    lower_is_better = metric_name.lower() in {"mae", "nmae", "mse", "rmse", "position_error", "interval_error"}
    raw_delta = pretrained - random_init
    return {
        "metric": metric_name, "pretrained": pretrained, "random_init": random_init,
        "raw_delta_pretrained_minus_random": raw_delta,
        "aligned_improvement_positive_is_better": -raw_delta if lower_is_better else raw_delta,
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def hierarchical_macro(rows: Iterable[dict], value_key: str = "value") -> dict:
    """Apply family -> group -> task -> domain -> domain-macro aggregation."""
    records = list(rows)
    if not records:
        raise ValueError("at least one row is required")
    required = {"family_id", "group_id", "task_type", "domain", value_key}
    for row in records:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"aggregation row misses {sorted(missing)}")

    def means_by(items: list[dict], keys: tuple[str, ...]) -> list[dict]:
        grouped: dict[tuple, list[float]] = defaultdict(list)
        for item in items:
            grouped[tuple(item[k] for k in keys)].append(float(item[value_key]))
        return [{**dict(zip(keys, key)), value_key: float(np.mean(values))} for key, values in grouped.items()]

    family = means_by(records, ("domain", "task_type", "group_id", "family_id"))
    group = means_by(family, ("domain", "task_type", "group_id"))
    task = means_by(group, ("domain", "task_type"))
    domain = means_by(task, ("domain",))
    return {
        "headline_domain_macro": float(np.mean([r[value_key] for r in domain])),
        "domain_scores": {r["domain"]: r[value_key] for r in domain},
        "raw_pooled_descriptive": float(np.mean([float(r[value_key]) for r in records])),
        "counts": {
            "records": len(records), "groups": len({r["group_id"] for r in records}),
            "families": len({r["family_id"] for r in records}),
            "qa_items": len(records),
        },
    }


def cluster_bootstrap(rows: Iterable[dict], statistic: Callable[[list[dict]], float], *,
                      iterations: int = 2000, seed: int = 20260918) -> dict:
    records = list(rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[str(row["group_id"])].append(row)
    keys = sorted(groups)
    if not keys:
        raise ValueError("at least one group is required")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sample_keys = rng.choice(keys, size=len(keys), replace=True)
        sample = [row for key in sample_keys for row in groups[str(key)]]
        estimates.append(float(statistic(sample)))
    return {
        "estimate": float(statistic(records)), "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)), "n_groups": len(keys),
        "iterations": iterations, "seed": seed, "ci_stability": "UNSTABLE_FEW_GROUPS" if len(keys) < 20 else "ADEQUATE_GROUP_COUNT",
    }
