"""Evaluation helpers with explicit metric direction and full-denominator pairing."""
from __future__ import annotations

from collections import defaultdict


LOWER_IS_BETTER = {"mae", "nmae", "mse", "rmse", "interval_error", "position_error"}


def improvement(metric: str, pretrained: float, random: float) -> float:
    """Return positive values when pretrained is better, retaining raw scores elsewhere."""
    return random - pretrained if metric.lower() in LOWER_IS_BETTER else pretrained - random


def paired_metrics(predictions: list[dict]) -> list[dict]:
    """Compute correctness-based pair metrics without dropping parse failures.

    Each prediction must carry family_id, task_id, condition, and correct. Missing counterparts
    stay in the denominator and make the corresponding pair incorrect.
    """
    grouped = defaultdict(dict)
    for row in predictions:
        grouped[(row["family_id"], row["task_id"])][row["condition"]] = bool(row.get("correct", False))
    pairs = {
        "relevant_edit_pair_accuracy": ("FULL", "RELEVANT_EDIT"),
        "irrelevant_edit_pair_accuracy": ("FULL", "IRRELEVANT_EDIT"),
    }
    out = []
    for name, conditions in pairs.items():
        scores = [all(rows.get(c, False) for c in conditions) for rows in grouped.values()]
        out.append({"metric": name, "numerator": sum(scores), "denominator": len(scores),
                    "value": sum(scores) / len(scores) if scores else None})
    return out
