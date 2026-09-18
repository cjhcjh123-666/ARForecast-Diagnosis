"""Strictly separated rendering for temporal, question, and native QA interfaces."""
from __future__ import annotations

import json

import numpy as np

from .benchmark import CHANNELS, N_STEPS, serialize_series


def render_temporal(values: np.ndarray) -> str:
    """Temporal branch: visible evidence and permitted units only."""
    return "TIME-SERIES EVIDENCE ONLY\n" + serialize_series(values)


def render_question_option(row: dict, option_label: str) -> str:
    """Question branch: question plus one candidate, with no observed values."""
    options = json.loads(row["options_json"])
    return (
        "QUESTION AND CANDIDATE ONLY\n"
        f"Question: {row['question_text']}\n"
        f"Candidate {option_label}: {options[option_label]}"
    )


def render_question_only(row: dict) -> str:
    options = json.loads(row["options_json"])
    choices = "\n".join(f"{label}. {options[label]}" for label in ("A", "B", "C", "D"))
    return (
        f"Metadata: {N_STEPS} observations, t=0..{N_STEPS - 1}, interval=1 step, "
        f"channels={','.join(CHANNELS)}, value_unit=arbitrary_units.\n"
        f"Question: {row['question_text']}\nOptions:\n{choices}\nAnswer with one option label."
    )


def render_native(row: dict, values: np.ndarray | None) -> str:
    question = render_question_only(row)
    if values is None:
        return "The numerical time-series evidence is unavailable.\n" + question
    return serialize_series(values) + "\n\n" + question
