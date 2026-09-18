"""Protocol-locked rendering and strict answer parsing for real TSQA v1."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np


PRECISION = 4
STRICT_ANSWER = re.compile(r"^\s*(?:Answer\s*:\s*)?([A-D])\s*[.]?\s*$", re.IGNORECASE)


def _array(family: dict[str, Any], variant_id: str) -> np.ndarray:
    key = "original_evidence" if variant_id == "x" else "transformed_evidence"
    return np.asarray(family[key], dtype=float)


def _stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(parts).encode()).digest()[:8], "big")


def temporal_payload(family: dict[str, Any], variant_id: str = "x", condition: str = "full") -> dict:
    values = _array(family, variant_id).copy()
    timestamps = list(family["evidence_schema"]["timestamps"])
    if condition == "shuffled":
        rng = np.random.default_rng(_stable_seed(family["benchmark_version"], family["family_id"], variant_id, "shuffle"))
        values = values[rng.permutation(len(values))]
    elif condition != "full":
        raise ValueError(f"unsupported temporal condition: {condition}")
    return {
        "timestamps": timestamps,
        "channels": family["evidence_schema"]["channels"],
        "units": family["evidence_schema"]["units"],
        "values": [[float(f"{v:.{PRECISION}f}") for v in row] for row in values],
        "decimal_places": PRECISION,
    }


def render_temporal(family: dict[str, Any], variant_id: str = "x", condition: str = "full") -> str:
    """Temporal branch: evidence/schema only, with no question, options or gold."""
    return "TIME-SERIES EVIDENCE ONLY\n" + json.dumps(
        temporal_payload(family, variant_id, condition), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )


def render_question_option(item: dict[str, Any], option_label: str) -> str:
    """Frozen question branch input shared byte-for-byte by P/R temporal branches."""
    options = item["option_order"]
    if option_label not in options:
        raise ValueError(f"unknown option {option_label}")
    semantic = options[option_label]
    text = item["semantic_options"][semantic]
    return (
        "QUESTION AND CANDIDATE ONLY\n"
        f"Question: {item['question']}\n"
        f"Candidate {option_label}: {text}"
    )


def render_question_only(item: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{label}. {item['semantic_options'][semantic]}"
        for label, semantic in sorted(item["option_order"].items())
    )
    return f"Question: {item['question']}\nOptions:\n{choices}\nRespond with exactly one option letter.\nAnswer:"


def render_native(family: dict[str, Any], item: dict[str, Any], condition: str = "full") -> str:
    if condition == "question_only":
        evidence = "TIME-SERIES EVIDENCE\n[omitted]"
    elif condition in {"full", "shuffled"}:
        evidence = render_temporal(family, item["variant_id"], condition)
    else:
        raise ValueError(condition)
    return evidence + "\n\n" + render_question_only(item)


def parse_strict_choice(response: str, allowed: tuple[str, ...] = ("A", "B", "C", "D")) -> str | None:
    match = STRICT_ANSWER.fullmatch(response)
    if not match:
        return None
    answer = match.group(1).upper()
    return answer if answer in allowed else None
