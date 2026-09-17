"""IRTS-ToolBench adapter (read-only).

Source of truth: `ChatTS-Training-main/WaveTLM/third_party/IRTS-ToolBench/benchmark/irts_cleaned_benchmark.parquet`
(1700 items, 10 task types, MC-ABCD/ABC/AB + true/false). We do **not** redefine the benchmark:
the question text, the gold answers and the answer-format taxonomy are used verbatim; we only
(a) split the inline series out of the question so the four input conditions can be built, and
(b) parse the model answer into the official label alphabet.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/9950backfile/chenjiahui/ChatTS-Training-main/WaveTLM/third_party/"
              "IRTS-ToolBench/benchmark/irts_cleaned_benchmark.parquet")
LABELS = {"multiple_choice_abcd": ["A", "B", "C", "D"], "multiple_choice_abc": ["A", "B", "C"],
          "multiple_choice_ab": ["A", "B"], "true_false": ["T", "F"]}
SERIES_RE = re.compile(r"(\[[^\[\]]*\])")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_items() -> list[dict[str, Any]]:
    df = pd.read_parquet(SOURCE)
    items = []
    for _, r in df.iterrows():
        q = str(r["question"])
        m = SERIES_RE.search(q)
        series_str = m.group(1) if m else ""
        items.append(dict(id=int(r["question_id"]), task_type=str(r["task_type"]), question=q,
                          series_str=series_str, pre=q[:m.start()] if m else q,
                          post=q[m.end():] if m else "",
                          answer_format=str(r["answer_format"]), gold=str(r["answer"]).strip()))
    return items


def _values(series_str: str) -> list:
    try:
        return json.loads(series_str)
    except Exception:
        return [float(x) if x not in ("None", "null", "nan") else None for x in NUM_RE.findall(series_str)]


def _fmt(values) -> str:
    return json.dumps([None if v is None else round(float(v), 2) for v in values])


INSTRUCTION = {
    "multiple_choice_abcd": "Answer with a single letter: A, B, C or D.",
    "multiple_choice_abc": "Answer with a single letter: A, B or C.",
    "multiple_choice_ab": "Answer with a single letter: A or B.",
    "true_false": "Answer with a single letter: T or F.",
}


def answer_suffix(item: dict) -> str:
    """Protocol-locked answer-format instruction (identical for every model and condition).

    Rationale: the released IRTS items are written for an instruction-following interface. A raw
    completion prompt makes a BASE language model continue the series numerically instead of
    answering, which measures formatting, not temporal reasoning. The suffix only states the answer
    format and never contains the answer.
    """
    ins = INSTRUCTION.get(item.get("answer_format", "multiple_choice_abcd"),
                          "Answer with a single letter.")
    return "\n" + ins + "\nAnswer:"


def render(item: dict, condition: str = "full", seed: int = 0) -> str:
    """Build the LM input for one condition.

    full          : untouched benchmark question
    question_only : the series block is removed (language prior / no-evidence control)
    shuffled_ts   : same values, random temporal order (multiset preserved, None positions kept)
    reversed_ts   : values reversed in time
    """
    if condition == "full":
        return item["question"] + answer_suffix(item)
    if condition == "question_only":
        return (item["pre"] + "[the time series is not available]" + item["post"]).strip() + answer_suffix(item)
    vals = _values(item["series_str"])
    if condition == "shuffled_ts":
        import numpy as np
        rng = np.random.default_rng(1000 + seed + item["id"])
        idx = [i for i, v in enumerate(vals) if v is not None]
        perm = rng.permutation(len(idx))
        out = list(vals)
        for pos, src in zip(idx, [idx[p] for p in perm]):
            out[pos] = vals[src]
        vals = out
    elif condition == "reversed_ts":
        vals = list(reversed(vals))
    else:
        raise ValueError(condition)
    return item["pre"] + _fmt(vals) + item["post"] + answer_suffix(item)


def parse(text: str, answer_format: str) -> str | None:
    """Official label alphabet; first standalone label wins (case-insensitive)."""
    labels = LABELS.get(answer_format, ["A", "B", "C", "D"])
    up = text.strip().upper()
    for lab in labels:
        if re.search(rf"(^|[^A-Z]){lab}([^A-Z]|$)", up):
            return lab
    return None


def is_correct(pred: str | None, gold: str) -> bool:
    return pred is not None and pred.strip().upper() == gold.strip().upper()


def manifest() -> dict:
    return dict(benchmark="IRTS-ToolBench", source=str(SOURCE), n_items=len(load_items()),
                sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                task_types={t: sum(1 for i in load_items() if i["task_type"] == t)
                            for t in sorted({i["task_type"] for i in load_items()})},
                answer_formats={f: sum(1 for i in load_items() if i["answer_format"] == f)
                                for f in sorted({i["answer_format"] for i in load_items()})})
