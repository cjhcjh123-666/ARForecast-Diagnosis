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
TEMPORAL_BLOCK_RE = re.compile(
    r"(?im)^(?P<name>(?:prefix|full)?\s*(?:irregular\s+)?(?:time series|timestamps?))"
    r"\s*:\s*(?P<array>\[[^\[\]\r\n]*\])"
)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_temporal_blocks(question: str) -> list[dict[str, Any]]:
    """Return every labelled series/timestamp array and its exact source span.

    IRTS temporal-relationship items contain three relevant arrays (prefix values, prefix times,
    and full times).  The legacy adapter kept only the first bracketed array.
    """
    return [
        {
            "name": m.group("name"),
            "array": m.group("array"),
            "start": m.start("array"),
            "end": m.end("array"),
        }
        for m in TEMPORAL_BLOCK_RE.finditer(question)
    ]


def prepare_item(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    item["temporal_blocks"] = extract_temporal_blocks(item["question"])
    return item


def load_items() -> list[dict[str, Any]]:
    df = pd.read_parquet(SOURCE)
    items = []
    for _, r in df.iterrows():
        q = str(r["question"])
        item = dict(id=int(r["question_id"]), task_type=str(r["task_type"]), question=q,
                    answer_format=str(r["answer_format"]), gold=str(r["answer"]).strip())
        items.append(prepare_item(item))
    return items


def _values(series_str: str) -> list:
    try:
        return json.loads(series_str)
    except Exception:
        return [float(x) if x not in ("None", "null", "nan") else None for x in NUM_RE.findall(series_str)]


def _array_lexemes(array: str) -> list[str]:
    inner = array.strip()[1:-1]
    return [part.strip() for part in inner.split(",")] if inner.strip() else []


def _permute_array(array: str, seed: int, reverse: bool = False) -> str:
    """Reorder exact lexical values; never round or reformat them."""
    values = _array_lexemes(array)
    movable = [i for i, value in enumerate(values) if value.lower() not in {"none", "null", "nan"}]
    if reverse:
        reordered = list(reversed([values[i] for i in movable]))
    else:
        import numpy as np
        rng = np.random.default_rng(seed)
        reordered = [values[i] for i in np.asarray(movable)[rng.permutation(len(movable))]]
    out = list(values)
    for pos, value in zip(movable, reordered):
        out[pos] = value
    return "[" + ", ".join(out) + "]"


def _replace_blocks(question: str, replacements: list[str]) -> str:
    blocks = extract_temporal_blocks(question)
    if len(blocks) != len(replacements):
        raise ValueError("temporal block/replacement count mismatch")
    out, cursor = [], 0
    for block, replacement in zip(blocks, replacements):
        out.extend((question[cursor:block["start"]], replacement))
        cursor = block["end"]
    out.append(question[cursor:])
    return "".join(out)


def render_question(item: dict[str, Any]) -> str:
    """Question branch with every labelled temporal array removed."""
    blocks = extract_temporal_blocks(item["question"])
    text = _replace_blocks(item["question"], ["[temporal evidence omitted]"] * len(blocks))
    return text.strip() + answer_suffix(item)


def render_temporal(item: dict[str, Any], condition: str = "full", seed: int = 0) -> str:
    """Temporal branch containing labelled values/times and no question or answer choices."""
    blocks = extract_temporal_blocks(item["question"])
    if condition == "question_only":
        return "Temporal evidence: [not available]"
    lines = []
    for block in blocks:
        array = block["array"]
        is_series = "series" in block["name"].lower()
        if is_series and condition == "shuffled_ts":
            array = _permute_array(array, 1000 + seed + int(item["id"]))
        elif is_series and condition == "reversed_ts":
            array = _permute_array(array, 0, reverse=True)
        elif condition not in {"full", "shuffled_ts", "reversed_ts"}:
            raise ValueError(condition)
        lines.append(f"{block['name']}: {array}")
    return "\n".join(lines)


def series_lexemes(text: str) -> list[str]:
    values = []
    for block in extract_temporal_blocks(text):
        if "series" in block["name"].lower():
            values.extend(_array_lexemes(block["array"]))
    return values


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
    blocks = extract_temporal_blocks(item["question"])
    if condition == "full":
        question = item["question"]
    elif condition == "question_only":
        question = _replace_blocks(item["question"], ["[temporal evidence omitted]"] * len(blocks))
    elif condition in {"shuffled_ts", "reversed_ts"}:
        replacements = []
        for block in blocks:
            if "series" not in block["name"].lower():
                replacements.append(block["array"])
            elif condition == "shuffled_ts":
                replacements.append(_permute_array(
                    block["array"], 1000 + seed + int(item["id"])
                ))
            else:
                replacements.append(_permute_array(block["array"], 0, reverse=True))
        question = _replace_blocks(item["question"], replacements)
    else:
        raise ValueError(condition)
    return question.strip() + answer_suffix(item)


_ANSWER_MARKER_RE = re.compile(
    r"(?im)\b(?:final answer|answer|choice)\s*[:\-]?\s*\**\s*(true|false|[a-e]|[tf])\b"
)
_STANDALONE_LINE_RE = re.compile(r"(?im)^\s*\**\s*(true|false|[a-e]|[tf])\s*\**\s*$")
_BOLDED_TOKEN_RE = re.compile(r"(?i)\*\*\s*(true|false|[a-e]|[tf])\s*\*\*")
_LEADING_TOKEN_RE = re.compile(r"(?i)^\s*[\(\[\{<\"'`*_#-]*\s*(true|false|[a-e]|[tf])\b")
_TRAILING_TOKEN_RE = re.compile(
    r"(?i)\b(true|false|[a-e]|[tf])\b\s*[\)\]\}>\'\"`*_.!?,:;~-]*\s*$"
)


def _normalize_answer(candidate: str, answer_format: str) -> str | None:
    value = candidate.strip().lower()
    if answer_format == "true_false":
        return "T" if value in {"t", "true"} else "F" if value in {"f", "false"} else None
    token = value.upper()
    return token if len(token) == 1 and token in LABELS.get(answer_format, []) else None


def parse(text: str, answer_format: str) -> str | None:
    """Match the released IRTS evaluator's final-answer extraction protocol."""
    stripped = text.strip()
    if not stripped:
        return None
    candidates = (stripped[-256:], "\n".join(stripped.splitlines()[-8:]), stripped)
    for candidate_text in candidates:
        for pattern in (_ANSWER_MARKER_RE, _STANDALONE_LINE_RE, _BOLDED_TOKEN_RE,
                        _TRAILING_TOKEN_RE):
            for candidate in reversed(pattern.findall(candidate_text)):
                token = _normalize_answer(candidate, answer_format)
                if token is not None:
                    return token
    leading = _LEADING_TOKEN_RE.match(stripped)
    if leading:
        token = _normalize_answer(leading.group(1), answer_format)
        if token is not None:
            return token
    normalized = stripped.lower()
    if normalized.startswith("option "):
        token = _normalize_answer(normalized.rsplit(" ", 1)[-1], answer_format)
        if token is not None:
            return token
    return _normalize_answer(normalized, answer_format)


def is_correct(pred: str | None, gold: str) -> bool:
    return pred is not None and pred.strip().upper() == gold.strip().upper()


def manifest() -> dict:
    return dict(benchmark="IRTS-ToolBench", source=str(SOURCE), n_items=len(load_items()),
                sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                task_types={t: sum(1 for i in load_items() if i["task_type"] == t)
                            for t in sorted({i["task_type"] for i in load_items()})},
                answer_formats={f: sum(1 for i in load_items() if i["answer_format"] == f)
                                for f in sorted({i["answer_format"] for i in load_items()})})
