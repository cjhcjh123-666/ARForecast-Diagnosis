#!/usr/bin/env python3
"""Assemble complete cross-model Real-TSQA A/B tables from paired summaries."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("results/tsqa_evidence/real_v1/tables")
MODELS = (
    "qwen3_8b_base",
    "llama31_8b",
    "llama32_3b",
    "gemma2_9b",
    "gemma2_2b",
    "mistral_7b_v03",
    "deepseek_llm_7b",
    "deepseek_v2_lite",
    "olmo2_7b",
    "olmo2_13b",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8:
        raise ValueError(f"expected 8 locked metrics in {path}, found {len(rows)}")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    a_rows: list[dict[str, str]] = []
    b_rows: list[dict[str, str]] = []
    same_sample: list[dict[str, str]] = []
    for model in MODELS:
        model_a = read_rows(ROOT / f"{model}_readout_pr_paired_bootstrap.csv")
        model_b = read_rows(ROOT / f"{model}_pr_paired_bootstrap.csv")
        a_rows.extend(model_a)
        b_rows.extend(model_b)
        same_sample.extend({"interface": "A_controlled_readout", **row} for row in model_a)
        same_sample.extend({"interface": "B_question_conditioned_supervised_QA", **row} for row in model_b)

    write_rows(ROOT / "interface_a_complete_models.csv", a_rows)
    write_rows(ROOT / "interface_b_complete_models.csv", b_rows)
    write_rows(ROOT / "same_sample_interface_comparison.csv", same_sample)
    print(f"models={len(MODELS)} A_rows={len(a_rows)} B_rows={len(b_rows)} same_sample_rows={len(same_sample)}")


if __name__ == "__main__":
    main()
