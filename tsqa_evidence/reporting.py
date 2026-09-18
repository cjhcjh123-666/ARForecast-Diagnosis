"""Gate-1 evidence examples and audit reports generated only from machine artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .benchmark import TASK_IDS, TASK_SPECS, load_dataset, row_values, serialize_series, stable_seed


CATEGORY_TASKS = {
    "A": ("A1_interval_trend", "A2_dominant_interval"),
    "B": ("B1_interval_mean", "B2_interval_mean_difference"),
    "C": ("C1_local_deviation_interval", "C2_change_point_interval"),
    "D": ("D1_repeat_interval_comparison", "D2_channel_lead_lag"),
    "E": ("E1_repeat_and_event", "E2_before_after_structure"),
    "F": ("F1_conditional_extremum", "F2_event_order"),
}


def build_review_examples(root: Path, docs_dir: Path) -> list[dict]:
    root, docs_dir = Path(root), Path(docs_dir)
    rows, arrays = load_dataset(root)
    by = {(r["family_id"], r["task_id"], r["condition"]): r for r in rows}
    families = sorted({r["family_id"] for r in rows if r["split"] in ("iid_test", "composition_ood_test")},
                      key=lambda x: stable_seed("review-examples-v1", x))
    examples, selected_arrays = [], []
    used = set()
    for category_index, (category, tasks) in enumerate(CATEGORY_TASKS.items()):
        category_families = [f for f in families if f not in used][:2]
        used.update(category_families)
        for example_index, family_id in enumerate(category_families):
            task_id = tasks[example_index % len(tasks)]
            base = by[(family_id, task_id, "FULL")]
            relevant = by[(family_id, task_id, "RELEVANT_EDIT")]
            irrelevant = by[(family_id, task_id, "IRRELEVANT_EDIT")]
            control_task = base["control_question_id"].replace(f"{family_id}_", "", 1)
            cross = next(r for r in rows if r["family_id"] == family_id and r["task_id"] == control_task
                         and r["condition"] == "CROSS_CONTROL"
                         and r["contrast_group_id"] == base["contrast_group_id"])
            base_control = by[(family_id, control_task, "FULL")]
            bundle = []
            for label, row in (("base_target", base), ("relevant_target", relevant),
                               ("irrelevant_target", irrelevant), ("base_control", base_control),
                               ("transformed_control", cross)):
                values = row_values(row, arrays)
                bundle.append({
                    "cell": label, "item_id": row["item_id"], "series_id": row["series_id"],
                    "question": row["question_text"], "options": json.loads(row["options_json"]),
                    "semantic_gold": row["semantic_gold"], "gold_option": row["gold_option"],
                    "expected_relation": row["expected_relation"],
                    "visible_input_sha256": row["visible_input_sha256"],
                    "visible_series": serialize_series(values),
                })
                selected_arrays.append(values.astype(np.float32))
            examples.append({
                "dataset_version": base["dataset_version"], "protocol_version": base["protocol_version"],
                "selection_rule": "two families per ability category by stable hash; never model score",
                "category": category, "family_id": family_id, "split": base["split"],
                "target_task_id": task_id, "control_task_id": control_task,
                "target_program": base["target_program"], "cells": bundle,
            })
    docs_dir.mkdir(parents=True, exist_ok=True)
    jsonl = docs_dir / "review_examples.jsonl"
    jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in examples), encoding="utf-8")
    lines = ["# 真实可复算样例（固定哈希抽取）", "",
             "共 12 个 family，每类能力 2 个。选择仅由 family ID 稳定哈希决定，未使用模型成绩。", ""]
    for ex in examples:
        lines += [f"## {ex['category']} / {ex['family_id']} / {ex['target_task_id']}", "",
                  f"- split: `{ex['split']}`", f"- target program: `{ex['target_program']}`",
                  f"- control: `{ex['control_task_id']}`", ""]
        for cell in ex["cells"]:
            lines += [f"### {cell['cell']}", "", f"问题：{cell['question']}", "",
                      f"选项：`{json.dumps(cell['options'], ensure_ascii=False)}`", "",
                      f"语义 gold：`{cell['semantic_gold']}`；选项：`{cell['gold_option']}`", "",
                      "```text", cell["visible_series"], "```", ""]
    (docs_dir / "review_examples.md").write_text("\n".join(lines), encoding="utf-8")
    np.savez_compressed(docs_dir / "example_arrays.npz", values=np.stack(selected_arrays))
    return examples


def validation_markdown(result: dict, output: Path) -> None:
    lines = ["# TSQA Evidence v1 有效性检查", "",
             f"总状态：**{result['status']}**；通过 {result['summary']['passed']}，失败 {result['summary']['failed']}。", "",
             "| gate | 检查 | 状态 | 摘要 |", "|---:|---|---|---|"]
    for item in result["checks"]:
        detail = json.dumps(item["detail"], ensure_ascii=False, sort_keys=True)
        if len(detail) > 180: detail = detail[:177] + "..."
        lines.append(f"| {item['gate']} | `{item['name']}` | {item['status']} | {detail.replace('|', '/')} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
