"""Machine-readable validity checks for TSQA Evidence v1."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .benchmark import (DATASET_VERSION, TASK_IDS, TASK_SPECS, gold, load_dataset,
                        parse_series, row_values, serialize_series, sha256_bytes, visible_hash)
from .rendering import render_question_only, render_question_option, render_temporal


def validate_dataset(root: Path) -> dict:
    root = Path(root)
    rows, arrays = load_dataset(root)
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: object, gate: int | str) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL",
                       "detail": detail, "gate": gate})

    family_splits: dict[str, set[str]] = defaultdict(set)
    item_ids, family_base_tasks = [], defaultdict(set)
    for row in rows:
        family_splits[row["family_id"]].add(row["split"])
        item_ids.append(row["item_id"])
        if row["condition"] == "FULL": family_base_tasks[row["family_id"]].add(row["task_id"])
    check("unique_item_ids", len(item_ids) == len(set(item_ids)),
          {"n": len(item_ids), "unique": len(set(item_ids))}, 6)
    check("family_level_split_disjoint", all(len(x) == 1 for x in family_splits.values()),
          {"families": len(family_splits), "violations": sum(len(x) != 1 for x in family_splits.values())}, 8)
    check("twelve_base_questions_per_family",
          all(tasks == set(TASK_IDS) for tasks in family_base_tasks.values()),
          {"families": len(family_base_tasks)}, 8)

    by_key = {(r["family_id"], r["task_id"], r["condition"]): r for r in rows
              if r["condition"] in ("FULL", "RELEVANT_EDIT", "IRRELEVANT_EDIT")}
    roundtrip_failures, gold_failures, relation_failures, isolation_failures = [], [], [], []
    for row in rows:
        values = row_values(row, arrays)
        visible = serialize_series(values)
        reparsed = parse_series(visible)
        if serialize_series(reparsed) != visible: roundtrip_failures.append(row["item_id"])
        if visible_hash(values) != row["visible_input_sha256"]: roundtrip_failures.append(row["item_id"] + ":hash")
        if gold(row["task_id"], reparsed) != row["semantic_gold"]: gold_failures.append(row["item_id"])
        temporal = render_temporal(values)
        question = render_question_option(row, "A")
        if row["question_text"] in temporal or row["semantic_gold"] in temporal or "Candidate A" in temporal:
            isolation_failures.append(row["item_id"] + ":temporal")
        # Question rendering may contain a correct candidate by design, but never observed CSV rows.
        first_observation_row = visible.splitlines()[2]
        if "t,sensor_A,sensor_B,sensor_C,sensor_D" in question or first_observation_row in question:
            isolation_failures.append(row["item_id"] + ":question")
    check("serialization_roundtrip_and_hash", not roundtrip_failures,
          {"failures": roundtrip_failures[:20], "n_checked": len(rows)}, 10)
    check("gold_recomputed_from_visible_input", not gold_failures,
          {"failures": gold_failures[:20], "n_checked": len(rows)}, 10)
    check("temporal_question_branch_isolation", not isolation_failures,
          {"failures": isolation_failures[:20], "n_checked": len(rows)}, 4)

    for family_id in family_splits:
        for task_id in TASK_IDS:
            base = by_key[(family_id, task_id, "FULL")]
            rel = by_key[(family_id, task_id, "RELEVANT_EDIT")]
            irr = by_key[(family_id, task_id, "IRRELEVANT_EDIT")]
            same_prompt = base["question_text"] == rel["question_text"] == irr["question_text"]
            same_options = base["options_json"] == rel["options_json"] == irr["options_json"]
            if not (same_prompt and same_options and base["semantic_gold"] != rel["semantic_gold"]
                    and base["semantic_gold"] == irr["semantic_gold"]):
                relation_failures.append(f"{family_id}:{task_id}")
    check("relevant_change_irrelevant_invariant", not relation_failures,
          {"failures": relation_failures[:20], "pairs": len(family_splits) * len(TASK_IDS)}, 11)

    split_counts = Counter(next(iter(x)) for x in family_splits.values())
    check("protocol_split_counts", split_counts == Counter({
        "train": 400, "validation": 100, "iid_test": 200, "composition_ood_test": 200}),
        dict(split_counts), 8)

    position = Counter(r["gold_option"] for r in rows if r["condition"] == "FULL")
    max_dev = max(abs(position[x] - sum(position.values()) / 4) for x in "ABCD")
    check("answer_position_distribution", max_dev / (sum(position.values()) / 4) < 0.06,
          dict(position), 12)

    distributions = []
    for split in ("train", "validation", "iid_test", "composition_ood_test"):
        for task_id in TASK_IDS:
            subset = [r for r in rows if r["split"] == split and r["task_id"] == task_id
                      and r["condition"] == "FULL"]
            support = Counter(r["semantic_gold"] for r in subset)
            distributions.append({"split": split, "task_id": task_id, "n": len(subset),
                                  "n_classes": len(support), "support_json": json.dumps(support, sort_keys=True)})
    constant = [x for x in distributions if x["n_classes"] < 2]
    check("nonconstant_reported_targets", not constant,
          {"constant": constant, "cells": len(distributions)}, 13)

    qo_leaks = []
    for row in rows[:2000]:
        q = render_question_only(row)
        values = row_values(row, arrays)
        if "\nt," in q or any(f"{values[0, j]:+.3f}" in q for j in range(4)):
            qo_leaks.append(row["item_id"])
    check("question_only_has_no_observed_values", not qo_leaks,
          {"failures": qo_leaks[:20], "n_checked": min(2000, len(rows))}, 5)

    required = {"values", "family_ids", "variant_ids", "timestamps", "channels"}
    check("archive_schema", set(arrays) == required,
          {"keys": sorted(arrays), "shape": list(arrays["values"].shape)}, 20)

    result = {
        "dataset_version": DATASET_VERSION,
        "status": "PASS" if all(x["status"] == "PASS" for x in checks) else "FAIL",
        "checks": checks,
        "summary": {"passed": sum(x["status"] == "PASS" for x in checks),
                    "failed": sum(x["status"] == "FAIL" for x in checks),
                    "families": len(family_splits), "items": len(rows)},
    }
    out = root / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with (out / "distribution_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distributions[0])); writer.writeheader(); writer.writerows(distributions)
    return result
