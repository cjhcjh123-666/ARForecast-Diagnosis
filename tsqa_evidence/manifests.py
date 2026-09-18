"""Stable split and native-subset manifests."""
from __future__ import annotations

import csv
from pathlib import Path

from .benchmark import TASK_SPECS, stable_seed


def build_native_subset(item_manifest: Path, output: Path) -> list[dict]:
    with Path(item_manifest).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full = [r for r in rows if r["condition"] == "FULL"]
    by_family = {}
    for row in full:
        by_family.setdefault(row["family_id"], []).append(row)
    selected_families = []
    for split in ("iid_test", "composition_ood_test"):
        families = sorted((f for f, rr in by_family.items() if rr[0]["split"] == split),
                          key=lambda f: stable_seed("native-v1", split, f))[:80]
        selected_families.extend(families)
    groups = (
        ("A1_interval_trend", "A2_dominant_interval"),
        ("B1_interval_mean", "B2_interval_mean_difference", "C1_local_deviation_interval", "C2_change_point_interval"),
        ("D1_repeat_interval_comparison", "D2_channel_lead_lag", "E1_repeat_and_event", "E2_before_after_structure"),
        ("F1_conditional_extremum", "F2_event_order"),
    )
    output_rows = []
    for family_id in selected_families:
        lookup = {r["task_id"]: r for r in by_family[family_id]}
        for group_index, group in enumerate(groups):
            task_id = group[stable_seed("native-v1", family_id, group_index) % len(group)]
            row = lookup[task_id]
            output_rows.append({
                "dataset_version": row["dataset_version"], "family_id": family_id,
                "split": row["split"], "task_id": task_id, "category": TASK_SPECS[task_id].category,
                "base_item_id": row["item_id"], "selection_rule": "stable_hash_80_per_split_4_strata",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    return output_rows
