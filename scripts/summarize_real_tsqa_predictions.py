#!/usr/bin/env python3
"""Recompute locked Real-TSQA metrics from per-item prediction JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

from tsqa_evidence.real_metrics import CELLS, assert_metric_consistency, four_cell_metrics, hierarchical_macro


ROOT = Path("results/tsqa_evidence/real_v1")
PROTOCOL_SHA = "9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def attach_manifest_fields(rows: list[dict]) -> None:
    """Backfill aggregation-only fields for early runner output without changing it."""
    manifest = read_jsonl(ROOT / "manifests/item_manifest.jsonl")
    lookup = {
        (item["family_id"], item["variant_id"], item["question_id"]): item
        for item in manifest
    }
    for row in rows:
        key = (row["family_id"], row["variant_id"], row["question_id"])
        if key not in lookup:
            raise ValueError(f"prediction does not match locked item manifest: {key}")
        item = lookup[key]
        for field in ("split", "track", "cell"):
            present = row.get(field)
            if present is not None and present != item[field]:
                raise ValueError(f"prediction/manifest {field} mismatch")
            row[field] = item[field]
        for field in ("gold_semantic", "gold_position", "option_order", "semantic_options"):
            if row.get(field) != item[field]:
                raise ValueError(f"prediction/manifest {field} mismatch for {key}")


def run_key(row: dict) -> tuple:
    return (
        row["model"], row["interface"], row["pretrained_or_random"],
        int(row["seed"]), row["condition"],
    )


def group_domain_macro(families: list[dict], value_key: str) -> dict:
    group_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in families:
        group_values[(row["domain"], row["group_id"])].append(float(row[value_key]))
    domain_values: dict[str, list[float]] = defaultdict(list)
    for (domain, _group), values in group_values.items():
        domain_values[domain].append(float(np.mean(values)))
    scores = {domain: float(np.mean(values)) for domain, values in sorted(domain_values.items())}
    return {
        "headline_domain_macro": float(np.mean(list(scores.values()))),
        "domain_scores": scores,
        "raw_family_pooled": float(np.mean([float(row[value_key]) for row in families])),
        "n_groups": len(group_values),
        "n_families": len(families),
    }


def family_records(rows: list[dict]) -> list[dict]:
    by_family: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["cell"] in CELLS:
            if row["cell"] in by_family[row["family_id"]]:
                raise ValueError(f"duplicate cell {row['family_id']} {row['cell']}")
            by_family[row["family_id"]][row["cell"]] = row
    output = []
    for family_id, cells in sorted(by_family.items()):
        if set(cells) != set(CELLS):
            raise ValueError(f"incomplete four-cell family {family_id}: {sorted(cells)}")
        first = cells["t0"]
        record = {
            "family_id": family_id,
            "group_id": first["group_id"],
            "domain": first["domain"],
        }
        for cell in CELLS:
            record[cell] = bool(cells[cell]["correct"])
            record[f"valid_{cell}"] = bool(cells[cell]["valid_output"])
            record[f"pred_{cell}"] = cells[cell].get("parsed_semantic", "")
            record[f"gold_{cell}"] = cells[cell]["gold_semantic"]
        record["target_joint"] = float(record["t0"] and record["t1"])
        record["control_joint"] = float(record["c0"] and record["c1"])
        record["four_joint"] = float(record["target_joint"] and record["control_joint"])
        record["qswitch0"] = float(record["t0"] and record["c0"])
        record["qswitch1"] = float(record["t1"] and record["c1"])
        output.append(record)
    return output


def summarize_run(rows: list[dict]) -> dict:
    if any(row.get("protocol_sha256") != PROTOCOL_SHA for row in rows):
        raise ValueError("prediction protocol SHA mismatch")
    manifest = [row for row in read_jsonl(ROOT / "manifests/item_manifest.jsonl") if row["split"] == "test"]
    expected = {(row["family_id"], row["variant_id"], row["question_id"]) for row in manifest}
    observed = [(row["family_id"], row["variant_id"], row["question_id"]) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate predictions within a run/condition")
    if set(observed) != expected:
        missing, extra = expected - set(observed), set(observed) - expected
        raise ValueError(f"prediction coverage mismatch: missing={len(missing)} extra={len(extra)}")
    families = family_records(rows)
    behavior = four_cell_metrics(families)
    assert_metric_consistency(behavior)

    qd_values = []
    for family in families:
        for suffix in ("0", "1"):
            target, control = f"t{suffix}", f"c{suffix}"
            if family[f"gold_{target}"] != family[f"gold_{control}"]:
                qd_values.append(float(family[target] and family[control]))
    qd = {
        "qsja_question_discriminative": float(np.mean(qd_values)) if qd_values else None,
        "question_discriminative_pair_n": len(qd_values),
    }

    item_sections = {}
    for track in ("R", "I", "ALL"):
        selected = rows if track == "ALL" else [row for row in rows if row["track"] == track]
        item_sections[track] = {
            "accuracy": float(np.mean([bool(row["correct"]) for row in selected])),
            "vor": float(np.mean([bool(row["valid_output"]) for row in selected])),
            "n_items": len(selected),
            "hierarchical_accuracy": hierarchical_macro([
                {**row, "value": float(bool(row["correct"]))} for row in selected
            ]),
        }
    return {
        "run": dict(zip(("model", "interface", "pretrained_or_random", "seed", "condition"), run_key(rows[0]))),
        "behavior": behavior,
        "question_discriminative": qd,
        "behavior_macro": {
            "fcjs": group_domain_macro(families, "four_joint"),
            "tus": group_domain_macro(families, "target_joint"),
            "cis": group_domain_macro(families, "control_joint"),
            "qswitch_original": group_domain_macro(families, "qswitch0"),
            "qswitch_transformed": group_domain_macro(families, "qswitch1"),
        },
        "items": item_sections,
    }


def flatten(summary: dict) -> dict:
    row = dict(summary["run"])
    behavior = summary["behavior"]
    for key in ("n_families", "acc_t0", "acc_c0", "acc_t1", "acc_c1", "fcjs", "tus", "cis", "qsja", "vor"):
        row[key] = behavior[key]
    for name in ("tus_given_t0_correct", "cis_given_c0_correct", "target_prediction_change_rate", "control_spurious_change_rate"):
        row[name] = behavior[name]["value"]
        row[f"{name}_n"] = behavior[name]["denominator"]
    row.update(summary["question_discriminative"])
    for metric, values in summary["behavior_macro"].items():
        row[f"{metric}_domain_macro"] = values["headline_domain_macro"]
    for track in ("R", "I", "ALL"):
        row[f"{track}_item_accuracy"] = summary["items"][track]["accuracy"]
        row[f"{track}_item_vor"] = summary["items"][track]["vor"]
        row[f"{track}_item_n"] = summary["items"][track]["n_items"]
        row[f"{track}_accuracy_domain_macro"] = summary["items"][track]["hierarchical_accuracy"]["headline_domain_macro"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables")
    parser.add_argument("--stem", default="formal_metrics")
    args = parser.parse_args()
    all_rows = []
    for path in args.inputs:
        all_rows.extend(read_jsonl(path))
    attach_manifest_fields(all_rows)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[run_key(row)].append(row)
    summaries = [summarize_run(rows) for _, rows in sorted(grouped.items())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.stem}.json"
    csv_path = args.output_dir / f"{args.stem}.csv"
    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    flat = [flatten(summary) for summary in summaries]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"runs": len(summaries), "json": str(json_path), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
