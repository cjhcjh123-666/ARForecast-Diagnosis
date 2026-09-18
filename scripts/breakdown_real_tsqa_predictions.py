#!/usr/bin/env python3
"""Write domain/task breakdowns from locked per-item Real-TSQA predictions."""

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

from scripts.summarize_real_tsqa_predictions import attach_manifest_fields, read_jsonl, run_key  # noqa: E402


def group_macro(rows: list[dict], field: str) -> float:
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(float(bool(row[field])))
    return float(np.mean([np.mean(values) for values in by_group.values()]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = [row for path in args.inputs for row in read_jsonl(path)]
    attach_manifest_fields(predictions)
    runs: dict[tuple, list[dict]] = defaultdict(list)
    for row in predictions:
        runs[run_key(row)].append(row)
    output = []
    for key, rows in sorted(runs.items()):
        model, interface, branch, seed, condition = key
        slices: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            slices[(row["track"], row["domain"], row["task_type"])].append(row)
        for (track, domain, task), selected in sorted(slices.items()):
            output.append({
                "model": model, "interface": interface, "pretrained_or_random": branch,
                "seed": seed, "condition": condition, "track": track,
                "domain": domain, "task_type": task,
                "group_macro_accuracy": group_macro(selected, "correct"),
                "raw_accuracy": float(np.mean([bool(row["correct"]) for row in selected])),
                "vor": float(np.mean([bool(row["valid_output"]) for row in selected])),
                "n_groups": len({row["group_id"] for row in selected}),
                "n_families": len({row["family_id"] for row in selected}),
                "n_items": len(selected),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    print(json.dumps({"runs": len(runs), "rows": len(output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
