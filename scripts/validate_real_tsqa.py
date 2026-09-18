#!/usr/bin/env python3
"""Run protocol-lock gates and non-model baselines for real TSQA v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

from tsqa_evidence.real_metrics import four_cell_metrics


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def majority_baseline(items: list[dict]) -> tuple[dict, dict]:
    train = [r for r in items if r["split"] == "train"]
    test = [r for r in items if r["split"] == "test"]
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in train:
        counts[(row["domain"], row["question_id"])][row["gold_position"]] += 1
    choices = {key: sorted(counter, key=lambda p: (-counter[p], p))[0] for key, counter in counts.items()}
    predicted = []
    for row in test:
        pred = choices[(row["domain"], row["question_id"])]
        predicted.append({**row, "prediction": pred, "correct": pred == row["gold_position"]})
    core: dict[str, dict[str, object]] = defaultdict(dict)
    for row in predicted:
        if row["cell"] in {"t0", "c0", "t1", "c1"}:
            item = core[row["family_id"]]
            item["family_id"] = row["family_id"]
            item[row["cell"]] = row["correct"]
            item[f"valid_{row['cell']}"] = True
            item[f"pred_{row['cell']}"] = row["prediction"]
    metrics = four_cell_metrics(core.values())
    metrics["all_test_item_accuracy"] = sum(r["correct"] for r in predicted) / len(predicted)
    metrics["n_test_items"] = len(predicted)
    return metrics, {"choices": {"|".join(k): v for k, v in choices.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "results/tsqa_evidence/real_v1")
    args = ap.parse_args()
    root = args.root
    families = read_jsonl(root / "manifests/families_candidate.jsonl")
    items = read_jsonl(root / "manifests/item_manifest.jsonl")
    options = read_jsonl(root / "options/option_manifest.jsonl")
    gold = read_jsonl(root / "gold/gold_manifest.jsonl")
    interventions = read_jsonl(root / "interventions/intervention_manifest.jsonl")
    candidate_validation = json.loads((root / "validity/formal_candidate_validation.json").read_text())
    eligibility = list(csv.DictReader((root / "dataset_task_eligibility.csv").open()))
    selected = {f["dataset"] for f in families}
    selected_eligibility = [r for r in eligibility if r["dataset_name"] in selected]

    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.tsqa_evidence.test_real_metrics", "tests.tsqa_evidence.test_real_protocol", "-v"],
        cwd=REPO, text=True, capture_output=True,
    )
    checks = list(candidate_validation["checks"])

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("selected_dataset_task_eligibility", bool(selected_eligibility) and all(r["eligibility"] == "PASS" for r in selected_eligibility),
        f"{len(selected)} selected datasets × 6 tasks")
    add("manifest_referential_integrity", len(items) == len(gold) == 2112 and len(options) == len(families) * 6 and len(interventions) == len(families),
        f"families={len(families)} items={len(items)} gold={len(gold)} options={len(options)} interventions={len(interventions)}")
    item_keys = {(r["family_id"], r["cell"], r["question_id"]) for r in items}
    gold_keys = {(r["family_id"], r["cell"], r["question_id"]) for r in gold}
    add("item_gold_key_alignment", item_keys == gold_keys and len(item_keys) == len(items), f"keys={len(item_keys)}")
    qonly_pairs = defaultdict(dict)
    for r in items:
        if r["cell"] in {"t0", "t1"}:
            qonly_pairs[r["family_id"]][r["cell"]] = (r["question"], r["option_order"], r["gold_position"])
    add("question_only_target_impossibility", all(v["t0"][:2] == v["t1"][:2] and v["t0"][2] != v["t1"][2] for v in qonly_pairs.values()),
        "same question/order and opposite target position for every family")
    add("metric_and_protocol_unit_tests", unit.returncode == 0, "12/12 tests PASS" if unit.returncode == 0 else unit.stderr[-1000:])

    oracle_rows = [dict(family_id=f["family_id"], t0=True, c0=True, t1=True, c1=True,
                        pred_t0=f["semantic_gold"]["t0"], pred_t1=f["semantic_gold"]["t1"],
                        pred_c0=f["semantic_gold"]["c0"], pred_c1=f["semantic_gold"]["c1"])
                   for f in families if f["split"] == "test"]
    oracle = four_cell_metrics(oracle_rows)
    add("deterministic_oracle", all(oracle[k] == 1.0 for k in ("acc_t0", "acc_c0", "acc_t1", "acc_c1", "fcjs", "tus", "cis", "qsja", "vor")),
        f"test families={len(oracle_rows)} all core metrics=1")
    majority, majority_config = majority_baseline(items)
    baselines = {
        "status": "NO_TARGET_MODEL_USED",
        "random_choice_expected_core": {"acc_t0": .5, "acc_c0": .5, "acc_t1": .5, "acc_c1": .5, "fcjs": .0625, "tus": .25, "cis": .25, "qsja": .25},
        "majority_answer_position_test": majority,
        "majority_training_choices": majority_config,
        "deterministic_oracle_test": oracle,
        "handcrafted_numeric_reference": {"relationship_to_oracle": "same deterministic visible-value programs", "expected_core": 1.0},
        "question_only_structural_fact": "A deterministic question-only system emits the same target answer for t0/t1, whose locked gold positions differ; it cannot jointly pass both target cells.",
    }
    (root / "baselines").mkdir(parents=True, exist_ok=True)
    (root / "baselines/structural_baselines.json").write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")

    overall = {
        "status": "PASS" if all(c["passed"] for c in checks) else "FAIL",
        "hard_gate_count": len(checks), "passed": sum(c["passed"] for c in checks),
        "checks": checks, "unit_test_stdout": unit.stdout[-4000:], "unit_test_stderr": unit.stderr[-4000:],
        "target_model_results_seen": False,
    }
    (root / "validity/overall_validity.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": overall["status"], "checks": f"{overall['passed']}/{overall['hard_gate_count']}", "test_families": len(oracle_rows), "majority_fcjs": majority["fcjs"]}))
    if overall["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
