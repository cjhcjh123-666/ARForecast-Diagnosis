#!/usr/bin/env python3
"""Create paired pretrained-vs-random tables with shared cluster bootstrap draws."""

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

from scripts.summarize_real_tsqa_predictions import (  # noqa: E402
    attach_manifest_fields,
    family_records,
    flatten,
    read_jsonl,
    run_key,
    summarize_run,
)


BRANCHES = ("pretrained_temporal", "random_temporal")
BEHAVIOR_VALUES = {
    "fcjs": "four_joint",
    "tus": "target_joint",
    "cis": "control_joint",
    "qswitch_original": "qswitch0",
    "qswitch_transformed": "qswitch1",
}


def average_seed_records(records: list[dict], identity: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        grouped[tuple(str(row[k]) for k in identity)].append(row)
    out = []
    for _, rows in grouped.items():
        first = rows[0]
        out.append({**{k: first[k] for k in first if k != "value"},
                    "value": float(np.mean([float(row["value"]) for row in rows]))})
    return out


def hierarchy(records: list[dict], *, use_task: bool) -> float:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in records:
        key = (row["domain"], row["group_id"], row["task_type"] if use_task else "all")
        grouped[key].append(float(row["value"]))
    group_scores = {key: float(np.mean(values)) for key, values in grouped.items()}
    task_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (domain, _group, task), value in group_scores.items():
        task_scores[(domain, task)].append(value)
    domain_scores: dict[str, list[float]] = defaultdict(list)
    for (domain, _task), values in task_scores.items():
        domain_scores[domain].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in domain_scores.values()]))


def paired_bootstrap(p_rows: list[dict], r_rows: list[dict], *, use_task: bool,
                     iterations: int, seed: int) -> dict:
    def group_map(rows: list[dict]) -> dict[tuple[str, str, str], float]:
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in rows:
            key = (row["domain"], row["group_id"], row["task_type"] if use_task else "all")
            grouped[key].append(float(row["value"]))
        return {key: float(np.mean(values)) for key, values in grouped.items()}

    pmap, rmap = group_map(p_rows), group_map(r_rows)
    if pmap.keys() != rmap.keys():
        raise ValueError("P/R aggregation units differ")
    domains = sorted({key[0] for key in pmap})
    groups = {domain: sorted({key[1] for key in pmap if key[0] == domain}) for domain in domains}
    tasks = {domain: sorted({key[2] for key in pmap if key[0] == domain}) for domain in domains}

    def estimate(samples: dict[str, list[str]] | None = None) -> tuple[float, float]:
        branch_scores = []
        for mapping in (pmap, rmap):
            per_domain = []
            for domain in domains:
                chosen = groups[domain] if samples is None else samples[domain]
                per_task = []
                for task in tasks[domain]:
                    values = [mapping[(domain, group, task)] for group in chosen
                              if (domain, group, task) in mapping]
                    per_task.append(float(np.mean(values)))
                per_domain.append(float(np.mean(per_task)))
            branch_scores.append(float(np.mean(per_domain)))
        return branch_scores[0], branch_scores[1]

    pretrained, random = estimate()
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        samples = {domain: list(rng.choice(groups[domain], size=len(groups[domain]), replace=True))
                   for domain in domains}
        p_value, r_value = estimate(samples)
        deltas.append(p_value - r_value)
    return {
        "pretrained": pretrained,
        "random": random,
        "delta_pretrained_minus_random": pretrained - random,
        "delta_ci95_low": float(np.quantile(deltas, 0.025)),
        "delta_ci95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "n_independent_groups": len({(row["domain"], row["group_id"]) for row in p_rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tsqa_evidence/real_v1/tables"))
    parser.add_argument("--stem", required=True)
    parser.add_argument("--interface", default="question_conditioned_supervised_QA")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in read_jsonl(path)]
    attach_manifest_fields(rows)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[run_key(row)].append(row)
    summaries = {key: summarize_run(value) for key, value in grouped.items()}
    full = {key: value for key, value in summaries.items() if key[-1] == "full"}
    seeds = sorted({key[3] for key in full})
    models = {key[0] for key in full}
    if len(models) != 1:
        raise ValueError("compare one model at a time")
    model = next(iter(models))
    if any((model, args.interface, branch, seed, "full") not in full
           for seed in seeds for branch in BRANCHES):
        raise ValueError("missing paired P/R run")

    per_seed = []
    selected_metrics = (
        "fcjs_domain_macro", "tus_domain_macro", "cis_domain_macro",
        "qswitch_original_domain_macro", "qswitch_transformed_domain_macro",
        "R_accuracy_domain_macro", "I_accuracy_domain_macro", "ALL_accuracy_domain_macro",
    )
    for seed in seeds:
        flat = {branch: flatten(full[(model, args.interface, branch, seed, "full")])
                for branch in BRANCHES}
        for metric in selected_metrics:
            p, r = float(flat[BRANCHES[0]][metric]), float(flat[BRANCHES[1]][metric])
            per_seed.append({"model": model, "seed": seed, "metric": metric,
                             "pretrained": p, "random": r,
                             "delta_pretrained_minus_random": p - r})

    by_run_rows = {key: value for key, value in grouped.items() if key[-1] == "full"}
    bootstrap_rows = []
    for metric, value_key in BEHAVIOR_VALUES.items():
        branch_records = {}
        for branch in BRANCHES:
            records = []
            for seed in seeds:
                key = (model, args.interface, branch, seed, "full")
                for family in family_records(by_run_rows[key]):
                    records.append({**family, "seed": seed, "task_type": "four_cell", "value": family[value_key]})
            branch_records[branch] = average_seed_records(records, ("family_id",))
        result = paired_bootstrap(branch_records[BRANCHES[0]], branch_records[BRANCHES[1]],
                                  use_task=False, iterations=args.iterations, seed=args.seed)
        bootstrap_rows.append({"model": model, "scope": "behavior", "metric": metric, **result})

    for track in ("R", "I", "ALL"):
        branch_records = {}
        for branch in BRANCHES:
            records = []
            for seed in seeds:
                key = (model, args.interface, branch, seed, "full")
                selected = by_run_rows[key] if track == "ALL" else [r for r in by_run_rows[key] if r["track"] == track]
                records.extend({**row, "seed": seed, "value": float(bool(row["correct"]))} for row in selected)
            branch_records[branch] = average_seed_records(records, ("family_id", "cell"))
        result = paired_bootstrap(branch_records[BRANCHES[0]], branch_records[BRANCHES[1]],
                                  use_task=True, iterations=args.iterations, seed=args.seed)
        bootstrap_rows.append({"model": model, "scope": f"items_{track}",
                               "metric": "accuracy_domain_macro", **result})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_dir / f"{args.stem}_per_seed.csv"
    boot_path = args.output_dir / f"{args.stem}_paired_bootstrap.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0])); writer.writeheader(); writer.writerows(per_seed)
    with boot_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0])); writer.writeheader(); writer.writerows(bootstrap_rows)
    print(json.dumps({"model": model, "seeds": seeds, "per_seed": str(seed_path),
                      "paired_bootstrap": str(boot_path)}))


if __name__ == "__main__":
    main()
