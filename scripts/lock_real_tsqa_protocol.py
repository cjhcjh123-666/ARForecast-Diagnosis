#!/usr/bin/env python3
"""Seal the validated real TSQA v1 artifacts before target-model testing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/tsqa_evidence/real_v1"
LOCKED = [
    "configs/tsqa_evidence/real_v1_locked.yaml",
    "docs/tsqa_evidence/20_REAL_BENCHMARK_SPEC_V1_LOCKED.md",
    "results/tsqa_evidence/real_v1/source_inventory.csv",
    "results/tsqa_evidence/real_v1/dataset_task_eligibility.csv",
    "results/tsqa_evidence/real_v1/manifests/families_candidate.jsonl",
    "results/tsqa_evidence/real_v1/manifests/candidate_summary.json",
    "results/tsqa_evidence/real_v1/manifests/item_manifest.jsonl",
    "results/tsqa_evidence/real_v1/manifests/split_manifest.csv",
    "results/tsqa_evidence/real_v1/gold/gold_manifest.jsonl",
    "results/tsqa_evidence/real_v1/options/option_manifest.jsonl",
    "results/tsqa_evidence/real_v1/interventions/intervention_manifest.jsonl",
    "results/tsqa_evidence/real_v1/tables/answer_position_counts.csv",
    "results/tsqa_evidence/real_v1/validity/overall_validity.json",
    "results/tsqa_evidence/real_v1/validity/checkpoint_cache_audit.csv",
    "tsqa_evidence/real_metrics.py",
    "tsqa_evidence/real_protocol.py",
    "scripts/build_real_tsqa_candidates.py",
    "scripts/validate_real_tsqa.py",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    validity = json.loads((ROOT / "validity/overall_validity.json").read_text())
    if validity["status"] != "PASS" or validity.get("target_model_results_seen") is not False:
        raise SystemExit("refusing to lock: validity is not PASS or target results were seen")
    summary_path = ROOT / "manifests/candidate_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "PROTOCOL_LOCKED_BEFORE_TARGET_MODEL_TEST"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    entries = []
    aggregate = hashlib.sha256()
    for relative in LOCKED:
        path = REPO / relative
        if not path.is_file():
            raise SystemExit(f"missing locked artifact: {relative}")
        sha = digest(path)
        line = f"{sha}  {relative}"
        entries.append(line)
        aggregate.update((relative + "\t" + sha + "\n").encode())
    protocol_sha = aggregate.hexdigest()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    output = [
        f"protocol_sha256  {protocol_sha}",
        f"git_head  {head}",
        "hash_rule  sha256(relative_path<TAB>file_sha256<LF>) in the listed order",
        *entries,
    ]
    (ROOT / "manifest_sha256.txt").write_text("\n".join(output) + "\n")
    (ROOT / "protocol_lock.json").write_text(json.dumps({
        "benchmark_version": "real_tsqa_v1.0", "protocol_sha256": protocol_sha,
        "git_head": head, "validity": validity["status"], "target_model_results_seen": False,
        "locked_files": len(LOCKED),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"protocol_sha256": protocol_sha, "locked_files": len(LOCKED), "git_head": head}))


if __name__ == "__main__":
    main()
