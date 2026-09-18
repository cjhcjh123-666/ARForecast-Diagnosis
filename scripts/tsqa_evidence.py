#!/usr/bin/env python3
"""Unified TSQA Evidence v1 build/validate/lock/report CLI."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tsqa_evidence.benchmark import build_dataset, sha256_bytes
from tsqa_evidence.manifests import build_native_subset
from tsqa_evidence.reporting import build_review_examples, validation_markdown
from tsqa_evidence.validation import validate_dataset

ROOT = REPO / "results/tsqa_evidence/v1"
DOCS = REPO / "docs/tsqa_evidence"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_build(_: argparse.Namespace) -> None:
    print(json.dumps(build_dataset(ROOT), indent=2))


def command_audit(_: argparse.Namespace) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    legacy_l2 = sorted((REPO / "results/temporal_abstraction/relations").glob("level2_*.csv"))
    corrected_l2 = sorted((REPO / "results/temporal_abstraction/corrected_v2/relations").glob("level2_*.csv"))
    level1 = sorted((REPO / "results/temporal_abstraction/reps").glob("*.npz"))
    qa_reps = sorted((REPO / "results/temporal_abstraction/corrected_v2/qa/reps").glob("*.npz"))
    qa_factorized = sorted((REPO / "results/temporal_abstraction/corrected_v2/qa").glob("factorized_*.csv"))
    qa_native = sorted((REPO / "results/temporal_abstraction/corrected_v2/qa/native").glob("*.csv"))
    entries = [
        {"asset": "level1_frozen_representations", "status": "COMPLETE", "validity": "VALID",
         "coverage": f"{len(level1)} npz", "path": "results/temporal_abstraction/reps",
         "reason": "10 models x P/R x 3 seeds", "audited_utc": now},
        {"asset": "legacy_level2", "status": "COMPLETE", "validity": "INVALID",
         "coverage": f"{len(legacy_l2)} csv", "path": "results/temporal_abstraction/relations",
         "reason": "superseded measurement protocol; retained", "audited_utc": now},
        {"asset": "corrected_v2_level2", "status": "RUNNING" if len(corrected_l2) < 60 else "COMPLETE",
         "validity": "PARTIAL" if len(corrected_l2) < 60 else "VALID", "coverage": f"{len(corrected_l2)}/60 runs",
         "path": "results/temporal_abstraction/corrected_v2/relations",
         "reason": "existing closure queue preserved", "audited_utc": now},
        {"asset": "corrected_irts_representations", "status": "COMPLETE_LIMITED", "validity": "VALID_LIMITED",
         "coverage": f"{len(qa_reps)} P/R caches", "path": "results/temporal_abstraction/corrected_v2/qa/reps",
         "reason": "Qwen and DeepSeek, seed 7", "audited_utc": now},
        {"asset": "corrected_irts_factorized", "status": "COMPLETE_LIMITED", "validity": "VALID_LIMITED",
         "coverage": f"{len(qa_factorized)} condition tables", "path": "results/temporal_abstraction/corrected_v2/qa",
         "reason": "trained QA diagnostic, not native", "audited_utc": now},
        {"asset": "corrected_irts_native", "status": "COMPLETE_LIMITED", "validity": "VALID_LIMITED",
         "coverage": f"{len(qa_native)} per-item csv", "path": "results/temporal_abstraction/corrected_v2/qa/native",
         "reason": "official Qwen seed 7; unequal parse coverage", "audited_utc": now},
        {"asset": "tsqa_evidence_v1", "status": "BUILT" if (ROOT / "item_manifest.csv").is_file() else "NOT_RUN",
         "validity": "PENDING_GATE" if not (ROOT / "validation/test_results.json").is_file() else
                     json.loads((ROOT / "validation/test_results.json").read_text())["status"],
         "coverage": "900 families / 43200 item rows" if (ROOT / "item_manifest.csv").is_file() else "0",
         "path": "results/tsqa_evidence/v1", "reason": "independent v1 directory", "audited_utc": now},
    ]
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    print(json.dumps(entries, indent=2))


def command_validate(_: argparse.Namespace) -> None:
    result = validate_dataset(ROOT)
    validation_markdown(result, DOCS / "02_validity_report.md")
    print(json.dumps(result["summary"] | {"status": result["status"]}, indent=2))
    if result["status"] != "PASS": raise SystemExit(2)


def command_report(_: argparse.Namespace) -> None:
    examples = build_review_examples(ROOT, ROOT / "review")
    print(f"wrote {len(examples)} review families")


def command_lock(_: argparse.Namespace) -> None:
    validation = json.loads((ROOT / "validation/test_results.json").read_text())
    if validation["status"] != "PASS":
        raise SystemExit("refusing to lock: data validation is not PASS")
    native = build_native_subset(ROOT / "item_manifest.csv", ROOT / "native_subset_manifest.csv")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=REPO)
    paths = [ROOT / "item_manifest.csv", ROOT / "split_manifest.csv", ROOT / "series_arrays.npz",
             ROOT / "native_subset_manifest.csv", REPO / "configs/tsqa_evidence/v1.yaml"]
    lock = {
        "status": "LOCKED_FOR_TARGET_MODEL_EVALUATION",
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head, "uncommitted_tracked_patch_sha256": sha256_bytes(diff),
        "target_model_test_scores_seen_before_lock": False,
        "manifest_sha256": {str(p.relative_to(REPO)): file_hash(p) for p in paths},
        "native_subset_rows": len(native), "bootstrap_unit": "family_id",
        "head_seeds": [7, 17, 27],
        "amendment_rule": "Any validity-changing edit increments protocol version and preserves invalid outputs.",
    }
    (ROOT / "protocol_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("audit", command_audit), ("build", command_build), ("validate", command_validate),
                     ("report", command_report), ("lock", command_lock)):
        p = sub.add_parser(name); p.set_defaults(func=fn)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
