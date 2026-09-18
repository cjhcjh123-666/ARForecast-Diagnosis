#!/usr/bin/env python3
"""Build the reviewable Real-TSQA v1 bundle without caches or source datasets."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "results/tsqa_evidence/real_v1"
STAGE = REPO / "deliverables/.real_tsqa_review_stage"
OUT = REPO / "deliverables/Real_TSQA_Evidence_v1_review.zip"


def copy_file(source: Path, relative: Path) -> None:
    target = STAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, relative: Path, *, suffixes: set[str] | None = None) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file() or (suffixes is not None and path.suffix not in suffixes):
            continue
        copy_file(path, relative / path.relative_to(source))


def command_text(args: list[str]) -> str:
    proc = subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    return proc.stdout


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    copy_file(REPO / "docs/tsqa_evidence/START_HERE_zh.md", Path("START_HERE_zh.md"))
    copy_tree(REPO / "docs/tsqa_evidence", Path("docs"), suffixes={".md", ".csv"})
    copy_tree(REPO / "configs/tsqa_evidence", Path("configs"), suffixes={".yaml", ".txt"})
    for directory in ("manifests", "gold", "options", "interventions", "validity", "review", "tables", "baselines"):
        copy_tree(RESULT / directory, Path(directory))
    copy_tree(RESULT / "diagnostic/predictions", Path("predictions/interface_B"), suffixes={".jsonl", ".json"})
    copy_tree(RESULT / "diagnostic/readout", Path("predictions/interface_A"), suffixes={".jsonl", ".json"})
    copy_tree(RESULT / "native", Path("predictions/interface_C"), suffixes={".jsonl", ".json"})
    for name in ("protocol_lock.json", "manifest_sha256.txt", "interface_a_addendum_sha256.txt",
                 "source_inventory.csv", "dataset_task_eligibility.csv", "job_status.csv",
                 "reproduction_commands.sh", "claim_evidence_matrix.csv"):
        path = RESULT / name
        if path.is_file():
            copy_file(path, Path(name))

    code_files = [
        "scripts/build_real_tsqa_candidates.py", "scripts/validate_real_tsqa.py",
        "scripts/extract_real_tsqa_reps.py", "scripts/run_real_tsqa_readout.py",
        "scripts/run_real_tsqa_supervised.py", "scripts/run_real_tsqa_native.py",
        "scripts/summarize_real_tsqa_predictions.py", "scripts/compare_real_tsqa_pr.py",
        "scripts/breakdown_real_tsqa_predictions.py", "scripts/make_real_tsqa_job_matrix.py",
        "scripts/build_real_tsqa_cross_model_tables.py",
        "scripts/transformers_runtime_compat.py", "scripts/openllm_suite.py",
        "scripts/audit_tsqa_checkpoints.py", "scripts/package_real_tsqa_review.py",
        "tsqa_evidence/real_protocol.py", "tsqa_evidence/real_metrics.py",
        "tsqa_evidence/candidate_scorer.py",
    ]
    for name in code_files:
        path = REPO / name
        if path.is_file():
            copy_file(path, Path("code") / name)
    copy_tree(REPO / "tests/tsqa_evidence", Path("code/tests/tsqa_evidence"), suffixes={".py"})
    if (REPO / "tests/test_deadline_p0.py").is_file():
        copy_file(REPO / "tests/test_deadline_p0.py", Path("code/tests/test_deadline_p0.py"))

    version = [
        "STATUS=LOCKED_MATRIX_COMPLETE",
        "HEAD=" + command_text(["git", "rev-parse", "HEAD"]).strip(),
        "BRANCH=" + command_text(["git", "branch", "--show-current"]).strip(),
        "",
        "GIT STATUS:",
        command_text(["git", "status", "--short", "--branch"]).rstrip(),
    ]
    (STAGE / "code_version.txt").write_text("\n".join(version) + "\n", encoding="utf-8")

    members = [p for p in sorted(STAGE.rglob("*")) if p.is_file()]
    digest_lines = []
    for path in members:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest_lines.append(f"{digest}  {path.relative_to(STAGE).as_posix()}")
    (STAGE / "SHA256SUMS").write_text("\n".join(digest_lines) + "\n", encoding="utf-8")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, Path("Real_TSQA_Evidence_v1_review") / path.relative_to(STAGE))
    with zipfile.ZipFile(OUT) as archive:
        names = archive.namelist()
        if any("diagnostic/reps" in name or name.endswith(".npz") for name in names):
            raise RuntimeError("representation cache leaked into review bundle")
        archive.testzip()
    print(f"wrote {OUT} files={len(names)} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
