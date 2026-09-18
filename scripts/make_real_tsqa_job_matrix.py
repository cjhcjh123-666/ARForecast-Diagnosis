#!/usr/bin/env python3
"""Write the locked B/C command matrix without launching GPU jobs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("results/tsqa_evidence/real_v1")
MODELS_B = ["qwen3_8b_base", "llama31_8b", "llama32_3b", "gemma2_9b", "gemma2_2b",
            "mistral_7b_v03", "deepseek_llm_7b", "deepseek_v2_lite", "olmo2_7b", "olmo2_13b"]
MODELS_C = ["qwen3_8b_base", "llama31_8b", "gemma2_9b", "deepseek_v2_lite", "olmo2_7b"]
SEEDS = (7, 17, 27)


def expected_output(row: dict) -> Path:
    model, seed = row["model"], row["seed"]
    if row["task"] == "B_question_cache":
        return ROOT / f"diagnostic/reps/{model}_question_pretrained.npz"
    if row["task"] == "B_temporal_cache":
        suffix = "pretrained_s7" if row["init"] == "pretrained" else f"random_s{seed}"
        return ROOT / f"diagnostic/reps/{model}_temporal_{suffix}.npz"
    if row["task"] == "A_controlled_readout":
        return ROOT / f"diagnostic/readout/{model}_s{seed}.jsonl"
    if row["task"] == "B_supervised_head":
        return ROOT / f"diagnostic/predictions/{model}_s{seed}.jsonl"
    if row["task"].startswith("C_native_"):
        mode = row["task"].removeprefix("C_native_")
        return ROOT / f"native/{model}_{mode}.jsonl"
    raise ValueError(row["task"])


def main() -> None:
    rows = []
    common = {"status": "NOT_STARTED_RESOURCE_GUARD", "validity": "LOCKED_PROTOCOL_PASS",
              "eta": "PENDING_FIRST_MEASURED_RUN", "next_step": "launch only after host GPU/process visibility is available"}
    for model in MODELS_B:
        rows.append({"task": "B_question_cache", "model": model, "init": "pretrained", "seed": 7,
                     "command": f"python scripts/extract_real_tsqa_reps.py --model-key {model} --branch question --init pretrained --seed 7 --device CUDA_DEVICE", **common})
        rows.append({"task": "B_temporal_cache", "model": model, "init": "pretrained", "seed": 7,
                     "command": f"python scripts/extract_real_tsqa_reps.py --model-key {model} --branch temporal --init pretrained --seed 7 --device CUDA_DEVICE", **common})
        for seed in SEEDS:
            rows.append({"task": "B_temporal_cache", "model": model, "init": "random", "seed": seed,
                         "command": f"python scripts/extract_real_tsqa_reps.py --model-key {model} --branch temporal --init random --seed {seed} --device CUDA_DEVICE", **common})
            rows.append({"task": "A_controlled_readout", "model": model, "init": "paired_P_R", "seed": seed,
                         "command": f"python scripts/run_real_tsqa_readout.py --model-key {model} --seed {seed} --device CUDA_DEVICE", **common})
            rows.append({"task": "B_supervised_head", "model": model, "init": "paired_P_R", "seed": seed,
                         "command": f"python scripts/run_real_tsqa_supervised.py --model-key {model} --seed {seed} --device cpu", **common})
    for model in MODELS_C:
        for mode in ("free", "constrained"):
            rows.append({"task": f"C_native_{mode}", "model": model, "init": "pretrained", "seed": 20260918,
                         "command": f"python scripts/run_real_tsqa_native.py --model-key {model} --mode {mode} --device CUDA_DEVICE", **common})
    for row in rows:
        output = expected_output(row)
        if output.is_file():
            row.update(status="COMPLETED", validity="OUTPUT_PRESENT_PROTOCOL_VALIDATION_PENDING",
                       eta="DONE", next_step=f"validate and summarize {output}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "job_status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "reproduction_commands.sh").open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n# Replace CUDA_DEVICE only after checking host GPU ownership.\n")
        for row in rows:
            handle.write(row["command"] + "\n")
    print(f"wrote {len(rows)} locked jobs")


if __name__ == "__main__":
    main()
