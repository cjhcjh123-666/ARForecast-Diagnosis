"""Pair corrected Level-2 pretrained/random results without hiding missing coverage."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/temporal_abstraction/corrected_v2"
SOURCE = ROOT / "relations"
OUT = ROOT / "tables"
MODELS = [
    "qwen3_8b_base", "llama31_8b", "llama32_3b", "gemma2_9b", "gemma2_2b",
    "mistral_7b_v03", "deepseek_llm_7b", "olmo2_7b", "olmo2_13b",
    "deepseek_v2_lite",
]
SEEDS = [7, 17, 27]
LOWER_IS_BETTER = {"nmae", "mae", "mse", "rmse"}


def load_runs() -> dict:
    runs = {}
    for path in sorted(SOURCE.glob("level2_*.csv")):
        rows = list(csv.DictReader(path.open()))
        if not rows:
            continue
        key = (rows[0]["model"], rows[0]["init"], int(rows[0]["seed"]))
        if any((r["model"], r["init"], int(r["seed"])) != key for r in rows):
            raise ValueError(f"mixed run identity in {path}")
        runs[key] = (path, rows)
    return runs


def main() -> None:
    runs = load_runs()
    paired, missing = [], []
    for model in MODELS:
        for seed in SEEDS:
            pk, rk = (model, "pretrained", seed), (model, "random", seed)
            if pk not in runs or rk not in runs:
                missing.append({
                    "model": model, "seed": seed,
                    "pretrained": pk in runs, "random": rk in runs,
                })
                continue
            p_path, p_rows = runs[pk]
            r_path, r_rows = runs[rk]
            p = {(x["task"], x["metric"]): x for x in p_rows}
            r = {(x["task"], x["metric"]): x for x in r_rows}
            if set(p) != set(r):
                raise ValueError(f"task mismatch: {p_path} vs {r_path}")
            for key in sorted(p):
                a, b = p[key], r[key]
                if a["manifest_sha256"] != b["manifest_sha256"]:
                    raise ValueError(f"manifest mismatch: {p_path} vs {r_path}")
                pre, rnd = float(a["score"]), float(b["score"])
                delta = rnd - pre if key[1].lower() in LOWER_IS_BETTER else pre - rnd
                paired.append({
                    "model": model, "seed": seed, "task": key[0], "metric": key[1],
                    "pretrained": pre, "random": rnd, "improvement": round(delta, 4),
                    "manifest_sha256": a["manifest_sha256"],
                    "protocol_version": a["protocol_version"],
                })
    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / "level2_paired.csv"
    fields = ["model", "seed", "task", "metric", "pretrained", "random",
              "improvement", "manifest_sha256", "protocol_version"]
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)
    coverage = {
        "expected_models": MODELS,
        "expected_seeds": SEEDS,
        "expected_paired_runs": len(MODELS) * len(SEEDS),
        "complete_paired_runs": len({(x["model"], x["seed"]) for x in paired}),
        "missing": missing,
        "may_claim_complete_10_model_coverage": not missing,
    }
    (OUT / "level2_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
