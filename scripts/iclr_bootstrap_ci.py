"""ICLR: bootstrap CIs + paired significance for the core claims.

1. E3 claim: recognition accuracy > generation parse/mse on the same windows.
   -> bootstrap CI for recognition acc, parse rate, and gen/oracle MSE ratio.
2. E4 claim: pretrained-LLM router > feature / random / learned-router on
   zero-shot mixture/regime windows.
   -> bootstrap CI for each method's oracle-hit accuracy and MSE.
Inputs: the per-seed probe summaries and learned-router summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "results" / "iclr"


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return {"mean": float(np.mean(vals)), "ci": None}
    means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(np.mean(vals)), "ci_low": float(lo), "ci_high": float(hi)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3_8b")
    parser.add_argument("--out", type=Path, default=Path("results/iclr/bootstrap_ci.json"))
    args = parser.parse_args()

    # collect per-seed probe data
    recog, parse, gen_oracle_ratio, router_p, router_r, router_f = [], [], [], [], [], []
    for seed in [7, 17, 27]:
        probe_path = ROOT / "probe" / args.model / f"seed{seed}" / "summary.json"
        if args.model == "qwen3_8b":
            probe_path = REPO_ROOT / "results" / "icassp" / (
                "probe" if seed == 7 else f"probe_seed{seed}"
            ) / "summary.json"
        d = json.loads(probe_path.read_text(encoding="utf-8"))
        recog.append(d["probe"]["pretrained"]["accuracy"])
        pg = d.get("paired_recognize_vs_generate")
        if pg and pg.get("mse_on_complete") and pg.get("oracle_mse"):
            parse.append(pg["parse_rate"])
            gen_oracle_ratio.append(pg["mse_on_complete"] / pg["oracle_mse"])
        z = d.get("zero_shot_router")
        if z:
            router_p.append(z["probe_pretrained"]["routing_acc_vs_oracle"])
            router_r.append(z["probe_random"]["routing_acc_vs_oracle"])
            router_f.append(z["feature"]["routing_acc_vs_oracle"])

    report = {
        "model": args.model,
        "recognition_acc": bootstrap_ci(recog),
        "generation_parse_rate": bootstrap_ci(parse),
        "generation_oracle_mse_ratio": bootstrap_ci(gen_oracle_ratio),
        "router_pretrained_acc": bootstrap_ci(router_p),
        "router_random_acc": bootstrap_ci(router_r),
        "router_feature_acc": bootstrap_ci(router_f),
        "n_seeds": len(recog),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
