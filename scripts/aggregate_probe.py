"""Aggregate probe results across seeds into mean +/- std tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", type=Path, default=[
        Path("results/icassp/probe/summary.json"),
        Path("results/icassp/probe_seed17/summary.json"),
    ])
    args = parser.parse_args()

    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in args.paths if p.is_file()]
    if not summaries:
        raise SystemExit("no summary.json files found")

    print(f"aggregated over {len(summaries)} seed(s): {[p.name for p in args.paths if p.is_file()]}")
    print("\n== E2 recognition (overall) ==")
    print("feature floor :", mean_std([s["feature_floor"]["accuracy"] for s in summaries]))
    for init in ["pretrained", "random"]:
        print(f"probe {init:11s}:", mean_std([s["probe"][init]["accuracy"] for s in summaries]))
    print("\n== E2 per-kind (pretrained) ==")
    for kind in summaries[0]["probe"]["pretrained"]["per_kind"]:
        vals = [s["probe"]["pretrained"]["per_kind"][kind] for s in summaries]
        print(f"  {kind:9s}: {mean_std(vals)}")

    if "paired_recognize_vs_generate" in summaries[0]:
        print("\n== E3 paired recognize-vs-generate ==")
        pg = [s["paired_recognize_vs_generate"] for s in summaries]
        print("recognize acc :", mean_std([p["recognize_accuracy"] for p in pg]))
        print("gen parse rate:", mean_std([p["parse_rate"] for p in pg]))
        print("gen mse       :", mean_std([p["mse_on_complete"] for p in pg]))
        print("oracle mse    :", mean_std([p["oracle_mse"] for p in pg]))

    if "zero_shot_router" in summaries[0]:
        print("\n== E4 zero-shot router (unseen mixture/regime) ==")
        for key in ["feature", "probe_pretrained", "probe_random"]:
            acc = mean_std([s["zero_shot_router"][key]["routing_acc_vs_oracle"] for s in summaries])
            mse = mean_std([s["zero_shot_router"][key]["mse"] for s in summaries])
            soft = mean_std([s["zero_shot_router"][key]["soft_routing_mse"] for s in summaries])
            print(f"{key:16s} oracle-acc={acc}  mse={mse}  soft={soft}")
        for key in ["oracle", "best_single_expert", "uniform_ensemble"]:
            mse = mean_std([s["zero_shot_router"][key]["mse"] for s in summaries])
            print(f"{key:16s} mse={mse}")


if __name__ == "__main__":
    main()
