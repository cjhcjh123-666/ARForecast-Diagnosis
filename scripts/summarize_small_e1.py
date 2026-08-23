"""Aggregate the E1 small-model 2x2 (pretrained/random x LoRA/full) table."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("results/icassp/small_e1")
    rows = []
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        p = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": p["model"],
                "mode": p["mode"],
                "init": p["initialization"],
                "mse": p["test"].get("mse"),
                "parse": p["parse_success_rate"],
            }
        )
    print("| Model | Mode | Init | MSE | Parse |")
    print("| --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=lambda r: (r["model"], r["mode"], r["init"])):
        mse = f"{r['mse']:.4f}" if r["mse"] is not None else "NA"
        print(f"| {r['model']} | {r['mode']} | {r['init']} | {mse} | {r['parse']:.3f} |")


if __name__ == "__main__":
    main()
