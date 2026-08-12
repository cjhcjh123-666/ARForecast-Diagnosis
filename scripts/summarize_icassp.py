"""Aggregate the ICASSP E1 controlled forecast sweep into a markdown table.

Reads `results/icassp/<dataset>/<init>/seed<seed>/metrics.json` plus
`predictions.npz`, groups by (dataset, init), and reports mean +/- std over
seeds for MSE, MAE, parse rate, and spectral amplitude MAE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.spectral_error import spectral_metrics


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/icassp"))
    parser.add_argument("--out", type=Path, default=Path("results/icassp/E1_table.md"))
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted((args.root).glob("*/qwen3_8b*/seed*/metrics.json")):
        run_dir = metrics_path.parent
        dataset = run_dir.parent.parent.name
        init = run_dir.parent.name
        seed = int(run_dir.name.replace("seed", ""))
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        mse = payload["test"]["mse"]
        mae = payload["test"]["mae"]
        parse_rate = payload["parse_success_rate"]
        pred_path = run_dir / "predictions.npz"
        if pred_path.is_file():
            data = np.load(pred_path)
            valid = data["valid"].astype(bool)
            pred = data["prediction"][valid]
            target = data["target"][valid]
            amplitude_mae = spectral_metrics(pred, target)["amplitude_mae"] if len(pred) else float("nan")
        else:
            amplitude_mae = float("nan")
        rows.append(
            {
                "dataset": dataset,
                "init": "random" if "random" in init else "pretrained",
                "seed": seed,
                "mse": mse,
                "mae": mae,
                "parse_rate": parse_rate,
                "amplitude_mae": amplitude_mae,
            }
        )

    lines = ["| Dataset | Init | MSE (mean±std) | MAE | Parse rate | Spectral amp MAE | Rel. MSE ↓ |"]
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    summary = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        for init in ["pretrained", "random"]:
            group = [r for r in rows if r["dataset"] == dataset and r["init"] == init]
            if not group:
                continue
            mse_mean, mse_std = mean_std([r["mse"] for r in group])
            mae_mean, _ = mean_std([r["mae"] for r in group])
            parse_mean, _ = mean_std([r["parse_rate"] for r in group])
            amp_mean, _ = mean_std([r["amplitude_mae"] for r in group])
            rel = ""
            if init == "pretrained":
                random_group = [r for r in rows if r["dataset"] == dataset and r["init"] == "random"]
                if random_group:
                    random_mse, _ = mean_std([r["mse"] for r in random_group])
                    rel = f"{(random_mse / mse_mean - 1.0) * 100:.1f}%"
            lines.append(
                f"| {dataset} | {init} | {mse_mean:.5f} ± {mse_std:.5f} | {mae_mean:.5f} "
                f"| {parse_mean:.4f} | {amp_mean:.5f} | {rel} |"
            )
            summary.setdefault(dataset, {})[init] = {
                "mse_mean": mse_mean,
                "mse_std": mse_std,
                "mae_mean": mae_mean,
                "parse_mean": parse_mean,
                "amplitude_mae": amp_mean,
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.root / "E1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
