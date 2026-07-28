"""Summarize saved Qwen forecast artifacts, including spectral diagnostics."""

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


def summarize_run(path: Path) -> dict[str, object]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    arrays = np.load(path / "predictions.npz")
    valid = arrays["valid"].astype(bool)
    spectral = None
    if np.any(valid):
        spectral = spectral_metrics(arrays["prediction"][valid], arrays["target"][valid])
    rollout = metrics.get("rollout")
    return {
        "run": str(path),
        "dataset": metrics["dataset"],
        "initialization": metrics.get("initialization", "pretrained"),
        "epochs": metrics["epochs"],
        "train_windows": metrics.get("max_train_windows"),
        "test_windows": metrics.get("max_test_windows"),
        "parse_success_rate": metrics["parse_success_rate"],
        "valid_samples": metrics["valid_samples"],
        "rmse": None if metrics["test"] is None else metrics["test"]["rmse"],
        "rollout_rmse_first": None if rollout is None else rollout["rmse"][0],
        "rollout_rmse_last": None if rollout is None else rollout["rmse"][-1],
        "spectral": spectral,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [summarize_run(root) for root in args.roots]
    payload = json.dumps(rows, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
