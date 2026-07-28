"""Measure text-rollout sensitivity for a saved Qwen LoRA forecast adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.instability_index import instability_error_correlation
from data.datasets import build_dataset
from models.qwen_text_ar import QwenTextARForecaster


def _to_array(rows: list[list[float]], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray([len(row) == shape[1] for row in rows], dtype=bool)
    values = np.full(shape, np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        if valid[index]:
            values[index] = np.asarray(row, dtype=np.float32)
    return values, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--dataset", default="synthetic_sine")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-test-windows", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("results/instability/synthetic_sine"))
    args = parser.parse_args()
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")

    _, _, test_set, _ = build_dataset(
        args.dataset,
        context_len=args.context_len,
        horizon=args.horizon,
        seed=7,
        max_train_windows=1,
        max_val_windows=1,
        max_test_windows=args.max_test_windows,
    )
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    forecaster = QwenTextARForecaster(
        args.model_path, device=args.device, adapter_path=args.adapter_path
    )

    contexts, targets, baseline_rows, perturbed_rows = [], [], [], []
    for context, future in loader:
        perturbed_context = context.clone()
        perturbed_context[:, -1] += args.epsilon
        baseline, _ = forecaster.generate_forecast_batch(
            context, args.horizon, max_new_tokens=args.max_new_tokens
        )
        perturbed, _ = forecaster.generate_forecast_batch(
            perturbed_context, args.horizon, max_new_tokens=args.max_new_tokens
        )
        contexts.append(context.numpy())
        targets.append(future.numpy())
        baseline_rows.extend(baseline)
        perturbed_rows.extend(perturbed)

    target_array = np.concatenate(targets, axis=0)
    baseline_array, baseline_valid = _to_array(
        baseline_rows, (len(target_array), args.horizon)
    )
    perturbed_array, perturbed_valid = _to_array(
        perturbed_rows, (len(target_array), args.horizon)
    )
    valid = baseline_valid & perturbed_valid
    if not np.any(valid):
        raise RuntimeError("no complete baseline and perturbed forecasts were generated")
    divergence = np.sqrt(np.mean((perturbed_array[valid] - baseline_array[valid]) ** 2, axis=1))
    instability = divergence / args.epsilon
    error = np.sqrt(np.mean((baseline_array[valid] - target_array[valid]) ** 2, axis=1))
    payload = {
        "dataset": args.dataset,
        "adapter_path": args.adapter_path,
        "epsilon": args.epsilon,
        "horizon": args.horizon,
        "requested_windows": len(target_array),
        "baseline_parse_success_rate": float(np.mean(baseline_valid)),
        "perturbed_parse_success_rate": float(np.mean(perturbed_valid)),
        "valid_samples": int(np.sum(valid)),
        "rollout_instability_index": {
            "mean": float(np.mean(instability)),
            "median": float(np.median(instability)),
            "max": float(np.max(instability)),
        },
        "baseline_rmse_mean": float(np.mean(error)),
        "instability_error_correlation": instability_error_correlation(instability, error),
        "note": "epsilon=0.01 matches the two-decimal text representation; smaller changes can be invisible.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "perturbation.npz",
        baseline=baseline_array,
        perturbed=perturbed_array,
        target=target_array,
        valid=valid,
    )
    print(json.dumps(payload, indent=2))
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
