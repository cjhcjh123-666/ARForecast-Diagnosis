"""Run a small formal Qwen3-8B text-numeric forecasting experiment."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.metrics import forecast_metrics
from analysis.rollout_error import rollout_error_by_horizon
from data.datasets import build_dataset
from models.qwen_text_ar import QwenTextARForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B")
    parser.add_argument("--dataset", default="synthetic_ar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-train-windows", type=int, default=128)
    parser.add_argument("--max-val-windows", type=int, default=32)
    parser.add_argument("--max-test-windows", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--history-noise-std", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_phase1/synthetic_ar"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_set, val_set, test_set, stats = build_dataset(
        args.dataset,
        context_len=args.context_len,
        horizon=args.horizon,
        seed=7,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        max_test_windows=args.max_test_windows,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    forecaster = QwenTextARForecaster(
        args.model_path, device=args.device, random_init=args.random_init
    )
    trainable = [parameter for parameter in forecaster.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        for context, future in train_loader:
            losses.append(
                forecaster.train_step(
                    context,
                    future,
                    optimizer,
                    history_noise_std=args.history_noise_std,
                )
            )
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.6f}")

    predictions = []
    targets = []
    generation_texts = []
    for context, future in test_loader:
        batch_predictions, batch_texts = forecaster.generate_forecast_batch(
            context, args.horizon, max_new_tokens=args.max_new_tokens or args.horizon * 8
        )
        predictions.extend(batch_predictions)
        targets.append(future.numpy())
        generation_texts.extend(batch_texts)
    target_array = np.concatenate(targets, axis=0)
    valid = np.asarray([len(row) == args.horizon for row in predictions], dtype=bool)
    prediction_array = np.full(target_array.shape, np.nan, dtype=np.float32)
    for index, row in enumerate(predictions):
        if len(row) == args.horizon:
            prediction_array[index] = np.asarray(row, dtype=np.float32)
    if not np.any(valid):
        raise RuntimeError("Qwen generated no complete numeric forecasts")
    metrics = forecast_metrics(prediction_array[valid], target_array[valid])
    rollout = rollout_error_by_horizon(prediction_array[valid], target_array[valid])
    payload = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "initialization": "random" if args.random_init else "pretrained",
        "history_noise_std": args.history_noise_std,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_train_windows": args.max_train_windows,
        "max_test_windows": args.max_test_windows,
        "max_new_tokens": args.max_new_tokens or args.horizon * 8,
        "normalization": {"mean": stats.mean, "std": stats.std},
        "total_parameters": forecaster.total_parameter_count(),
        "trainable_parameters": forecaster.trainable_parameter_count(),
        "history": history,
        "parse_success_rate": float(np.mean(valid)),
        "valid_samples": int(np.sum(valid)),
        "test": metrics,
        "rollout": rollout,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        prediction=prediction_array,
        target=target_array,
        valid=valid,
    )
    (args.output_dir / "generation_samples.json").write_text(
        json.dumps(generation_texts[:8], indent=2, ensure_ascii=True), encoding="utf-8"
    )
    forecaster.model.save_pretrained(args.output_dir / "lora_adapter")
    print(json.dumps({"parse_success_rate": payload["parse_success_rate"], "test": metrics}, indent=2))
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
