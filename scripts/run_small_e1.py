"""E1 fairness control on small Qwen3 models: pretrained vs random x full-FT vs LoRA.

Addresses the reviewer concern that the 8B LoRA-only comparison may be
unfair because a random 8B backbone frozen except a rank-8 adapter can barely
learn.  This script trains Qwen3-0.6B / Qwen3-1.7B under the same protocol
with either LoRA (rank 8) or full fine-tuning, for pretrained and random
initializations, and reports forecast MSE + parse coverage.
"""

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
    parser.add_argument("--model-path", default="/9950backfile/chenjiahui/hf_cache/Qwen3-0.6B")
    parser.add_argument("--dataset", default="ettm1")
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-windows", type=int, default=256)
    parser.add_argument("--max-val-windows", type=int, default=64)
    parser.add_argument("--max-test-windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/icassp/small_e1"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_set, val_set, test_set, stats = build_dataset(
        args.dataset,
        context_len=args.context_len,
        horizon=args.horizon,
        seed=args.seed,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        max_test_windows=args.max_test_windows,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    forecaster = QwenTextARForecaster(
        args.model_path,
        device=args.device,
        random_init=args.random_init,
        use_lora=(args.mode == "lora"),
    )
    trainable = [p for p in forecaster.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        for context, future in train_loader:
            losses.append(forecaster.train_step(context, future, optimizer))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.6f}")

    predictions, targets = [], []
    for context, future in test_loader:
        batch_pred, _ = forecaster.generate_forecast_batch(context, args.horizon)
        predictions.extend(batch_pred)
        targets.append(future.numpy())
    target_array = np.concatenate(targets, axis=0)
    valid = np.asarray([len(row) == args.horizon for row in predictions], dtype=bool)
    prediction_array = np.full(target_array.shape, np.nan, dtype=np.float32)
    for i, row in enumerate(predictions):
        if len(row) == args.horizon:
            prediction_array[i] = np.asarray(row, dtype=np.float32)
    metrics = (
        forecast_metrics(prediction_array[valid], target_array[valid])
        if np.any(valid)
        else {"mse": None, "mae": None}
    )
    payload = {
        "model": Path(args.model_path).name,
        "dataset": args.dataset,
        "mode": args.mode,
        "initialization": "random" if args.random_init else "pretrained",
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "parse_success_rate": float(np.mean(valid)),
        "valid_samples": int(np.sum(valid)),
        "test": metrics,
        "total_parameters": forecaster.total_parameter_count(),
        "trainable_parameters": forecaster.trainable_parameter_count(),
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"parse": payload["parse_success_rate"], "test": metrics}))
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
