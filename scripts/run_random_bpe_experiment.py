"""Run a tokenizer-matched random BPE text forecasting baseline."""

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
from models.random_bpe_text_ar import RandomBPETextARModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B")
    parser.add_argument("--dataset", default="synthetic_ar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=64)
    parser.add_argument("--max-test-windows", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/formal_match/random_bpe"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_set, _, test_set, stats = build_dataset(
        args.dataset,
        context_len=args.context_len,
        horizon=args.horizon,
        seed=7,
        max_train_windows=args.max_train_windows,
        max_val_windows=16,
        max_test_windows=args.max_test_windows,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = RandomBPETextARModel(args.tokenizer_path, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for context, future in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(context, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.6f}")

    predictions, targets, texts = [], [], []
    for context, future in test_loader:
        batch_predictions, batch_texts = model.generate_forecast_batch(
            context, args.horizon, max_new_tokens=args.max_new_tokens or args.horizon * 8
        )
        predictions.extend(batch_predictions)
        targets.append(future.numpy())
        texts.extend(batch_texts)
    target_array = np.concatenate(targets, axis=0)
    valid = np.asarray([len(row) == args.horizon for row in predictions], dtype=bool)
    prediction_array = np.full(target_array.shape, np.nan, dtype=np.float32)
    for index, row in enumerate(predictions):
        if len(row) == args.horizon:
            prediction_array[index] = np.asarray(row, dtype=np.float32)
    metrics = forecast_metrics(prediction_array[valid], target_array[valid]) if np.any(valid) else None
    rollout = rollout_error_by_horizon(prediction_array[valid], target_array[valid]) if np.any(valid) else None
    payload = {
        "dataset": args.dataset,
        "tokenizer_path": args.tokenizer_path,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_train_windows": args.max_train_windows,
        "max_test_windows": args.max_test_windows,
        "max_new_tokens": args.max_new_tokens or args.horizon * 8,
        "normalization": {"mean": stats.mean, "std": stats.std},
        "trainable_parameters": model.trainable_parameter_count(),
        "history": history,
        "parse_success_rate": float(np.mean(valid)),
        "valid_samples": int(np.sum(valid)),
        "test": metrics,
        "rollout": rollout,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "predictions.npz", prediction=prediction_array, target=target_array, valid=valid)
    (args.output_dir / "generation_samples.json").write_text(json.dumps(texts[:8], indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    print(json.dumps({"parse_success_rate": payload["parse_success_rate"], "test": metrics}, indent=2))
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
