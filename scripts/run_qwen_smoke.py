"""Run one Qwen3-8B LoRA text-numeric forecasting step on a local checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import build_dataset
from models.qwen_text_ar import QwenTextARForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    train_set, _, _, _ = build_dataset(
        "synthetic_ar",
        context_len=args.context_len,
        horizon=args.horizon,
        seed=7,
        max_train_windows=args.batch_size,
        max_val_windows=2,
        max_test_windows=2,
    )
    context = torch.stack([train_set[index][0] for index in range(args.batch_size)])
    future = torch.stack([train_set[index][1] for index in range(args.batch_size)])
    forecaster = QwenTextARForecaster(args.model_path, device=args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in forecaster.model.parameters() if parameter.requires_grad],
        lr=2e-4,
    )
    loss = forecaster.train_step(context, future, optimizer)
    prediction, generated_text = forecaster.generate_forecast(context[:1], args.horizon)
    print(f"model_path={args.model_path}")
    print(f"total_parameters={forecaster.total_parameter_count()}")
    print(f"trainable_parameters={forecaster.trainable_parameter_count()}")
    print(f"batch_shape={tuple(context.shape)} target_shape={tuple(future.shape)}")
    print(f"one_step_loss={loss:.6f}")
    print(f"generated_text={generated_text!r}")
    print(f"parsed_forecast={prediction}")


if __name__ == "__main__":
    main()
