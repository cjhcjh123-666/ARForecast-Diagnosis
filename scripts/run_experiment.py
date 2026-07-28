"""Run one controlled forecasting experiment."""

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

from analysis.instability_index import instability_error_correlation, rollout_instability_index
from analysis.rollout_error import rollout_error_by_horizon
from analysis.spectral_error import spectral_metrics
from data.datasets import build_dataset
from experiments.train import evaluate_model, train_model
from models import ContinuousARModel, DirectForecastModel, TextARModel


DEFAULTS = {
    "dataset": "synthetic_ar",
    "mode": "direct",
    "context_len": 32,
    "horizon": 8,
    "epochs": 2,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 1,
    "dropout": 0.0,
    "max_train_windows": 256,
    "max_val_windows": 128,
    "max_test_windows": 128,
    "total_length": 12000,
    "seed": 7,
    "ett_root": "/public/chenjiahui/波数据时序基座大模型/UniTS-main/dataset/ETT-small",
    "target": "OT",
    "output_root": "results",
    "device": "auto",
}


def parse_args() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    for key, value in DEFAULTS.items():
        argument = f"--{key.replace('_', '-')}"
        if isinstance(value, int):
            parser.add_argument(argument, dest=key, type=int, default=None)
        elif isinstance(value, float):
            parser.add_argument(argument, dest=key, type=float, default=None)
        else:
            parser.add_argument(argument, dest=key, type=str, default=None)
    parsed = vars(parser.parse_args())
    config = dict(DEFAULTS)
    if parsed["config"] is not None:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required when --config is used") from exc
        with parsed["config"].open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config.update(loaded)
    config.update({key: value for key, value in parsed.items() if key != "config" and value is not None})
    if config["mode"] not in {"direct", "continuous_ar", "text_ar"}:
        raise ValueError("mode must be direct, continuous_ar, or text_ar")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(config: dict[str, object]) -> torch.nn.Module:
    common = {
        "context_len": int(config["context_len"]),
        "horizon": int(config["horizon"]),
        "d_model": int(config["d_model"]),
        "n_heads": int(config["n_heads"]),
        "n_layers": int(config["n_layers"]),
        "dropout": float(config["dropout"]),
    }
    if config["mode"] == "direct":
        return DirectForecastModel(**common)
    if config["mode"] == "continuous_ar":
        return ContinuousARModel(**common)
    return TextARModel(**common)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    config = parse_args()
    set_seed(int(config["seed"]))
    device = choose_device(str(config["device"]))
    train_set, val_set, test_set, stats = build_dataset(
        str(config["dataset"]),
        context_len=int(config["context_len"]),
        horizon=int(config["horizon"]),
        seed=int(config["seed"]),
        total_length=int(config["total_length"]),
        ett_root=str(config["ett_root"]),
        target=str(config["target"]),
        max_train_windows=int(config["max_train_windows"]),
        max_val_windows=int(config["max_val_windows"]),
        max_test_windows=int(config["max_test_windows"]),
    )
    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(config["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=int(config["batch_size"]), shuffle=False)
    model = make_model(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"dataset={config['dataset']} mode={config['mode']} device={device} "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)} params={parameter_count}"
    )

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        device=device,
    )
    test_metrics, prediction, target, context = evaluate_model(model, test_loader, device)
    rollout = rollout_error_by_horizon(prediction, target)
    spectral = spectral_metrics(prediction, target)
    instability_count = min(len(context), 128)
    context_tensor = torch.from_numpy(context[:instability_count]).to(device)
    instability = rollout_instability_index(model, context_tensor)
    sample_rmse = np.sqrt(np.mean((prediction[:instability_count] - target[:instability_count]) ** 2, axis=1))
    instability_report = {
        "epsilon": 1e-3,
        "mean": float(np.mean(instability)),
        "median": float(np.median(instability)),
        "max": float(np.max(instability)),
        "error_correlation": instability_error_correlation(instability, sample_rmse),
    }

    output_dir = Path(str(config["output_root"])) / str(config["dataset"]) / str(config["mode"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "device": str(device),
        "parameter_count": parameter_count,
        "normalization": {"mean": stats.mean, "std": stats.std},
        "history": history,
        "test": test_metrics,
        "rollout": rollout,
        "spectral": spectral,
        "instability": instability_report,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    np.savez_compressed(output_dir / "predictions.npz", context=context, prediction=prediction, target=target)
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(json.dumps({"test": test_metrics, "instability": instability_report}, indent=2))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
