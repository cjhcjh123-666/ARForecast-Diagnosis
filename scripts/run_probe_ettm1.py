"""Zero-shot frozen-LLM expert router applied to real ETT data (GPU).

Trains the routing probe on synthetic dynamics hidden states (cached from the
probe experiment), extracts frozen-Qwen hidden states for ETTm1 test windows,
and routes among the three univariate experts.  Compares against the feature
router and the oracle / best-single baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _format_values, _load_model, extract_last_hidden

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def _cache(cache_dir: Path, stem: str, seed: int) -> Path:
    seeded = cache_dir / f"{stem}_seed{seed}.npy"
    if seeded.is_file():
        return seeded
    legacy = cache_dir / f"{stem}.npy"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"missing cache {stem} (seed {seed}) under {cache_dir}")


def load_ett_all_channels(root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    wanted = f"{dataset}.csv".lower()
    csv_path = next(
        (p for p in root.iterdir() if p.is_file() and p.name.lower() == wanted),
        root / f"{dataset}.csv",
    )
    frame = pd.read_csv(csv_path)
    channels = frame[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]].to_numpy(
        dtype=np.float32
    )
    is_minute = dataset.startswith("ettm")
    per_month = 30 * 24 * (4 if is_minute else 1)
    train_end = 12 * per_month
    val_end = train_end + 4 * per_month
    train = channels[:train_end]
    test = channels[val_end:]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (test - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B"
    )
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--dataset", default="ettm1", choices=["ettm1", "etth1", "etth2"])
    parser.add_argument(
        "--ett-root", type=Path,
        default=Path("/public/chenjiahui/波数据时序基座大模型/UniTS-main/dataset/ETT-small"),
    )
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-cache", type=Path, default=Path("results/icassp/probe/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/icassp/ettm1_router/probe_summary.json"))
    args = parser.parse_args()

    _, test_all = load_ett_all_channels(args.ett_root, args.dataset)
    ot = 6
    rng = np.random.default_rng(args.seed)
    available = len(test_all) - args.context_len - args.horizon + 1
    starts = np.sort(rng.choice(available, size=min(args.max_windows, available), replace=False))
    contexts = np.stack([test_all[s : s + args.context_len, ot] for s in starts])
    targets = np.stack(
        [test_all[s + args.context_len : s + args.context_len + args.horizon, ot] for s in starts]
    )

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    errors = np.zeros((len(contexts), len(experts)))
    for j, expert in enumerate(experts):
        for i in range(len(contexts)):
            pred = expert.predict(contexts[i], args.horizon)
            errors[i, j] = float(np.mean((pred - targets[i]) ** 2))
    oracle = errors.argmin(axis=1)

    # synthetic training hidden states (probe router)
    train_h = np.load(_cache(args.probe_cache, "train_h_pretrained", args.seed))
    c, _, kk = build_labeled_windows(KINDS, args.context_len, args.horizon, 150, args.seed)
    tr = np.concatenate([np.arange(k0 * 150, k0 * 150 + 90) for k0 in range(3)])
    clean_label = kk[tr]

    model, tokenizer = _load_model(args.model_path, args.device, False)
    et_h = extract_last_hidden(model, tokenizer, contexts, args.device, batch_size=32)
    del model
    torch.cuda.empty_cache()

    _, probe_pred = softmax_predict(et_h, *softmax_regression(train_h[tr], clean_label))

    # random same-architecture probe control
    model_r, tokenizer_r = _load_model(args.model_path, args.device, True)
    et_h_r = extract_last_hidden(model_r, tokenizer_r, contexts, args.device, batch_size=32)
    del model_r
    torch.cuda.empty_cache()
    train_h_r = np.load(_cache(args.probe_cache, "train_h_random", 7))
    _, probe_r_pred = softmax_predict(et_h_r, *softmax_regression(train_h_r[tr], clean_label))

    # feature router
    syn_feats = np.stack([temporal_features(x) for x in c[tr]])
    et_feats = np.stack([temporal_features(x) for x in contexts])
    _, fpred = softmax_predict(et_feats, *softmax_regression(syn_feats, clean_label))

    best_single = int(errors.mean(axis=0)[:3].argmin())
    report = {
        "windows": len(contexts),
        "expert_names": [e.name for e in experts],
        "per_expert_mean_mse": {
            e.name: float(errors[:, j].mean()) for j, e in enumerate(experts)
        },
        "oracle_mse": float(np.mean(errors.min(axis=1))),
        "best_single_mse": float(np.mean(errors[:, best_single])),
        "best_single_name": experts[best_single].name,
        "uniform_ensemble_mse": float(np.mean(errors.mean(axis=1))),
        "feature_router": {
            "mse": float(np.mean(errors[np.arange(len(contexts)), fpred])),
            "acc_vs_oracle": accuracy(fpred, oracle),
            "pred_dist": [int((fpred == k).sum()) for k in range(3)],
        },
        "probe_router": {
            "mse": float(np.mean(errors[np.arange(len(contexts)), probe_pred])),
            "acc_vs_oracle": accuracy(probe_pred, oracle),
            "pred_dist": [int((probe_pred == k).sum()) for k in range(3)],
        },
        "probe_random_router": {
            "mse": float(np.mean(errors[np.arange(len(contexts)), probe_r_pred])),
            "acc_vs_oracle": accuracy(probe_r_pred, oracle),
            "pred_dist": [int((probe_r_pred == k).sum()) for k in range(3)],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
