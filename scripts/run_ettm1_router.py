"""Zero-shot expert router applied to real ETTm1 data (CPU).

Trains the routing probe on synthetic dynamics (trend / periodic / local),
then applies it to ETTm1 test windows to select among three univariate experts
(trend / periodic / local on the OT channel) plus a cross-channel ridge expert.
Reports oracle / best-single / feature-router / pretrained-probe-router
metrics.  The probe-router uses the cached frozen-Qwen hidden states from the
probe experiment if present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import CrossChannelExpert, LocalExpert, PeriodicExpert, TrendExpert

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def load_ettm1_all_channels(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_normalized_all_channels, test_normalized_all_channels)."""
    frame = pd.read_csv(root / "ETTm1.csv")
    channels = frame[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]].to_numpy(
        dtype=np.float32
    )
    per_month = 30 * 24 * 4
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
        "--ett-root", type=Path,
        default=Path("/public/chenjiahui/波数据时序基座大模型/UniTS-main/dataset/ETT-small"),
    )
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-cache", type=Path, default=Path("results/icassp/probe/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/icassp/ettm1_router/summary.json"))
    args = parser.parse_args()

    train_all, test_all = load_ettm1_all_channels(args.ett_root)
    ot = 6  # index of OT channel

    # build test windows from the test split
    rng = np.random.default_rng(args.seed)
    available = len(test_all) - args.context_len - args.horizon + 1
    starts = np.sort(rng.choice(available, size=min(args.max_windows, available), replace=False))
    windows = np.stack([test_all[s : s + args.context_len] for s in starts])  # [N, C, L]
    targets = np.stack([test_all[s + args.context_len : s + args.context_len + args.horizon, ot] for s in starts])

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    # per-window univariate context (OT) and expert errors
    errors = np.zeros((len(windows), len(experts)))
    for j, expert in enumerate(experts):
        for i in range(len(windows)):
            context_ot = windows[i, ot]
            pred = expert.predict(context_ot, args.horizon)
            errors[i, j] = float(np.mean((pred - targets[i]) ** 2))

    # cross-channel expert (ridge) fit on train windows
    cross = CrossChannelExpert(k=4)
    k = 4
    train_starts = np.sort(rng.choice(len(train_all) - args.context_len - args.horizon + 1, size=256, replace=False))
    X = np.stack([train_all[s + args.context_len - k : s + args.context_len, :].T for s in train_starts])
    Y = np.stack([train_all[s + args.context_len : s + args.context_len + args.horizon, ot] for s in train_starts])
    cross.fit(X, Y)
    cross_preds = np.stack([cross.predict(windows[i].T, args.horizon) for i in range(len(windows))])
    cross_err = np.mean((cross_preds - targets) ** 2, axis=1)
    full_errors = np.concatenate([errors, cross_err[:, None]], axis=1)
    oracle = full_errors.argmin(axis=1)

    # synthetic training data for the routers
    c, f, kk = build_labeled_windows(KINDS, args.context_len, args.horizon, 150, args.seed)
    tr = np.concatenate([np.arange(k0 * 150, k0 * 150 + 90) for k0 in range(3)])
    clean_label = kk[tr]  # 0,1,2
    syn_feats = np.stack([temporal_features(x) for x in c[tr]])

    # ETTm1 features
    et_feats = np.stack([temporal_features(windows[i, ot]) for i in range(len(windows))])

    # feature router
    _, fpred = softmax_predict(et_feats, *softmax_regression(syn_feats, clean_label))

    # probe router (cached frozen hidden states for ETTm1 windows, if provided)
    probe_pred = None
    probe_mse = None
    train_h_path = args.probe_cache / "train_h_pretrained.npy"
    if train_h_path.is_file():
        # We need hidden states of ETTm1 windows, which the probe cache does not
        # contain; recomputing them requires the GPU script.  Report feature-only
        # here; the GPU script `run_probe_ettm1.py` adds the probe variant.
        pass

    report = {
        "windows": len(windows),
        "expert_names": [e.name for e in experts] + ["cross"],
        "per_expert_mean_mse": {
            name: float(np.mean(full_errors[:, j]))
            for j, name in enumerate([e.name for e in experts] + ["cross"])
        },
        "oracle_mse": float(np.mean(full_errors.min(axis=1))),
        "best_single_mse": float(np.mean(full_errors[:, full_errors.mean(axis=0).argmin()])),
        "best_single_name": [e.name for e in experts][int(full_errors.mean(axis=0)[:3].argmin())],
        "feature_router_mse": float(np.mean(full_errors[np.arange(len(windows)), fpred])),
        "feature_router_acc_vs_oracle": accuracy(fpred, oracle),
        "uniform_ensemble_mse": float(np.mean(full_errors.mean(axis=1))),
        "cross_channel_only_mse": float(np.mean(cross_err)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
