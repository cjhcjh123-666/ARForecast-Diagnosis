"""ICLR: zero-shot frozen-LLM expert router across 15 real datasets.

Reuses the synthetic-dynamics probe (trained on trend/periodic/local hidden
states, cached by run_frozen_probe.py) and routes among the three univariate
experts on windows sampled from real test splits.  Reports per-dataset
oracle-hit accuracy and downstream MSE for: feature router, frozen random
probe, frozen pretrained probe; plus oracle / best-single / uniform baselines.

Datasets: ETTm1, ETTm2, ETTh1, ETTh2, electricity, weather, exchange_rate,
traffic, illness, M4-Daily/Hourly/Weekly/Monthly/Quarterly/Yearly.
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

ETT_ROOT = Path("/public/chenjiahui/波数据时序基座大模型/UniTS-main/dataset")


def _cache(cache_dir: Path, stem: str, seed: int) -> Path:
    seeded = cache_dir / f"{stem}_seed{seed}.npy"
    if seeded.is_file():
        return seeded
    legacy = cache_dir / f"{stem}.npy"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"missing cache {stem} (seed {seed}) under {cache_dir}")


# ---------------------------------------------------------------- data loaders

def _load_ett(name: str) -> np.ndarray:
    """Load OT channel of an ETT dataset, standardized by train."""
    csv_path = ETT_ROOT / "ETT-small" / f"{name}.csv"
    frame = pd.read_csv(csv_path)
    values = frame["OT"].to_numpy(dtype=np.float32)
    is_minute = name.endswith("m1") or name.endswith("m2")
    per_month = 30 * 24 * (4 if is_minute else 1)
    train_end = 12 * per_month
    val_end = train_end + 4 * per_month
    return values[train_end:], values[:train_end]


def _load_channel_csv(folder: str, target: str, cut: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Load one channel of a date+channels csv; returns (test_vals, train_vals)."""
    csv_path = ETT_ROOT / folder / f"{folder}.csv"
    frame = pd.read_csv(csv_path)
    if target not in frame.columns:
        target = frame.columns[1]
    values = frame[target].to_numpy(dtype=np.float32)
    cut_idx = int(cut * len(values))
    return values[cut_idx:], values[:cut_idx]


def _load_illness() -> tuple[np.ndarray, np.ndarray]:
    csv_path = ETT_ROOT / "illness" / "national_illness.csv"
    frame = pd.read_csv(csv_path)
    values = frame["OT"].to_numpy(dtype=np.float32)
    cut = int(0.7 * len(values))
    return values[cut:], values[:cut]


def _load_m4(freq: str) -> tuple[np.ndarray, np.ndarray]:
    """Load M4 train CSV (series as rows); returns list of series arrays.

    M4 test files only contain the short fixed horizon (<=48 points), which is
    shorter than our 16-step future for most frequencies, so we sample windows
    from within each training series (long enough for all frequencies).
    """
    train = pd.read_csv(ETT_ROOT / "m4" / f"{freq}-train.csv", index_col=0)
    rows = []
    for col in train.columns:
        tr = train[col].dropna().to_numpy(dtype=np.float32)
        if len(tr) > 2 * 64:
            rows.append(tr)
    return rows  # list of train series


DATASETS = {
    "ettm1": lambda: _load_ett("ETTm1"),
    "ettm2": lambda: _load_ett("ETTm2"),
    "etth1": lambda: _load_ett("ETTh1"),
    "etth2": lambda: _load_ett("ETTh2"),
    "electricity": lambda: _load_channel_csv("electricity", "OT"),
    "weather": lambda: _load_channel_csv("weather", "T (degC)"),
    "exchange_rate": lambda: _load_channel_csv("exchange_rate", "OT"),
    "traffic": lambda: _load_channel_csv("traffic", "OT"),
    "illness": _load_illness,
    "m4_daily": lambda: _load_m4("Daily"),
    "m4_hourly": lambda: _load_m4("Hourly"),
    "m4_weekly": lambda: _load_m4("Weekly"),
    "m4_monthly": lambda: _load_m4("Monthly"),
    "m4_quarterly": lambda: _load_m4("Quarterly"),
    "m4_yearly": lambda: _load_m4("Yearly"),
}


def _standardize(series: np.ndarray, train: np.ndarray) -> np.ndarray:
    return (series - train.mean()) / (train.std() + 1e-6)


def sample_windows(
    dataset: str, context_len: int, horizon: int, max_windows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (contexts, futures) standardized windows from the test split."""
    rng = np.random.default_rng(seed)
    if dataset.startswith("m4_"):
        series_list = DATASETS[dataset]()
        contexts, futures = [], []
        for full in series_list:
            if len(full) < context_len + horizon:
                continue
            s = len(full) - context_len - horizon
            start = int(rng.integers(0, s + 1))
            contexts.append(full[start : start + context_len])
            futures.append(full[start + context_len : start + context_len + horizon])
        if len(contexts) == 0:
            raise RuntimeError(f"no m4 series long enough for {dataset}")
        idx = rng.choice(len(contexts), size=min(max_windows, len(contexts)), replace=False)
        ctx = np.stack([contexts[i] for i in idx])
        fut = np.stack([futures[i] for i in idx])
        mean = ctx.mean(axis=1, keepdims=True)
        std = ctx.std(axis=1, keepdims=True) + 1e-6
        return (ctx - mean) / std, (fut - mean) / std

    test, train = DATASETS[dataset]()
    test = _standardize(test, train)
    available = len(test) - context_len - horizon + 1
    n = min(max_windows, available)
    starts = np.sort(rng.choice(available, size=n, replace=False))
    contexts = np.stack([test[s : s + context_len] for s in starts])
    futures = np.stack(
        [test[s + context_len : s + context_len + horizon] for s in starts]
    )
    return contexts, futures


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B"
    )
    parser.add_argument("--model-tag", default="qwen3_8b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--probe-cache", type=Path, default=Path("results/icassp/probe/cache")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/iclr/multi_dataset_router/summary.json")
    )
    parser.add_argument(
        "--datasets", default="ettm1,ettm2,etth1,etth2,electricity,weather,exchange_rate,traffic,illness,m4_daily,m4_hourly,m4_weekly,m4_monthly,m4_quarterly,m4_yearly"
    )
    args = parser.parse_args()

    # synthetic training hidden states + labels (trend/periodic/local only)
    train_h = np.load(_cache(args.probe_cache, "train_h_pretrained", args.seed))
    c, _, kk = build_labeled_windows(KINDS, args.context_len, args.horizon, 150, args.seed)
    tr = np.concatenate([np.arange(k * 150, k * 150 + 90) for k in range(3)])
    clean_label = kk[tr]
    syn_feats = np.stack([temporal_features(x) for x in c[tr]])

    train_h_r = np.load(_cache(args.probe_cache, "train_h_random", args.seed))

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    report = {"model": args.model_tag, "context_len": args.context_len,
              "horizon": args.horizon, "seed": args.seed, "datasets": {}}

    datasets = [d for d in args.datasets.split(",") if d]
    windows_cache = Path(args.out).parent / f"windows_{args.model_tag}_s{args.seed}"
    windows_cache.mkdir(parents=True, exist_ok=True)

    # ---- pre-load each initialization once; extract + cache per dataset ----
    data_ctx, data_fut = {}, {}
    for dname in datasets:
        wpath = windows_cache / f"{dname}_windows.npz"
        if wpath.is_file():
            z = np.load(wpath)
            data_ctx[dname], data_fut[dname] = z["ctx"], z["fut"]
            print(f"[{dname}] cached windows {data_ctx[dname].shape}")
            continue
        try:
            contexts, futures = sample_windows(
                dname, args.context_len, args.horizon, args.max_windows, args.seed
            )
            data_ctx[dname], data_fut[dname] = contexts, futures
            np.savez(wpath, ctx=contexts, fut=futures)
            print(f"[{dname}] windows {contexts.shape}")
        except Exception as exc:  # noqa: BLE001
            report["datasets"][dname] = {"error": str(exc)}
            print(f"[{dname}] ERROR {exc}")

    hidden_cache = Path(args.out).parent / f"hidden_{args.model_tag}_s{args.seed}"
    hidden_cache.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model(args.model_path, args.device, False)
    for dname in datasets:
        if dname not in data_ctx:
            continue
        hpath = hidden_cache / f"{dname}_pretrained.npy"
        if hpath.is_file():
            continue
        h = extract_last_hidden(model, tokenizer, data_ctx[dname], args.device, batch_size=32)
        np.save(hpath, h)
        print(f"[{dname}] extracted pretrained hidden")
    del model
    torch.cuda.empty_cache()

    model_r, tokenizer_r = _load_model(args.model_path, args.device, True)
    for dname in datasets:
        if dname not in data_ctx:
            continue
        hpath = hidden_cache / f"{dname}_random.npy"
        if hpath.is_file():
            continue
        h = extract_last_hidden(model_r, tokenizer_r, data_ctx[dname], args.device, batch_size=32)
        np.save(hpath, h)
        print(f"[{dname}] extracted random hidden")
    del model_r
    torch.cuda.empty_cache()

    # ---- compute routers for every dataset ----
    for dname in datasets:
        if dname not in data_ctx:
            continue
        contexts, futures = data_ctx[dname], data_fut[dname]

        errors = np.zeros((len(contexts), len(experts)))
        for j, expert in enumerate(experts):
            for i in range(len(contexts)):
                pred = expert.predict(contexts[i], args.horizon)
                errors[i, j] = float(np.mean((pred - futures[i]) ** 2))
        oracle = errors.argmin(axis=1)
        best_single = int(errors.mean(axis=0).argmin())

        et_feats = np.stack([temporal_features(x) for x in contexts])
        _, fpred = softmax_predict(et_feats, *softmax_regression(syn_feats, clean_label))

        et_h = np.load(hidden_cache / f"{dname}_pretrained.npy")
        _, ppred = softmax_predict(et_h, *softmax_regression(train_h[tr], clean_label))

        et_h_r = np.load(hidden_cache / f"{dname}_random.npy")
        _, rpred = softmax_predict(et_h_r, *softmax_regression(train_h_r[tr], clean_label))

        row = {
            "windows": len(contexts),
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
            "probe_pretrained": {
                "mse": float(np.mean(errors[np.arange(len(contexts)), ppred])),
                "acc_vs_oracle": accuracy(ppred, oracle),
                "pred_dist": [int((ppred == k).sum()) for k in range(3)],
            },
            "probe_random": {
                "mse": float(np.mean(errors[np.arange(len(contexts)), rpred])),
                "acc_vs_oracle": accuracy(rpred, oracle),
                "pred_dist": [int((rpred == k).sum()) for k in range(3)],
            },
        }
        report["datasets"][dname] = row
        print(f"[{dname}] windows={row['windows']} "
              f"feat_acc={row['feature_router']['acc_vs_oracle']:.3f} "
              f"llm_acc={row['probe_pretrained']['acc_vs_oracle']:.3f} "
              f"rand_acc={row['probe_random']['acc_vs_oracle']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
