"""ICLR: compute zero-shot router numbers purely from cached hidden states.

No model load: reads the seed-aware hidden-state caches produced by
run_frozen_probe.py and computes the same zero-shot router metrics (oracle-hit
accuracy + downstream MSE for feature / frozen-random probe / frozen-pretrained
probe).  Used as a fallback when the model-loading generation step is flaky, and
for cheap multi-seed aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-cache", type=Path, default=Path("results/iclr/probe/qwen3_0.6b/seed7/cache"))
    parser.add_argument("--model-tag", default="qwen3_0.6b")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=150)
    parser.add_argument("--n-train-per-kind", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/iclr/router_from_cache/summary.json"))
    args = parser.parse_args()

    contexts, futures, kind_ids = build_labeled_windows(
        KINDS, args.context_len, args.horizon, args.n_per_kind, args.seed
    )
    n = args.n_per_kind
    n_train = args.n_train_per_kind
    train_idx = np.concatenate([np.arange(k * n, k * n + n_train) for k in range(5)])
    test_idx = np.concatenate([np.arange(k * n + n_train, (k + 1) * n) for k in range(5)])
    train_label, test_label = kind_ids[train_idx], kind_ids[test_idx]

    def cache(stem: str) -> Path:
        seeded = args.probe_cache / f"{stem}_seed{args.seed}.npy"
        if seeded.is_file():
            return seeded
        legacy = args.probe_cache / f"{stem}.npy"
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(f"missing {stem} seed {args.seed} in {args.probe_cache}")

    train_h = np.load(cache("train_h_pretrained"))
    test_h = np.load(cache("test_h_pretrained"))
    train_h_r = np.load(cache("train_h_random"))
    test_h_r = np.load(cache("test_h_random"))

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    err = np.zeros((len(test_idx), len(experts)))
    for j, expert in enumerate(experts):
        for i in range(len(test_idx)):
            pred = expert.predict(contexts[test_idx[i]], args.horizon)
            err[i, j] = float(np.mean((pred - futures[test_idx[i]]) ** 2))
    oracle = err.argmin(axis=1)

    clean = train_label < 3
    hard = np.isin(test_label, [3, 4])
    hard_err, hard_oracle = err[hard], oracle[hard]
    best_single = int(err.mean(axis=0).argmin())

    train_feats = np.stack([temporal_features(x) for x in contexts[train_idx]])
    te_feats = np.stack([temporal_features(x) for x in contexts[test_idx[hard]]])
    _, fpred = softmax_predict(te_feats, *softmax_regression(train_feats[clean], train_label[clean]))
    _, ppred = softmax_predict(test_h[hard], *softmax_regression(train_h[clean], train_label[clean]))
    _, rpred = softmax_predict(test_h_r[hard], *softmax_regression(train_h_r[clean], train_label[clean]))

    report = {
        "model": args.model_tag, "seed": args.seed, "hard_windows": int(hard.sum()),
        "oracle": {"mse": float(np.mean(hard_err.min(axis=1)))},
        "best_single": {"mse": float(np.mean(hard_err[:, best_single])), "name": experts[best_single].name},
        "uniform": {"mse": float(np.mean(hard_err.mean(axis=1)))},
        "feature": {"acc": accuracy(fpred, hard_oracle), "mse": float(np.mean(hard_err[np.arange(len(fpred)), fpred]))},
        "probe_pretrained": {"acc": accuracy(ppred, hard_oracle), "mse": float(np.mean(hard_err[np.arange(len(ppred)), ppred]))},
        "probe_random": {"acc": accuracy(rpred, hard_oracle), "mse": float(np.mean(hard_err[np.arange(len(rpred)), rpred]))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
