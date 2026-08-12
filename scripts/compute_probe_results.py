"""Compute probe results from cached hidden states (no GPU needed).

Loads `results/icassp/probe/cache/{train,test}_h_{pretrained,random}.npy`
produced by `run_frozen_probe.py`, rebuilds the same windows/labels, and
reports recognition accuracy (feature floor / frozen pretrained / frozen
random), per-kind accuracy, leave-one-kind-out, and the zero-shot expert
router numbers.  The paired generation part still needs `run_frozen_probe.py`.
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
    parser.add_argument("--cache", type=Path, default=Path("results/icassp/probe/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/icassp/probe/core_results.json"))
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=150)
    parser.add_argument("--n-train-per-kind", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    contexts, futures, kind_ids = build_labeled_windows(
        KINDS, args.context_len, args.horizon, args.n_per_kind, args.seed
    )
    n_per_kind = args.n_per_kind
    n_train = args.n_train_per_kind
    train_idx = np.concatenate(
        [np.arange(k * n_per_kind, k * n_per_kind + n_train) for k in range(len(KINDS))]
    )
    test_idx = np.concatenate(
        [
            np.arange(k * n_per_kind + n_train, (k + 1) * n_per_kind)
            for k in range(len(KINDS))
        ]
    )
    train_label = kind_ids[train_idx]
    test_label = kind_ids[test_idx]
    train_feats = np.stack([temporal_features(x) for x in contexts[train_idx]])
    test_feats = np.stack([temporal_features(x) for x in contexts[test_idx]])

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]

    def expert_errors(idx: np.ndarray) -> np.ndarray:
        err = np.zeros((len(idx), len(experts)))
        for j, expert in enumerate(experts):
            for i in range(len(idx)):
                pred = expert.predict(contexts[idx[i]], args.horizon)
                err[i, j] = float(np.mean((pred - futures[idx[i]]) ** 2))
        return err

    train_err = expert_errors(train_idx)
    test_err = expert_errors(test_idx)
    oracle = test_err.argmin(axis=1)
    best_single_idx = int(train_err.mean(axis=0).argmin())

    def fit(feats: np.ndarray, labels: np.ndarray):
        return softmax_regression(feats, labels, n_iter=800)

    results = {
        "kinds": KINDS,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "n_train_per_kind": n_train,
        "n_test_per_kind": n_per_kind - n_train,
    }

    _, floor_pred = softmax_predict(test_feats, *fit(train_feats, train_label))
    results["feature_floor"] = {"accuracy": accuracy(floor_pred, test_label)}

    probe = {}
    for init in ["pretrained", "random"]:
        train_h = np.load(args.cache / f"train_h_{init}.npy")
        test_h = np.load(args.cache / f"test_h_{init}.npy")
        _, pred = softmax_predict(test_h, *fit(train_h, train_label))
        per_kind = {
            KINDS[k]: accuracy(pred[test_label == k], test_label[test_label == k])
            for k in range(len(KINDS))
        }
        loio = {}
        for held in range(len(KINDS)):
            tr = np.concatenate(
                [np.arange(k * n_train, (k + 1) * n_train) for k in range(len(KINDS)) if k != held]
            )
            n_test = n_per_kind - n_train
            te = np.arange(held * n_test, (held + 1) * n_test)
            _, p = softmax_predict(test_h[te], *fit(train_h[tr], train_label[tr]))
            loio[KINDS[held]] = accuracy(p, test_label[te])
        probe[init] = {
            "accuracy": accuracy(pred, test_label),
            "per_kind": per_kind,
            "loio": loio,
            "loio_mean": float(np.mean(list(loio.values()))),
        }
    results["probe"] = probe

    hard = np.isin(test_label, [3, 4])
    hard_idx = np.where(hard)[0]
    clean = train_label < 3
    router = {}
    for name, tr_feats, te_feats in [
        ("feature", train_feats, test_feats),
        ("probe", np.load(args.cache / "train_h_pretrained.npy"), np.load(args.cache / "test_h_pretrained.npy")),
    ]:
        _, pred = softmax_predict(te_feats[hard], *fit(tr_feats[clean], train_label[clean]))
        router[name] = {
            "routing_acc_vs_oracle": accuracy(pred, oracle[hard]),
            "mse": float(np.mean(test_err[hard_idx, pred])),
        }
    router["oracle"] = {"mse": float(np.mean(test_err[hard].min(axis=1)))}
    router["best_single_expert"] = {
        "mse": float(np.mean(test_err[hard][:, best_single_idx])),
        "name": experts[best_single_idx].name,
    }
    router["uniform_ensemble"] = {"mse": float(np.mean(test_err[hard].mean(axis=1)))}
    results["zero_shot_router"] = router

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
