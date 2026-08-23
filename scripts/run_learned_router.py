"""E4 control: non-LLM learned routers trained on the same 3 clean dynamics.

Reviewer asks: does routing need a language model, or would any supervised
nonlinear classifier trained on the same three classes transfer OOD?  This
script trains (a) a small MLP directly on raw 64-step windows and (b) an MLP on
the 9 hand features, then evaluates their zero-shot routing on unseen
mixture/regime, against the pretrained-LLM probe and the feature floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.features import temporal_features
from analysis.linear_probe import accuracy
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


class SmallMLP(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 256, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> SmallMLP:
    torch.manual_seed(seed)
    model = SmallMLP(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.long)
    for _ in range(1200):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xt), Yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/icassp/learned_router/summary.json"))
    args = parser.parse_args()

    train = build_labeled_windows(
        ["trend", "periodic", "local"], args.context_len, args.horizon, args.n_train, args.seed
    )
    test = build_labeled_windows(KINDS, args.context_len, args.horizon, args.n_test, args.seed + 11)
    contexts, futures, kind_ids = test

    # experts + oracle
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    err = np.zeros((len(contexts), len(experts)))
    for j, expert in enumerate(experts):
        for i in range(len(contexts)):
            err[i, j] = float(np.mean((expert.predict(contexts[i], args.horizon) - futures[i]) ** 2))
    oracle = err.argmin(axis=1)
    best_single = int(err.mean(axis=0).argmin())

    # raw-window MLP
    Xtr = train[0].astype(np.float32)
    Ytr = train[2].astype(np.int64)
    model_raw = train_mlp(Xtr, Ytr)
    with torch.no_grad():
        logits = model_raw(torch.tensor(contexts, dtype=torch.float32))
        raw_pred = logits.argmax(dim=1).numpy()

    # feature MLP
    Xfeat_tr = np.stack([temporal_features(x) for x in train[0]]).astype(np.float32)
    model_feat = train_mlp(Xfeat_tr, Ytr)
    Xfeat_te = np.stack([temporal_features(x) for x in contexts]).astype(np.float32)
    with torch.no_grad():
        feat_pred = model_feat(torch.tensor(Xfeat_te)).argmax(dim=1).numpy()

    # linear probe on raw windows (simpler learned baseline)
    from analysis.linear_probe import softmax_predict, softmax_regression
    _, lin_pred = softmax_predict(contexts, *softmax_regression(Xtr, Ytr, n_iter=800))

    hard = np.isin(kind_ids, [3, 4])
    hidx = np.where(hard)[0]
    report = {
        "n_train_per_kind": args.n_train,
        "n_test_per_kind": args.n_test,
        "oracle_mse": float(np.mean(err.min(axis=1))),
        "best_single_mse": float(np.mean(err[:, best_single])),
        "best_single_name": experts[best_single].name,
        "routers": {
            "mlp_raw": {
                "ood_oracle_acc": accuracy(raw_pred[hard], oracle[hard]),
                "ood_mse": float(np.mean(err[hidx, raw_pred[hard]])),
                "ood_pred_dist": [int((raw_pred[hard] == k).sum()) for k in range(3)],
            },
            "mlp_features": {
                "ood_oracle_acc": accuracy(feat_pred[hard], oracle[hard]),
                "ood_mse": float(np.mean(err[hidx, feat_pred[hard]])),
                "ood_pred_dist": [int((feat_pred[hard] == k).sum()) for k in range(3)],
            },
            "linear_raw": {
                "ood_oracle_acc": accuracy(lin_pred[hard], oracle[hard]),
                "ood_mse": float(np.mean(err[hidx, lin_pred[hard]])),
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
