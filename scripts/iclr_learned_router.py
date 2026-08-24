"""ICLR: non-LLM learned temporal router baselines for the zero-shot routing claim.

Trains a small PatchTST-style patch-embedding Transformer encoder (and an MLP
on raw windows) to classify the 3 *clean* synthetic dynamics (trend/periodic/
local) -- the same training signal the frozen-LLM probe sees -- then evaluates
zero-shot routing on the *unseen* mixture/regime windows (same protocol as E4).
Reports oracle-hit accuracy and downstream MSE against the three univariate
experts, vs the frozen-LLM probe and the hand-feature floor.

This directly answers the reviewer question: "does this need a language model,
or would any supervised nonlinear classifier trained on the same data do?"
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


class PatchEmbed(nn.Module):
    def __init__(self, context_len: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (context_len - patch_len) // stride + 1
        self.n_patches = n_patches
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, P, patch_len)
        return self.proj(patches)


class PatchTSTRouter(nn.Module):
    def __init__(self, context_len: int, patch_len: int, stride: int,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 n_classes: int = 3, dropout: float = 0.1, seed: int = 7):
        super().__init__()
        torch.manual_seed(seed)
        self.patch = PatchEmbed(context_len, patch_len, stride, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.patch.n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch(x) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class MLPRouter(nn.Module):
    def __init__(self, context_len: int, hidden: int = 128, n_classes: int = 3, seed: int = 7):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(context_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_eval(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
               test_x: torch.Tensor, epochs: int, lr: float, batch_size: int,
               device: str, seed: int) -> tuple[np.ndarray, list, nn.Module]:
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = len(train_x)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = train_x[idx].to(device), train_y[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))
    model.eval()
    with torch.no_grad():
        logits = model(test_x.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
    return pred, history, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=150)
    parser.add_argument("--n-train-per-kind", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/iclr/learned_router/summary.json"))
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

    # clean (seen) train subset and hard (unseen) test subset
    clean = train_label < 3
    hard = np.isin(test_label, [3, 4])
    tr_x = torch.from_numpy(contexts[train_idx][clean]).float()
    tr_y = torch.from_numpy(train_label[clean]).long()
    te_x = torch.from_numpy(contexts[test_idx[hard]]).float()
    te_y = test_label[hard]

    # expert errors for oracle / downstream routing
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    err = np.zeros((len(test_idx), len(experts)))
    for j, expert in enumerate(experts):
        for i in range(len(test_idx)):
            pred = expert.predict(contexts[test_idx[i]], args.horizon)
            err[i, j] = float(np.mean((pred - futures[test_idx[i]]) ** 2))
    oracle = err.argmin(axis=1)
    hard_err = err[hard]
    hard_oracle = oracle[hard]
    best_single = int(err.mean(axis=0).argmin())

    results = {
        "context_len": args.context_len, "horizon": args.horizon, "seed": args.seed,
        "train_windows": int(clean.sum()), "hard_test_windows": int(hard.sum()),
        "baselines": {},
    }
    results["baselines"]["oracle"] = {"mse": float(np.mean(hard_err.min(axis=1)))}
    results["baselines"]["best_single"] = {
        "mse": float(np.mean(hard_err[:, best_single])),
        "name": experts[best_single].name,
    }
    results["baselines"]["uniform"] = {"mse": float(np.mean(hard_err.mean(axis=1)))}

    # hand-feature floor
    all_train_feats = np.stack([temporal_features(x) for x in contexts[train_idx]])
    train_feats = all_train_feats[clean]
    from analysis.linear_probe import softmax_predict, softmax_regression
    te_feats = np.stack([temporal_features(x) for x in contexts[test_idx[hard]]])
    _, fpred = softmax_predict(te_feats, *softmax_regression(train_feats, train_label[clean]))
    results["baselines"]["feature"] = {
        "acc_vs_oracle": accuracy(fpred, hard_oracle),
        "mse": float(np.mean(hard_err[np.arange(len(fpred)), fpred])),
        "pred_dist": [int((fpred == k).sum()) for k in range(3)],
    }

    # PatchTST-style encoder
    model = PatchTSTRouter(
        args.context_len, args.patch_len, args.patch_stride,
        d_model=args.d_model, n_layers=args.n_layers, seed=args.seed,
    )
    pred, hist, model = train_eval(model, tr_x, tr_y, te_x, args.epochs, args.lr, args.batch_size, args.device, args.seed)
    # in-domain accuracy on held-out clean test (trend/periodic/local)
    clean_test = np.isin(test_label, [0, 1, 2])
    ct_x = torch.from_numpy(contexts[test_idx[clean_test]]).float()
    with torch.no_grad():
        ct_pred = model(ct_x.to(args.device)).argmax(dim=1).cpu().numpy()
    in_domain_acc = accuracy(ct_pred, test_label[clean_test])
    results["baselines"]["patchtst"] = {
        "acc_vs_oracle": accuracy(pred, hard_oracle),
        "mse": float(np.mean(hard_err[np.arange(len(pred)), pred])),
        "pred_dist": [int((pred == k).sum()) for k in range(3)],
        "train_loss_last": hist[-1] if hist else None,
        "in_domain_acc": float(in_domain_acc),
    }
    print(f"PatchTST router: acc_vs_oracle={results['baselines']['patchtst']['acc_vs_oracle']:.3f} in_domain={in_domain_acc:.3f}")

    # MLP on raw windows
    mlp = MLPRouter(args.context_len, seed=args.seed)
    pred, hist, mlp = train_eval(mlp, tr_x, tr_y, te_x, args.epochs, args.lr, args.batch_size, args.device, args.seed)
    with torch.no_grad():
        ct_pred = mlp(ct_x.to(args.device)).argmax(dim=1).cpu().numpy()
    in_domain_acc = accuracy(ct_pred, test_label[clean_test])
    results["baselines"]["mlp"] = {
        "acc_vs_oracle": accuracy(pred, hard_oracle),
        "mse": float(np.mean(hard_err[np.arange(len(pred)), pred])),
        "pred_dist": [int((pred == k).sum()) for k in range(3)],
        "train_loss_last": hist[-1] if hist else None,
        "in_domain_acc": float(in_domain_acc),
    }
    print(f"MLP router: acc_vs_oracle={results['baselines']['mlp']['acc_vs_oracle']:.3f} in_domain={in_domain_acc:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
