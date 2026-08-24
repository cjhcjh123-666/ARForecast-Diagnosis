"""ICLR: Chronos-Bolt (patch-based T5) recognition + zero-shot routing probe.

Bolt consumes raw time series (no value tokenizer): InstanceNorm -> Patch
(patch=16) -> concat missing-mask -> patch embedding -> [REG token] -> T5
encoder.  We take the REG token's encoder hidden state as the representation
and run the identical linear-probe / zero-shot-routing protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CHRONOS_SRC = REPO_ROOT / "third_party/chronos-forecasting/src"
if str(CHRONOS_SRC) not in sys.path:
    sys.path.insert(0, str(CHRONOS_SRC))

from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


@torch.no_grad()
def bolt_embeddings(model, contexts: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    """REG-token encoder hidden state for each window."""
    outs = []
    for start in range(0, len(contexts), batch_size):
        batch = torch.from_numpy(contexts[start : start + batch_size]).float().to(device)
        hidden, _, _, attn = model.encode(context=batch)
        # REG token is the last position; fall back to last attended position
        idx = attn.sum(dim=1).long() - 1
        idx = torch.clamp(idx, min=0)
        row = torch.arange(len(batch), device=device)
        outs.append(hidden[row, idx].float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/public/chenjiahui/SAFER-TS/code/model_cache/amazon__chronos-bolt-small")
    parser.add_argument("--model-tag", default="chronos-bolt-small")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=150)
    parser.add_argument("--n-train-per-kind", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--softmax-iters", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("results/iclr/chronos/chronos-bolt-small_seed7/summary.json"))
    args = parser.parse_args()

    from chronos import ChronosBoltPipeline
    pipeline = ChronosBoltPipeline.from_pretrained(args.model_path)
    model = pipeline.model.to(args.device).eval()

    contexts, futures, kind_ids = build_labeled_windows(
        KINDS, args.context_len, args.horizon, args.n_per_kind, args.seed
    )
    n = args.n_per_kind
    n_train = args.n_train_per_kind
    train_idx = np.concatenate([np.arange(k * n, k * n + n_train) for k in range(5)])
    test_idx = np.concatenate([np.arange(k * n + n_train, (k + 1) * n) for k in range(5)])
    train_label, test_label = kind_ids[train_idx], kind_ids[test_idx]

    h = bolt_embeddings(model, contexts, args.device)
    print(f"[bolt] embeddings shape {h.shape}")

    _, pred = softmax_predict(h[test_idx], *softmax_regression(h[train_idx], train_label, n_iter=args.softmax_iters))
    recog = float(accuracy(pred, test_label))
    per_kind = {KINDS[k]: float(accuracy(pred[test_label == k], test_label[test_label == k])) for k in range(5)}

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    err = np.zeros((len(test_idx), len(experts)))
    for j, expert in enumerate(experts):
        for i in range(len(test_idx)):
            p = expert.predict(contexts[test_idx[i]], args.horizon)
            err[i, j] = float(np.mean((p - futures[test_idx[i]]) ** 2))
    oracle = err.argmin(axis=1)
    clean = train_label < 3
    hard = np.isin(test_label, [3, 4])
    _, rpred = softmax_predict(h[test_idx[hard]], *softmax_regression(h[train_idx[clean]], train_label[clean], n_iter=args.softmax_iters))
    router_acc = float(accuracy(rpred, oracle[hard]))
    router_mse = float(np.mean(err[hard][np.arange(hard.sum()), rpred]))

    clean_test = test_label < 3
    _, cpred = softmax_predict(h[test_idx[clean_test]], *softmax_regression(h[train_idx[clean]], train_label[clean], n_iter=args.softmax_iters))
    in_domain_acc = float(accuracy(cpred, oracle[clean_test]))

    feats = np.stack([temporal_features(x) for x in contexts])
    _, fpred = softmax_predict(feats[test_idx[hard]], *softmax_regression(feats[train_idx[clean]], train_label[clean], n_iter=args.softmax_iters))
    feat_acc = float(accuracy(fpred, oracle[hard]))

    report = {
        "model": args.model_tag,
        "context_len": args.context_len, "horizon": args.horizon, "seed": args.seed,
        "recognition_acc": recog,
        "per_kind": per_kind,
        "router": {"acc_vs_oracle": router_acc, "mse": router_mse,
                   "in_domain_acc_vs_oracle": in_domain_acc,
                   "feature_acc_vs_oracle": feat_acc,
                   "oracle_mse": float(np.mean(err[hard].min(axis=1))),
                   "best_single_mse": float(np.mean(err[hard][:, err[hard].mean(axis=0).argmin()]))},
        "n_hard": int(hard.sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
