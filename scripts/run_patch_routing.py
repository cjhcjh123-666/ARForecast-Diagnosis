"""Patch-level dynamic temporal expert routing (CPU).

Splits the horizon into segments and routes each segment independently using a
router on the most recent context patch.  The showcase case is *regime*
windows (trend switching to periodic mid-window), where a whole-window router
can only pick one expert while a patch router can follow the local dynamics.

Router: hand-feature logistic floor trained on 32-step clean dynamics patches.
Baselines: whole-window routing (64-step router), per-segment oracle (dynamic
upper bound), whole-window oracle (static upper bound), best single expert.
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
from data.dynamics import build_labeled_windows, generate_window
from models.experts import LocalExpert, PeriodicExpert, TrendExpert

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--seg-len", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/icassp/patch_routing/summary.json"))
    args = parser.parse_args()

    horizon = args.horizon
    n_segments = horizon // args.seg_len
    assert horizon % args.seg_len == 0

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]

    # ---- train patch router on 32-step clean-dynamics patches ----
    rng = np.random.default_rng(args.seed)
    train_patches, train_labels = [], []
    for kind_id, kind in enumerate(["trend", "periodic", "local"]):
        for _ in range(args.n_train):
            c, _ = generate_window(kind, args.patch_len, args.seg_len, rng)
            train_patches.append(c)
            train_labels.append(kind_id)
    train_patches = np.stack(train_patches)
    train_labels = np.asarray(train_labels)
    patch_feats = np.stack([temporal_features(x) for x in train_patches])
    patch_router = softmax_regression(patch_feats, train_labels)

    # ---- train whole-window router on 64-step clean dynamics ----
    full_train = build_labeled_windows(
        ["trend", "periodic", "local"], args.context_len, horizon, args.n_train, args.seed
    )
    full_feats = np.stack([temporal_features(x) for x in full_train[0]])
    full_router = softmax_regression(full_feats, full_train[2])

    # ---- test windows across all kinds ----
    test = build_labeled_windows(KINDS, args.context_len, horizon, args.n_test, args.seed + 11)
    contexts, futures, kind_ids = test

    _, whole_pred = softmax_predict(
        np.stack([temporal_features(x) for x in contexts]), *full_router
    )

    # per-segment expert errors and oracle
    full_series = np.concatenate([contexts, futures], axis=1)  # [N, 64+16]
    seg_err = np.zeros((len(contexts), n_segments, len(experts)))
    seg_contexts = np.zeros((len(contexts), n_segments, args.patch_len))
    for i in range(len(contexts)):
        for s in range(n_segments):
            end = args.context_len + s * args.seg_len
            seg_contexts[i, s] = full_series[i, end - args.patch_len : end]
            target = futures[i, s * args.seg_len : (s + 1) * args.seg_len]
            for j, expert in enumerate(experts):
                pred = expert.predict(seg_contexts[i, s], args.seg_len)
                seg_err[i, s, j] = float(np.mean((pred - target) ** 2))
    seg_oracle = seg_err.argmin(axis=2)  # [N, S]

    # patch router decisions per segment
    seg_feats = np.stack(
        [[temporal_features(seg_contexts[i, s]) for s in range(n_segments)] for i in range(len(contexts))]
    )
    seg_prob, seg_pred = softmax_predict(seg_feats.reshape(-1, seg_feats.shape[-1]), *patch_router)
    seg_pred = seg_pred.reshape(len(contexts), n_segments)
    seg_prob = seg_prob.reshape(len(contexts), n_segments, -1)

    # ---- forecast MSE ----
    def forecast_mse(pred_plan: np.ndarray) -> float:
        """pred_plan: [N, H] forecast."""
        return float(np.mean((pred_plan - futures) ** 2))

    def build_forecast(decisions: np.ndarray, per_segment: bool = False) -> np.ndarray:
        """decisions: [N] (whole) or [N, S] (per segment)."""
        out = np.zeros((len(contexts), horizon))
        for i in range(len(contexts)):
            if per_segment:
                for s in range(n_segments):
                    j = decisions[i, s]
                    out[i, s * args.seg_len : (s + 1) * args.seg_len] = experts[j].predict(
                        seg_contexts[i, s], args.seg_len
                    )
            else:
                j = decisions[i]
                out[i] = experts[j].predict(contexts[i], horizon)
        return out

    whole_mse = forecast_mse(build_forecast(whole_pred))
    patch_mse = forecast_mse(build_forecast(seg_pred, per_segment=True))

    # soft patch routing
    soft_forecast = np.zeros((len(contexts), horizon))
    for i in range(len(contexts)):
        for s in range(n_segments):
            for j, expert in enumerate(experts):
                soft_forecast[i, s * args.seg_len : (s + 1) * args.seg_len] += (
                    seg_prob[i, s, j] * expert.predict(seg_contexts[i, s], args.seg_len)
                )
    soft_patch_mse = forecast_mse(soft_forecast)

    # oracle forecasts
    whole_oracle = seg_err.mean(axis=1).argmin(axis=1)  # [N] best expert averaged over segments
    whole_oracle_mse = forecast_mse(build_forecast(whole_oracle))
    patch_oracle_mse = forecast_mse(build_forecast(seg_oracle, per_segment=True))

    # best single expert (global)
    best_single = int(seg_err.mean(axis=(0, 1)).argmin())
    best_single_mse = forecast_mse(
        np.stack([experts[best_single].predict(contexts[i], horizon) for i in range(len(contexts))])
    )

    # routing accuracy vs per-segment oracle
    whole_seg_acc = accuracy(
        np.repeat(whole_pred[:, None], n_segments, axis=1).ravel(), seg_oracle.ravel()
    )
    patch_seg_acc = accuracy(seg_pred.ravel(), seg_oracle.ravel())

    report = {
        "context_len": args.context_len,
        "horizon": horizon,
        "patch_len": args.patch_len,
        "seg_len": args.seg_len,
        "n_segments": n_segments,
        "seed": args.seed,
        "kinds": KINDS,
        "forecast_mse": {
            "whole_router": whole_mse,
            "patch_router": patch_mse,
            "patch_router_soft": soft_patch_mse,
            "whole_oracle": whole_oracle_mse,
            "patch_oracle": patch_oracle_mse,
            "best_single": best_single_mse,
            "best_single_name": experts[best_single].name,
        },
        "routing_acc_vs_seg_oracle": {
            "whole_router": whole_seg_acc,
            "patch_router": patch_seg_acc,
        },
        "per_kind": {},
    }
    for k, name in enumerate(KINDS):
        mask = kind_ids == k
        report["per_kind"][name] = {
            "n": int(mask.sum()),
            "whole_mse": float(np.mean(
                (build_forecast(whole_pred)[mask] - futures[mask]) ** 2
            )),
            "patch_mse": float(np.mean(
                (build_forecast(seg_pred, per_segment=True)[mask] - futures[mask]) ** 2
            )),
            "whole_oracle_mse": float(np.mean(
                (build_forecast(whole_oracle)[mask] - futures[mask]) ** 2
            )),
            "patch_oracle_mse": float(np.mean(
                (build_forecast(seg_oracle, per_segment=True)[mask] - futures[mask]) ** 2
            )),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
