"""ICLR: WINDOW-level bootstrap CIs for the core claims.

Resampling unit = individual evaluation window (not seed).
For each model x seed, we recompute per-window:
  - recognition correctness (5-way probe on cached hidden states)
  - routing oracle-hit + routed MSE (probe trained on 3 clean kinds, applied
    to unseen mixture/regime windows; oracle = best expert per window)
  - generation MSE / oracle ratio (per parsed window; requires GPU for
    frozen generation, saved once to JSON)

All per-window outcomes are pooled across seeds and bootstrapped 2,000x
(percentile 95% CI), which yields much tighter intervals than seed-level.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _cache_dir(model: str, seed: int) -> Path:
    override = os.environ.get("ICLR_PROBE_CACHE_ROOT")
    if override:
        return Path(override) / f"seed{seed}" / "cache"
    if model == "qwen3_8b":
        root = REPO_ROOT / "results/icassp" / ("probe" if seed == 7 else f"probe_seed{seed}")
    else:
        root = REPO_ROOT / "results/iclr" / "probe" / model / f"seed{seed}"
    return root / "cache"


def per_window_outcomes(model: str, seed: int) -> dict:
    """Return per-window recognition + routing outcomes for one seed."""
    cd = _cache_dir(model, seed)
    tr_h = np.load(cd / f"train_h_pretrained_seed{seed}.npy")
    te_h = np.load(cd / f"test_h_pretrained_seed{seed}.npy")
    tr_h_r = np.load(cd / f"train_h_random_seed{seed}.npy")
    te_h_r = np.load(cd / f"test_h_random_seed{seed}.npy")

    contexts, futures, kind_ids = build_labeled_windows(KINDS, 64, 16, 150, seed)
    n = 150
    n_train = 90
    train_idx = np.concatenate([np.arange(k * n, k * n + n_train) for k in range(5)])
    test_idx = np.concatenate([np.arange(k * n + n_train, (k + 1) * n) for k in range(5)])
    train_label, test_label = kind_ids[train_idx], kind_ids[test_idx]

    # ---- recognition: per-window correct ----
    def recog_correct(h_tr, h_te):
        _, pred = softmax_predict(h_te, *softmax_regression(h_tr, train_label, n_iter=2000))
        return (pred == test_label).astype(int)
    recog_pre = recog_correct(tr_h, te_h)
    recog_rnd = recog_correct(tr_h_r, te_h_r)

    # ---- experts / oracle / feature ----
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    err = np.zeros((len(test_idx), len(experts)))
    for j, expert in enumerate(experts):
        for i in range(len(test_idx)):
            p = expert.predict(contexts[test_idx[i]], 16)
            err[i, j] = float(np.mean((p - futures[test_idx[i]]) ** 2))
    oracle = err.argmin(axis=1)
    oracle_mse = err.min(axis=1)

    # ---- routing: train on 3 clean kinds, evaluate hard windows ----
    clean = train_label < 3
    hard = np.isin(test_label, [3, 4])
    hard_idx = np.where(hard)[0]
    hard_err = err[hard]
    hard_oracle = oracle[hard]
    hard_oracle_mse = oracle_mse[hard]

    def route(h_tr, h_te):
        _, p = softmax_predict(h_te[hard], *softmax_regression(h_tr[clean], train_label[clean], n_iter=2000))
        return p  # per-hard-window predicted expert (0..2)

    pred_pre = route(tr_h, te_h)
    pred_rnd = route(tr_h_r, te_h_r)

    feats = np.stack([temporal_features(x) for x in contexts[train_idx]])
    te_feats = np.stack([temporal_features(x) for x in contexts[test_idx[hard]]])
    _, pred_feat = softmax_predict(te_feats, *softmax_regression(feats[clean], train_label[clean], n_iter=2000))

    out = {
        "recog_pre": recog_pre, "recog_rnd": recog_rnd,
        "route_pre_acc": (pred_pre == hard_oracle).astype(int),
        "route_rnd_acc": (pred_rnd == hard_oracle).astype(int),
        "route_feat_acc": (pred_feat == hard_oracle).astype(int),
        "route_pre_mse": hard_err[np.arange(len(pred_pre)), pred_pre],
        "route_rnd_mse": hard_err[np.arange(len(pred_rnd)), pred_rnd],
        "route_feat_mse": hard_err[np.arange(len(pred_feat)), pred_feat],
        "oracle_mse_hard": hard_oracle_mse,
    }
    return out


def bootstrap_ci(values, n_boot=2000, seed=0):
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)])
    return {"n": int(len(vals)), "mean": float(np.mean(vals)),
            "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3_8b")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--gen-json", type=Path, default=None,
                        help="per-window generation MSE json (from iclr_gen_save_windows.py)")
    parser.add_argument("--out", type=Path, default=Path("results/iclr/bootstrap_window/qwen3_8b.json"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    acc = {k: [] for k in ["recog_pre", "recog_rnd", "route_pre_acc", "route_rnd_acc",
                           "route_feat_acc", "route_pre_mse", "route_rnd_mse",
                           "route_feat_mse", "oracle_mse_hard"]}
    for seed in seeds:
        try:
            o = per_window_outcomes(args.model, seed)
        except Exception as e:  # noqa: BLE001
            print(f"[seed {seed}] skipped: {e}")
            continue
        for k in acc:
            acc[k].append(o[k])

    report = {"model": args.model, "seeds": seeds,
              "resampling_unit": "individual window (pooled across seeds)", "n_boot": 2000,
              "recognition": {}, "routing": {}, "generation": {}}

    # recognition (pool over all test windows)
    report["recognition"]["pretrained"] = bootstrap_ci(np.concatenate(acc["recog_pre"]))
    report["recognition"]["random"] = bootstrap_ci(np.concatenate(acc["recog_rnd"]))

    # routing (pool over hard windows)
    for k, name in [("route_pre_acc", "pretrained"), ("route_rnd_acc", "random"),
                    ("route_feat_acc", "feature")]:
        report["routing"][f"{name}_acc"] = bootstrap_ci(np.concatenate(acc[k]))
    for k, name in [("route_pre_mse", "pretrained"), ("route_rnd_mse", "random"),
                    ("route_feat_mse", "feature")]:
        report["routing"][f"{name}_mse"] = bootstrap_ci(np.concatenate(acc[k]))
    report["routing"]["oracle_mse"] = bootstrap_ci(np.concatenate(acc["oracle_mse_hard"]))

    # generation: window-level bootstrap of the RATIO OF SUMS (matches the
    # paper's mean(mse)/mean(oracle) definition); per-window pairs from json
    if args.gen_json is not None and args.gen_json.is_file():
        g = json.loads(args.gen_json.read_text())
        pairs = []
        for seed in seeds:
            per = g.get(str(seed), g.get(seed))
            if per:
                pairs.extend(per["pairs"])
        if pairs:
            pairs = np.asarray(pairs, dtype=float)
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(pairs), size=(2000, len(pairs)))
            s = pairs[idx]  # (2000, n, 2)
            sums = s.sum(axis=1)  # (2000, 2)
            ratios = sums[:, 0] / (sums[:, 1] + 1e-12)
            report["generation"]["mse_over_oracle_ratio"] = {
                "n_parsed_windows": int(len(pairs)),
                "point_estimate": float(pairs[:, 0].sum() / (pairs[:, 1].sum() + 1e-12)),
                "ci_low": float(np.percentile(ratios, 2.5)),
                "ci_high": float(np.percentile(ratios, 97.5)),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
