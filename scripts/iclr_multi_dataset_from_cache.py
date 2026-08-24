"""ICLR: compute real-data router metrics from cached hidden states only.

Reads the windows/hidden caches produced by iclr_multi_dataset_router.py (no
model load) and computes feature / frozen-random / frozen-pretrained routers
against the three univariate experts.  Used to finish runs whose model-loading
phase hung, and for cheap per-seed aggregation.
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
    parser.add_argument("--run-dir", type=Path,
                        default=Path("results/iclr/multi_dataset_router/qwen3_8b"))
    parser.add_argument("--model-tag", default="qwen3_8b")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=150)
    parser.add_argument("--n-train-per-kind", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-cache", type=Path, default=Path("results/icassp/probe/cache"))
    args = parser.parse_args()

    windows_dir = args.run_dir / f"windows_{args.model_tag}_s{args.seed}"
    hidden_dir = args.run_dir / f"hidden_{args.model_tag}_s{args.seed}"

    def cache(path: Path, stem: str) -> Path:
        seeded = path / f"{stem}_seed{args.seed}.npy"
        if seeded.is_file():
            return seeded
        legacy = path / f"{stem}.npy"
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(f"missing {stem} in {path}")

    train_h = np.load(cache(args.probe_cache, "train_h_pretrained"))
    train_h_r = np.load(cache(args.probe_cache, "train_h_random"))
    c, _, kk = build_labeled_windows(KINDS, args.context_len, args.horizon, args.n_per_kind, args.seed)
    tr = np.concatenate([np.arange(k * args.n_per_kind, k * args.n_per_kind + args.n_train_per_kind) for k in range(3)])
    clean_label = kk[tr]
    syn_feats = np.stack([temporal_features(x) for x in c[tr]])

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    report = {"model": args.model_tag, "seed": args.seed, "datasets": {}}

    for wpath in sorted(windows_dir.glob("*_windows.npz")):
        dname = wpath.name.replace("_windows.npz", "")
        z = np.load(wpath)
        contexts, futures = z["ctx"], z["fut"]
        h_p = np.load(hidden_dir / f"{dname}_pretrained.npy")
        h_r = np.load(hidden_dir / f"{dname}_random.npy")

        errors = np.zeros((len(contexts), len(experts)))
        for j, expert in enumerate(experts):
            for i in range(len(contexts)):
                pred = expert.predict(contexts[i], args.horizon)
                errors[i, j] = float(np.mean((pred - futures[i]) ** 2))
        oracle = errors.argmin(axis=1)
        best_single = int(errors.mean(axis=0).argmin())

        et_feats = np.stack([temporal_features(x) for x in contexts])
        _, fpred = softmax_predict(et_feats, *softmax_regression(syn_feats, clean_label))
        _, ppred = softmax_predict(h_p, *softmax_regression(train_h[tr], clean_label))
        _, rpred = softmax_predict(h_r, *softmax_regression(train_h_r[tr], clean_label))

        report["datasets"][dname] = {
            "windows": len(contexts),
            "oracle_mse": float(np.mean(errors.min(axis=1))),
            "best_single_mse": float(np.mean(errors[:, best_single])),
            "best_single_name": experts[best_single].name,
            "feature_router": {"acc_vs_oracle": accuracy(fpred, oracle),
                               "mse": float(np.mean(errors[np.arange(len(fpred)), fpred])),
                               "pred_dist": [int((fpred == k).sum()) for k in range(3)]},
            "probe_pretrained": {"acc_vs_oracle": accuracy(ppred, oracle),
                                 "mse": float(np.mean(errors[np.arange(len(ppred)), ppred])),
                                 "pred_dist": [int((ppred == k).sum()) for k in range(3)]},
            "probe_random": {"acc_vs_oracle": accuracy(rpred, oracle),
                             "mse": float(np.mean(errors[np.arange(len(rpred)), rpred])),
                             "pred_dist": [int((rpred == k).sum()) for k in range(3)]},
        }
        print(f"[{dname}] feat={report['datasets'][dname]['feature_router']['acc_vs_oracle']:.3f} "
              f"llm={report['datasets'][dname]['probe_pretrained']['acc_vs_oracle']:.3f} "
              f"rand={report['datasets'][dname]['probe_random']['acc_vs_oracle']:.3f}")

    out = args.run_dir / "summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
