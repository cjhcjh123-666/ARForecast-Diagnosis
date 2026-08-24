"""ICLR P1.4: zero-shot routing at multiple horizons on real data.

The frozen hidden states depend only on the context window, so we resample
windows with the target horizon, extract hidden states once per (dataset,
horizon), and evaluate the zero-shot router (trained on synthetic clean
dynamics) at horizon 24 and 48 for a representative set of real datasets.
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

from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.iclr_multi_dataset_router import DATASETS, sample_windows
from scripts.run_frozen_probe import _load_model, extract_last_hidden

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B")
    parser.add_argument("--model-tag", default="qwen3_8b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizons", default="24,48")
    parser.add_argument("--max-windows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-cache", type=Path, default=Path("results/icassp/probe/cache"))
    parser.add_argument("--datasets", default="ettm1,etth1,electricity,weather,exchange_rate")
    parser.add_argument("--out", type=Path, default=Path("results/iclr/multi_horizon/qwen3_8b/summary.json"))
    args = parser.parse_args()

    def cache(path: Path, stem: str) -> Path:
        seeded = path / f"{stem}_seed{args.seed}.npy"
        if seeded.is_file():
            return seeded
        legacy = path / f"{stem}.npy"
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(f"missing {stem} in {path}")

    train_h = np.load(cache(args.probe_cache, "train_h_pretrained"))
    c, _, kk = build_labeled_windows(KINDS, args.context_len, 16, 150, args.seed)
    tr = np.concatenate([np.arange(k * 150, k * 150 + 90) for k in range(3)])
    clean_label = kk[tr]
    syn_feats = np.stack([temporal_features(x) for x in c[tr]])

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    report = {"model": args.model_tag, "seed": args.seed, "datasets": {}}

    horizons = [int(h) for h in args.horizons.split(",")]
    max_h = max(horizons)
    for dname in [d for d in args.datasets.split(",") if d]:
        report["datasets"][dname] = {}
        # sample windows once at the max horizon; hidden states depend only on
        # the 64-step context, so they are shared across horizons
        contexts, futures = sample_windows(dname, args.context_len, max_h, args.max_windows, args.seed)
        model, tokenizer = _load_model(args.model_path, args.device, False)
        h = extract_last_hidden(model, tokenizer, contexts, args.device, batch_size=32)
        del model
        torch.cuda.empty_cache()
        et_feats = np.stack([temporal_features(x) for x in contexts])
        for horizon in horizons:
            future_h = futures[:, :horizon]
            err = np.zeros((len(contexts), len(experts)))
            for j, expert in enumerate(experts):
                for i in range(len(contexts)):
                    p = expert.predict(contexts[i], horizon)
                    err[i, j] = float(np.mean((p - future_h[i]) ** 2))
            oracle = err.argmin(axis=1)
            best_single = int(err.mean(axis=0).argmin())

            _, fpred = softmax_predict(et_feats, *softmax_regression(syn_feats, clean_label))
            _, ppred = softmax_predict(h, *softmax_regression(train_h[tr], clean_label))

            report["datasets"][dname][str(horizon)] = {
                "oracle_mse": float(np.mean(err.min(axis=1))),
                "best_single_mse": float(np.mean(err[:, best_single])),
                "best_single_name": experts[best_single].name,
                "feature_router": {"acc": accuracy(fpred, oracle),
                                   "mse": float(np.mean(err[np.arange(len(fpred)), fpred]))},
                "llm_router": {"acc": accuracy(ppred, oracle),
                               "mse": float(np.mean(err[np.arange(len(ppred)), ppred]))},
            }
            r = report["datasets"][dname][str(horizon)]
            print(f"[{dname} H={horizon}] feat={r['feature_router']['acc']:.3f}/{r['feature_router']['mse']:.3f} "
                  f"llm={r['llm_router']['acc']:.3f}/{r['llm_router']['mse']:.3f} oracle={r['oracle_mse']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
