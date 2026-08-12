"""Patch-level dynamic temporal expert routing with the frozen LLM probe (GPU).

The CPU experiment (`run_patch_routing.py`) showed that dynamic routing has a
large oracle headroom (per-segment oracle MSE far below whole-window oracle)
but that hand features fail to route short 32-step patches.  This script tests
whether the frozen pretrained LLM probe recognizes patch-level dynamics better:
it trains a patch router on 32-step numeric-text patches and routes each
horizon segment independently, compared against whole-window LLM routing,
the feature patch router, and per-segment / whole-window oracles.
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
from data.dynamics import build_labeled_windows, generate_window
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _load_model, extract_last_hidden

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B"
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--seg-len", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache", type=Path, default=Path("results/icassp/patch_routing/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/icassp/patch_routing/llm_summary.json"))
    args = parser.parse_args()

    horizon = args.horizon
    n_segments = horizon // args.seg_len
    assert horizon % args.seg_len == 0
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    args.cache.mkdir(parents=True, exist_ok=True)

    def cached(name: str, fn):
        path = args.cache / f"{name}_seed{args.seed}.npy"
        if path.is_file():
            print(f"[cache] {name}")
            return np.load(path)
        value = fn()
        np.save(path, value)
        return value

    rng = np.random.default_rng(args.seed)

    # ---- training data ----
    train_patch_windows = np.stack(
        [
            generate_window(kind, args.patch_len, args.seg_len, rng)[0]
            for kind in ["trend", "periodic", "local"]
            for _ in range(args.n_train)
        ]
    )
    train_patch_label = np.asarray(
        [k for k in range(3) for _ in range(args.n_train)]
    )
    train_full = build_labeled_windows(
        ["trend", "periodic", "local"], args.context_len, horizon, args.n_train, args.seed
    )

    model, tokenizer = _load_model(args.model_path, args.device, False)
    train_patch_h = cached(
        "train_patch_h",
        lambda: extract_last_hidden(model, tokenizer, train_patch_windows, args.device),
    )
    train_full_h = cached(
        "train_full_h",
        lambda: extract_last_hidden(model, tokenizer, train_full[0], args.device),
    )
    full_router_w = softmax_regression(train_full_h, train_full[2], n_iter=800)
    patch_router_w = softmax_regression(train_patch_h, train_patch_label, n_iter=800)
    patch_feats = np.stack([temporal_features(x) for x in train_patch_windows])
    feat_router_w = softmax_regression(patch_feats, train_patch_label, n_iter=800)
    torch.cuda.empty_cache()

    # ---- test windows ----
    test = build_labeled_windows(KINDS, args.context_len, horizon, args.n_test, args.seed + 11)
    contexts, futures, kind_ids = test
    full_series = np.concatenate([contexts, futures], axis=1)
    test_full_h = cached(
        "test_full_h",
        lambda: extract_last_hidden(model, tokenizer, contexts, args.device),
    )
    seg_windows = np.stack(
        [
            full_series[i, args.context_len + s * args.seg_len - args.patch_len
                        : args.context_len + s * args.seg_len]
            for i in range(len(contexts))
            for s in range(n_segments)
        ]
    )
    test_seg_h = cached(
        "test_seg_h",
        lambda: extract_last_hidden(model, tokenizer, seg_windows, args.device),
    )
    del model
    torch.cuda.empty_cache()

    # whole-window LLM routing
    _, whole_pred = softmax_predict(test_full_h, *full_router_w)
    # patch LLM routing
    _, seg_pred = softmax_predict(test_seg_h, *patch_router_w)
    seg_pred = seg_pred.reshape(len(contexts), n_segments)

    # feature patch router for comparison
    _, fseg = softmax_predict(
        np.stack([[temporal_features(seg_windows[i * n_segments + s]) for s in range(n_segments)] for i in range(len(contexts))]).reshape(-1, 9),
        *feat_router_w,
    )
    fseg = fseg.reshape(len(contexts), n_segments)

    # per-segment expert errors and oracle
    seg_err = np.zeros((len(contexts), n_segments, len(experts)))
    for i in range(len(contexts)):
        for s in range(n_segments):
            ctx = full_series[i, args.context_len + s * args.seg_len - args.patch_len
                              : args.context_len + s * args.seg_len]
            target = futures[i, s * args.seg_len : (s + 1) * args.seg_len]
            for j, expert in enumerate(experts):
                seg_err[i, s, j] = float(np.mean((expert.predict(ctx, args.seg_len) - target) ** 2))
    seg_oracle = seg_err.argmin(axis=2)

    def build_forecast(decisions, per_segment=False):
        out = np.zeros((len(contexts), horizon))
        for i in range(len(contexts)):
            if per_segment:
                for s in range(n_segments):
                    ctx = full_series[i, args.context_len + s * args.seg_len - args.patch_len
                                      : args.context_len + s * args.seg_len]
                    out[i, s * args.seg_len : (s + 1) * args.seg_len] = experts[decisions[i, s]].predict(ctx, args.seg_len)
            else:
                out[i] = experts[decisions[i]].predict(contexts[i], horizon)
        return out

    def mse(f):
        return float(np.mean((f - futures) ** 2))

    report = {
        "whole_llm_mse": mse(build_forecast(whole_pred)),
        "patch_llm_mse": mse(build_forecast(seg_pred, per_segment=True)),
        "patch_feature_mse": mse(build_forecast(fseg, per_segment=True)),
        "whole_oracle_mse": mse(build_forecast(seg_err.mean(axis=1).argmin(axis=1))),
        "patch_oracle_mse": mse(build_forecast(seg_oracle, per_segment=True)),
        "best_single_mse": mse(
            np.stack([experts[seg_err.mean(axis=(0, 1)).argmin()].predict(contexts[i], horizon) for i in range(len(contexts))])
        ),
        "routing_acc_vs_seg_oracle": {
            "whole_llm": accuracy(np.repeat(whole_pred[:, None], n_segments, axis=1).ravel(), seg_oracle.ravel()),
            "patch_llm": accuracy(seg_pred.ravel(), seg_oracle.ravel()),
            "patch_feature": accuracy(fseg.ravel(), seg_oracle.ravel()),
        },
        "per_kind": {},
    }
    for k, name in enumerate(KINDS):
        mask = kind_ids == k
        report["per_kind"][name] = {
            "whole_llm_mse": float(np.mean((build_forecast(whole_pred)[mask] - futures[mask]) ** 2)),
            "patch_llm_mse": float(np.mean((build_forecast(seg_pred, True)[mask] - futures[mask]) ** 2)),
            "patch_oracle_mse": float(np.mean((build_forecast(seg_oracle, True)[mask] - futures[mask]) ** 2)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
