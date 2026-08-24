"""ICLR: asymmetry sweep - recognition vs generation under task difficulty.

Varies context length, noise level, and horizon for the SAME frozen LLM and the
same labeled windows, and measures:
  - recognition: linear-probe accuracy on frozen hidden states (5 kinds)
  - generation: greedy frozen numeric generation parse rate and MSE on the same
    windows
  - oracle expert MSE (best of trend/periodic/local on those windows)

The point: recognition stays robust as dynamics get harder while generation
collapses - the "recognize, don't generate" asymmetry persists across the
difficulty axis (not just one operating point).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import generate_window
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _format_values, _load_model, frozen_generation

KINDS5 = ["trend", "periodic", "local", "mixture", "regime"]


def build_windows_noise(
    kinds: list[str], context_len: int, horizon: int, n_per_kind: int,
    seed: int, noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    contexts, futures, kind_ids = [], [], []
    for kind_id, kind in enumerate(kinds):
        for _ in range(n_per_kind):
            c, f = generate_window(kind, context_len, horizon, rng, params={"noise": noise})
            contexts.append(c)
            futures.append(f)
            kind_ids.append(kind_id)
    return np.stack(contexts), np.stack(futures), np.asarray(kind_ids)


def run_config(model, tokenizer, device, context_len, horizon, n_per_kind,
               n_train, seed, noise, softmax_iters) -> dict:
    contexts, futures, kind_ids = build_windows_noise(
        KINDS5, context_len, horizon, n_per_kind, seed, noise
    )
    n = n_per_kind
    train_idx = np.concatenate([np.arange(k * n, k * n + n_train) for k in range(5)])
    test_idx = np.concatenate([np.arange(k * n + n_train, (k + 1) * n) for k in range(5)])
    train_label, test_label = kind_ids[train_idx], kind_ids[test_idx]

    # hidden states (frozen, last layer)
    from scripts.run_frozen_probe import extract_last_hidden
    tr_h = extract_last_hidden(model, tokenizer, contexts[train_idx], device)
    te_h = extract_last_hidden(model, tokenizer, contexts[test_idx], device)
    _, pred = softmax_predict(te_h, *softmax_regression(tr_h, train_label, n_iter=softmax_iters))
    recog_acc = float(accuracy(pred, test_label))

    # expert errors (oracle) on test windows
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    err = np.zeros((len(test_idx), len(experts)))
    for j, expert in enumerate(experts):
        for i in range(len(test_idx)):
            p = expert.predict(contexts[test_idx[i]], horizon)
            err[i, j] = float(np.mean((p - futures[test_idx[i]]) ** 2))
    oracle_mse = float(np.mean(err.min(axis=1)))

    # frozen generation on a stratified subset
    per_kind_gen = min(8, n - n_train)
    gen_idx = np.concatenate(
        [np.arange(k * (n - n_train), k * (n - n_train) + per_kind_gen) for k in range(5)]
    )
    gen_contexts = contexts[test_idx[gen_idx]]
    gen_futures = futures[test_idx[gen_idx]]
    gen = frozen_generation(model, tokenizer, gen_contexts, gen_futures, horizon, device)
    return {
        "context_len": context_len, "horizon": horizon, "noise": noise,
        "n_per_kind": n_per_kind, "seed": seed,
        "recognition_acc": recog_acc,
        "parse_rate": gen["parse_rate"],
        "gen_mse_on_complete": gen["mse_on_complete"],
        "oracle_mse": oracle_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/9950backfile/chenjiahui/hf_cache/Qwen3-0.6B")
    parser.add_argument("--model-tag", default="qwen3_0.6b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-per-kind", type=int, default=60)
    parser.add_argument("--n-train-per-kind", type=int, default=36)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--softmax-iters", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("results/iclr/asymmetry/summary.json"))
    args = parser.parse_args()

    model, tokenizer = _load_model(args.model_path, args.device, False)

    # context sweep
    results = {"model": args.model_tag, "seed": args.seed, "configs": []}
    for context_len in [32, 64, 128]:
        results["configs"].append(run_config(
            model, tokenizer, args.device, context_len, 16, args.n_per_kind,
            args.n_train_per_kind, args.seed, 0.05, args.softmax_iters))
        print(f"[ctx={context_len}] {results['configs'][-1]}")
    # noise sweep
    for noise in [0.02, 0.05, 0.1, 0.2]:
        results["configs"].append(run_config(
            model, tokenizer, args.device, 64, 16, args.n_per_kind,
            args.n_train_per_kind, args.seed, noise, args.softmax_iters))
        print(f"[noise={noise}] {results['configs'][-1]}")
    # horizon sweep
    for horizon in [8, 16, 32]:
        results["configs"].append(run_config(
            model, tokenizer, args.device, 64, horizon, args.n_per_kind,
            args.n_train_per_kind, args.seed, 0.05, args.softmax_iters))
        print(f"[horizon={horizon}] {results['configs'][-1]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
