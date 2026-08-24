"""ICLR P0.2: robustness of the frozen-generation failure.

Attacks the claim "frozen generation is fragile" from every prompt/decoding
angle a reviewer could try:
  - greedy vs temperature sampling (0.7, 1.0)
  - few-shot in-context demonstrations (1, 2 shots)
  - explicit instruction ("only output numbers")
On the SAME stratified windows used by E3, reporting parse rate and MSE
relative to the oracle expert for frozen Qwen3-0.6B and 8B.

If the recognition/generation gap persists across all of these, the
"don't generate" conclusion is robust to prompt/decoding choices.
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

from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _format_values, _load_model

KINDS = ["trend", "periodic", "local", "mixture", "regime"]

BASE = ("Forecast the next values of this normalized time series.\n"
        "History: {hist}\nForecast:")
INSTRUCT = ("You are forecasting a normalized time series. Output exactly the "
            "next {H} numeric values, space-separated, two decimals, no "
            "explanation.\nHistory: {hist}\nForecast:")
FEWSHOT_EX = ("Example: History: {hist}\nForecast: {fut}\n")


def build_prompts(contexts, futures, mode: str, horizon: int) -> list[str]:
    prompts = []
    for i, row in enumerate(contexts):
        hist = _format_values(row)
        if mode == "instruct":
            prompts.append(INSTRUCT.format(H=horizon, hist=hist))
        elif mode.startswith("fewshot"):
            n_shots = int(mode.replace("fewshot", "") or 1)
            ex = ""
            for k in range(n_shots):
                # demonstrations from clean trend windows far from query i
                j = (i * 7 + k * 13) % len(contexts)
                ex += FEWSHOT_EX.format(hist=_format_values(contexts[j]), fut=_format_values(futures[j][:horizon]))
            prompts.append(ex + BASE.format(hist=hist))
        else:
            prompts.append(BASE.format(hist=hist))
    return prompts


@torch.no_grad()
def generate(model, tokenizer, prompts, device, horizon, mode, batch_size=8,
             max_new_tokens=200, temp=1.0, top_p=1.0) -> dict:
    forecasts, valid = [], []
    do_sample = mode not in ("greedy",)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temp, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, **gen_kwargs)
        ilen = enc["input_ids"].size(1)
        for r in range(gen.size(0)):
            text = tokenizer.decode(gen[r, ilen:], skip_special_tokens=True)
            nums = [float(m) for m in re.findall(r"[+-]?\d\.\d\d", text)]
            forecasts.append(nums[:horizon])
            valid.append(len(nums[:horizon]) == horizon)
    return forecasts, np.asarray(valid, dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/9950backfile/chenjiahui/hf_cache/Qwen3-0.6B")
    parser.add_argument("--model-tag", default="qwen3_0.6b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=150)
    parser.add_argument("--n-train-per-kind", type=int, default=90)
    parser.add_argument("--max-gen-windows", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/iclr/gen_robustness/qwen3_0.6b/summary.json"))
    args = parser.parse_args()

    contexts, futures, _ = build_labeled_windows(KINDS, args.context_len, args.horizon, args.n_per_kind, args.seed)
    n = args.n_per_kind
    n_train = args.n_train_per_kind
    n_test = n - n_train
    # stratify across all 5 kinds: first 8 TEST windows of each kind (matches
    # run_frozen_probe.py, so numbers are directly comparable with E3/family-scale)
    test_idx = np.concatenate(
        [np.arange(k * n + n_train, (k + 1) * n) for k in range(5)]
    )
    per_kind_gen = args.max_gen_windows // len(KINDS)
    gen_idx = np.concatenate(
        [np.arange(k * n_test, k * n_test + per_kind_gen) for k in range(5)]
    )
    ctx = contexts[test_idx[gen_idx]]
    fut = futures[test_idx[gen_idx]]

    # oracle expert error on these windows
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    err = np.zeros((len(ctx), len(experts)))
    for j, e in enumerate(experts):
        for i in range(len(ctx)):
            p = e.predict(ctx[i], args.horizon)
            err[i, j] = float(np.mean((p - fut[i]) ** 2))
    oracle_mse = float(np.mean(err.min(axis=1)))

    model, tokenizer = _load_model(args.model_path, args.device, False)

    configs = [("greedy", 1.0), ("temp0.7", 0.7), ("temp1.0", 1.0)]
    report = {"model": args.model_tag, "windows": int(len(ctx)), "oracle_mse": oracle_mse, "configs": {}}
    for mode, temp in configs + [("instruct", 1.0), ("fewshot1", 1.0), ("fewshot2", 1.0)]:
        if mode.startswith("temp"):
            prompts = build_prompts(ctx, fut, "greedy", args.horizon)
        else:
            prompts = build_prompts(ctx, fut, mode, args.horizon)
        forecasts, valid = generate(model, tokenizer, prompts, args.device, args.horizon, mode, temp=temp)
        mses = []
        for i in range(len(forecasts)):
            if valid[i]:
                mses.append(float(np.mean((np.asarray(forecasts[i]) - fut[i]) ** 2)))
        row = {
            "parse_rate": float(valid.mean()),
            "mse_on_complete": float(np.mean(mses)) if mses else None,
            "oracle_ratio": (float(np.mean(mses)) / oracle_mse) if mses else None,
        }
        report["configs"][mode] = row
        print(f"[{mode}] parse={row['parse_rate']:.3f} mse={row['mse_on_complete']} oracle_ratio={row['oracle_ratio']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
