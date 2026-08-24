"""Save per-window frozen generation MSE (and oracle MSE) for window bootstrap.

Runs the same stratified 40-window protocol as run_frozen_probe.py and stores
per-parsed-window MSE / oracle-MSE ratios to a JSON used by
iclr_bootstrap_window.py.
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

from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _load_model, frozen_generation

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B")
    parser.add_argument("--model-tag", default="qwen3_8b")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--out", type=Path, default=Path("results/iclr/bootstrap_window/generation_qwen3_8b.json"))
    args = parser.parse_args()

    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    report = {"model": args.model_tag}
    for seed in [int(s) for s in args.seeds.split(",")]:
        contexts, futures, _ = build_labeled_windows(KINDS, 64, 16, 150, seed)
        n_test = 150 - 90
        per_kind_gen = 40 // len(KINDS)
        test_idx = np.concatenate([np.arange(k * 150 + 90, (k + 1) * 150) for k in range(5)])
        gen_idx = np.concatenate([np.arange(k * n_test, k * n_test + per_kind_gen) for k in range(5)])
        ctx = contexts[test_idx[gen_idx]]
        fut = futures[test_idx[gen_idx]]

        err = np.zeros((len(ctx), len(experts)))
        for j, e in enumerate(experts):
            for i in range(len(ctx)):
                p = e.predict(ctx[i], 16)
                err[i, j] = float(np.mean((p - fut[i]) ** 2))
        oracle_mse = err.min(axis=1)

        per = run_generation_per_window(model_tag=args.model_tag, seed=seed,
                                        ctx=ctx, fut=fut, oracle_mse=oracle_mse,
                                        args=args)
        report[str(seed)] = per
        mean_ratio = float(np.mean([p[0] / (p[1] + 1e-12) for p in per["pairs"]])) if per["pairs"] else float("nan")
        print(f"[seed {seed}] parsed={per['n_parsed']}/{len(ctx)} "
              f"mean_ratio={mean_ratio:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.out}")


def run_generation_per_window(model_tag, seed, ctx, fut, oracle_mse, args):
    import re
    import torch
    from scripts.run_frozen_probe import _format_values, _load_model
    prompts = [
        "Forecast the next values of this normalized time series.\n"
        "History: " + _format_values(row) + "\nForecast:" for row in ctx
    ]
    model, tokenizer = _load_model(args.model_path, args.device, False)
    pairs = []  # (mse, oracle_mse) per parsed window
    batch_size = 8
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        gen = model.generate(**enc, max_new_tokens=160, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        ilen = enc["input_ids"].size(1)
        for r in range(gen.size(0)):
            text = tokenizer.decode(gen[r, ilen:], skip_special_tokens=True)
            nums = [float(m) for m in re.findall(r"[+-]?\d\.\d\d", text)][:16]
            i = start + r
            if len(nums) == 16:
                mse = float(np.mean((np.asarray(nums) - fut[i]) ** 2))
                pairs.append([mse, float(oracle_mse[i])])
    del model
    torch.cuda.empty_cache()
    return {"n_parsed": len(pairs), "n_total": int(len(ctx)),
            "pairs": pairs,  # [mse, oracle_mse] per parsed window
            "oracle_mse_all": float(np.mean(oracle_mse))}


if __name__ == "__main__":
    main()
