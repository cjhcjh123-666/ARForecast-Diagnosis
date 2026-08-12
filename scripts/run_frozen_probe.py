"""Frozen-LLM temporal-structure recognition probe and zero-shot expert router.

Evidence for the "recognize, don't generate" paper:

1. A frozen Qwen3-8B (no tuning) reads numeric-text history and its final-token
   hidden state linearly separates temporal dynamics kinds (trend / periodic /
   local / mixture / regime) far above a random-initialized same-architecture
   control, and at least as well as a hand-feature logistic floor.
2. Leave-one-kind-out: the frozen probe generalizes to dynamics it never saw
   during probe training, where the feature floor degrades.
3. On the *same windows*, frozen greedy numeric generation is far less reliable
   than recognition: low parse coverage and high error, while the recognition
   probe stays accurate.  This is the paired recognize-vs-generate claim.
4. The probe acts as a zero-shot expert router: routing accuracy vs the oracle
   and downstream forecast MSE vs best-single-expert / feature router.
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

from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def _reinit_model(model: torch.nn.Module) -> None:
    """In-place re-initialization: random weights, same architecture/tokenizer."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.reset_parameters()


def _load_model(
    model_path: str,
    device: str,
    random_init: bool,
    random_init_mode: str = "reinit",
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if random_init and random_init_mode == "from_config":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    else:
        device_index = int(device.split(":")[1]) if ":" in device else 0
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": device_index},
            **model_kwargs,
        )
    if random_init and random_init_mode == "reinit":
        _reinit_model(model)
    model.eval()
    return model, tokenizer


def _format_values(values: np.ndarray) -> str:
    clipped = np.clip(values, -9.99, 9.99)
    return " ".join(f"{float(value):+.2f}" for value in clipped)


def _prompts(contexts: np.ndarray) -> list[str]:
    return [
        "Forecast the next values of this normalized time series.\n"
        "History: " + _format_values(row) + "\nForecast:"
        for row in contexts
    ]


def _cache_path(cache_dir: Path, stem: str, seed: int) -> Path:
    """Seed-aware cache path with fallback to the legacy (seed-less) name."""
    seeded = cache_dir / f"{stem}_seed{seed}.npy"
    if seeded.is_file():
        return seeded
    legacy = cache_dir / f"{stem}.npy"
    if legacy.is_file():
        return legacy
    return seeded


def _save_results(args: argparse.Namespace, results: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


@torch.no_grad()
def extract_last_hidden(
    model, tokenizer, contexts: np.ndarray, device: str, batch_size: int = 32
) -> np.ndarray:
    """Final-token hidden state of the last layer for every context window."""
    prompts = _prompts(contexts)
    vectors = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(
            chunk, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        out = model(**encoded, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        row_idx = torch.arange(len(chunk), device=device)
        vectors.append(hidden[row_idx, lengths].float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


@torch.no_grad()
def frozen_generation(
    model,
    tokenizer,
    contexts: np.ndarray,
    futures: np.ndarray,
    horizon: int,
    device: str,
    max_new_tokens: int = 160,
    batch_size: int = 8,
) -> dict:
    """Greedy frozen generation of numeric forecasts; returns parse/error stats."""
    prompts = _prompts(contexts)
    forecasts = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(
            chunk, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_length = encoded["input_ids"].size(1)
        for row_index in range(generated.size(0)):
            text = tokenizer.decode(
                generated[row_index, input_length:], skip_special_tokens=True
            )
            numbers = [float(match) for match in re.findall(r"[+-]?\d\.\d\d", text)]
            forecasts.append(numbers[:horizon])
    valid = [len(row) == horizon for row in forecasts]
    errors = []
    for i, ok in enumerate(valid):
        if ok:
            errors.append(float(np.mean((np.asarray(forecasts[i]) - futures[i]) ** 2)))
    return {
        "windows": int(len(contexts)),
        "parse_rate": float(np.mean(valid)),
        "complete_windows": int(np.sum(valid)),
        "mse_on_complete": float(np.mean(errors)) if errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B"
    )
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=150)
    parser.add_argument("--n-train-per-kind", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-gen-windows", type=int, default=40)
    parser.add_argument("--softmax-iters", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/icassp/probe"))
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
    train_feats = np.stack([temporal_features(x) for x in contexts[train_idx]])
    test_feats = np.stack([temporal_features(x) for x in contexts[test_idx]])

    # ---- expert errors for oracle and downstream routing ----
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]

    def expert_errors(idx: np.ndarray) -> np.ndarray:
        err = np.zeros((len(idx), len(experts)))
        for j, expert in enumerate(experts):
            for i in range(len(idx)):
                pred = expert.predict(contexts[idx[i]], args.horizon)
                err[i, j] = float(np.mean((pred - futures[idx[i]]) ** 2))
        return err

    train_err = expert_errors(train_idx)
    test_err = expert_errors(test_idx)
    oracle = test_err.argmin(axis=1)
    best_single_idx = int(train_err.mean(axis=0).argmin())

    results: dict = {
        "kinds": KINDS,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "n_train_per_kind": n_train,
        "n_test_per_kind": n_per_kind - n_train,
    }

    def fit(predictor_feats: np.ndarray, labels: np.ndarray) -> tuple:
        return softmax_regression(predictor_feats, labels, n_iter=args.softmax_iters)

    # ---- hand-feature floor ----
    _, floor_pred = softmax_predict(test_feats, *fit(train_feats, train_label))
    floor_acc = accuracy(floor_pred, test_label)
    results["feature_floor"] = {"accuracy": floor_acc}

    # ---- frozen probes: pretrained vs random same architecture ----
    probe = {}
    pretrained = {}
    for init in ["pretrained", "random"]:
        cache_dir = args.output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_cache = _cache_path(cache_dir, f"train_h_{init}", args.seed)
        test_cache = _cache_path(cache_dir, f"test_h_{init}", args.seed)
        if train_cache.is_file() and test_cache.is_file():
            train_h = np.load(train_cache)
            test_h = np.load(test_cache)
            print(f"[probe] using cached hidden states for {init}")
        else:
            model, tokenizer = _load_model(args.model_path, args.device, init == "random")
            train_h = extract_last_hidden(model, tokenizer, contexts[train_idx], args.device)
            test_h = extract_last_hidden(model, tokenizer, contexts[test_idx], args.device)
            del model
            torch.cuda.empty_cache()
            np.save(train_cache, train_h)
            np.save(test_cache, test_h)

        _, pred = softmax_predict(test_h, *fit(train_h, train_label))
        acc = accuracy(pred, test_label)
        per_kind = {
            KINDS[k]: accuracy(pred[test_label == k], test_label[test_label == k])
            for k in range(len(KINDS))
        }
        probe[init] = {
            "accuracy": acc,
            "per_kind": per_kind,
        }
        if init == "pretrained":
            pretrained = {"train_h": train_h, "test_h": test_h}
    results["probe"] = probe
    _save_results(args, results)

    # ---- paired recognize vs generate on the same windows (frozen pretrained) ----
    # stratify across kinds so the comparison is not dominated by one dynamics
    per_kind_gen = args.max_gen_windows // len(KINDS)
    n_test = n_per_kind - n_train
    gen_windows = np.concatenate(
        [
            np.arange(k * n_test, k * n_test + per_kind_gen)
            for k in range(len(KINDS))
            if per_kind_gen <= n_test
        ]
    )
    gen_contexts = contexts[test_idx[gen_windows]]
    gen_futures = futures[test_idx[gen_windows]]
    gen_label = test_label[gen_windows]
    gen_h = pretrained["test_h"][gen_windows]
    _, gen_pred = softmax_predict(
        gen_h, *fit(pretrained["train_h"], train_label)
    )
    gen_acc = accuracy(gen_pred, gen_label)

    model, tokenizer = _load_model(args.model_path, args.device, False)
    gen_stats = frozen_generation(
        model, tokenizer, gen_contexts, gen_futures, args.horizon, args.device
    )
    del model
    torch.cuda.empty_cache()
    gen_oracle_mse = float(np.mean(test_err[gen_windows].min(axis=1)))
    gen_best_single_mse = float(np.mean(test_err[gen_windows][:, best_single_idx]))
    results["paired_recognize_vs_generate"] = {
        "windows": int(len(gen_windows)),
        "recognize_accuracy": gen_acc,
        **gen_stats,
        "oracle_mse": gen_oracle_mse,
        "best_single_expert_mse": gen_best_single_mse,
        "best_single_expert": experts[best_single_idx].name,
    }
    _save_results(args, results)

    # ---- zero-shot expert router on unseen mixture/regime ----
    hard = np.isin(test_label, [3, 4])
    hard_idx = np.where(hard)[0]
    clean = train_label < 3
    router = {}
    probe_h = {
        "pretrained": (pretrained["train_h"], pretrained["test_h"]),
        "random": (
            np.load(_cache_path(args.output_dir / "cache", "train_h_random", args.seed)),
            np.load(_cache_path(args.output_dir / "cache", "test_h_random", args.seed)),
        ),
    }
    for name, tr_feats, te_feats in [
        ("feature", train_feats, test_feats),
        ("probe_pretrained", *probe_h["pretrained"]),
        ("probe_random", *probe_h["random"]),
    ]:
        _, pred = softmax_predict(
            te_feats[hard], *fit(tr_feats[clean], train_label[clean])
        )
        prob, _ = softmax_predict(
            te_feats[hard], *fit(tr_feats[clean], train_label[clean])
        )
        # soft routing: probability-weighted average of expert forecasts
        soft_pred = np.zeros((hard_idx.size, args.horizon))
        for j, expert in enumerate(experts):
            for i, w_idx in enumerate(hard_idx):
                soft_pred[i] += prob[i, j] * expert.predict(
                    contexts[test_idx[w_idx]], args.horizon
                )
        soft_targets = futures[test_idx[hard_idx]]
        soft_mse = float(np.mean((soft_pred - soft_targets) ** 2))
        router[name] = {
            "routing_acc_vs_oracle": accuracy(pred, oracle[hard]),
            "mse": float(np.mean(test_err[hard_idx, pred])),
            "soft_routing_mse": soft_mse,
        }
    router["oracle"] = {"mse": float(np.mean(test_err[hard].min(axis=1)))}
    router["best_single_expert"] = {
        "mse": float(np.mean(test_err[hard][:, best_single_idx])),
        "name": experts[best_single_idx].name,
    }
    router["uniform_ensemble"] = {"mse": float(np.mean(test_err[hard].mean(axis=1)))}
    results["zero_shot_router"] = router

    _save_results(args, results)
    print(json.dumps(results, indent=2))
    print(f"saved={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
