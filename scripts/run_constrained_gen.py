"""E3 hardened: constrained-generation control for 'recognize vs generate'.

The original E3 showed frozen generation parses only ~27% of windows, but a
reviewer may argue the failure is just poor output *formatting*.  This script
removes the formatting confound by masking the logits at every step so that
only numeric-format tokens are allowed (parse rate ~100%), then re-measures
forecast error on the same windows.  It also re-measures recognition accuracy
and the oracle expert error on the same windows, with bootstrap 95% CIs.
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

from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _load_model, extract_last_hidden

KINDS = ["trend", "periodic", "local", "mixture", "regime"]
NUMERIC_CHARS = set("0123456789+-. ")


def parse_numbers(text: str, horizon: int) -> list[float]:
    return [float(m) for m in re.findall(r"[+-]?\d\.\d\d", text)][:horizon]


def numeric_token_mask(tokenizer) -> torch.Tensor:
    """Ids of tokens that decode to numeric-format characters only."""
    vocab = tokenizer.get_vocab()
    ids = []
    for token, tid in vocab.items():
        if not token:
            continue
        if token.startswith("<") and token.endswith(">"):
            continue
        if all(c in NUMERIC_CHARS for c in token):
            ids.append(tid)
    return torch.tensor(sorted(ids), dtype=torch.long)


def _expected_char(mod: int) -> str:
    """Expected char class within a 6-char number group: ' +d.dd'."""
    if mod == 0:
        return " "
    if mod == 1:
        return "+-"
    if mod in (2, 4, 5):
        return "0123456789"
    return "."


def _allowed_by_position(tokenizer, horizon: int) -> list[list[int]]:
    """Precompute, for each of the 6 positions, the token ids whose (normalized)
    characters all match the expected '+d.dd ' grammar starting there."""
    vocab = tokenizer.get_vocab()
    tokens = list(vocab.items())
    allowed = [[] for _ in range(6)]
    for tok, tid in tokens:
        norm = tok.replace("▁", " ")
        if not norm or (norm.startswith("<") and norm.endswith(">")):
            continue
        ok_all = True
        for mod in range(6):
            ok = all(c in _expected_char((mod + i) % 6) for i, c in enumerate(norm))
            if ok:
                allowed[mod].append(tid)
            else:
                ok_all = False
        _ = ok_all
    return [torch.tensor(sorted(lst), dtype=torch.long) for lst in allowed]


@torch.no_grad()
def constrained_generate(
    model, tokenizer, prompts, horizon, device, batch_size=4, max_tokens=120
) -> list[list[float]]:
    """Grammar-constrained decoding enforcing the exact '+d.dd ' format.

    The expected grammar is 16 groups of [space][sign][digit][.][digit][digit],
    so the generated text is always a valid numeric forecast; any remaining
    forecast error cannot be attributed to output formatting.
    """
    allowed = _allowed_by_position(tokenizer, horizon)
    allowed = [a.to(device) for a in allowed]
    gen_chars = [0] * len(prompts)
    outputs = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        seq = enc["input_ids"]
        attn = enc["attention_mask"]
        n_chars = [0] * len(chunk)
        for _ in range(max_tokens):
            logits = model(input_ids=seq, attention_mask=attn).logits[:, -1, :]
            masked = torch.full_like(logits, -float("inf"))
            for row_i in range(len(chunk)):
                if n_chars[row_i] >= horizon * 6:
                    continue
                lst = allowed[n_chars[row_i] % 6]
                masked[row_i, lst] = logits[row_i, lst]
            nxt = masked.argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            attn = torch.cat([attn, torch.ones_like(nxt)], dim=1)
            for row_i, tid in enumerate(nxt[:, 0].tolist()):
                tok = tokenizer.convert_ids_to_tokens(tid)
                n_chars[row_i] += len(tok.replace("▁", " "))
        for row in seq:
            text = tokenizer.decode(row[len(enc["input_ids"][0]) :], skip_special_tokens=True)
            outputs.append(parse_numbers(text, horizon))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B"
    )
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-cache", type=Path, default=Path("results/icassp/probe/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/icassp/constrained_gen/summary.json"))
    args = parser.parse_args()

    contexts, futures, kind_ids = build_labeled_windows(
        KINDS, args.context_len, args.horizon, args.n_per_kind, args.seed
    )
    n_total = len(contexts)
    n_per_kind = args.n_per_kind

    # probe trained on cached seed-7 pretrained hidden states (5 kinds)
    train_h = np.load(args.probe_cache / "train_h_pretrained_seed7.npy")
    c, _, kk = build_labeled_windows(KINDS, args.context_len, args.horizon, 150, 7)
    n_train = 90
    tr = np.concatenate([np.arange(k * n_train, (k + 1) * n_train) for k in range(5)])
    train_labels = np.repeat(np.arange(5), n_train)
    probe_w = softmax_regression(train_h[tr], train_labels, n_iter=800)

    model, tokenizer = _load_model(args.model_path, args.device, False)
    test_h = extract_last_hidden(model, tokenizer, contexts, args.device)
    _, probe_pred = softmax_predict(test_h, *probe_w)
    recog_acc = float(np.mean(probe_pred == kind_ids))
    report = {
        "n_windows": n_total,
        "n_per_kind": n_per_kind,
        "recognition_accuracy": recog_acc,
        "status": "recognition_done",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    prompts = [
        "Forecast the next values of this normalized time series.\n"
        "History: " + " ".join(f"{float(v):+.2f}" for v in row) + "\nForecast:"
        for row in contexts
    ]

    # constrained generation (numeric-only tokens)
    constr = constrained_generate(model, tokenizer, prompts, args.horizon, args.device)
    constr_valid = [len(r) == args.horizon for r in constr]
    constr_mse = float(
        np.mean([(np.asarray(constr[i]) - futures[i]) ** 2 for i in range(n_total) if constr_valid[i]])
    )
    report.update(
        {
            "constrained": {
                "parse_rate": float(np.mean(constr_valid)),
                "mse": constr_mse,
            },
            "status": "constrained_done",
        }
    )
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[save] constrained done", flush=True)

    # unconstrained generation on the same windows
    unconst = []
    unconst_subset = np.arange(0, min(40, n_total))
    for start in range(0, len(unconst_subset), 8):
        idxs = unconst_subset[start : start + 8]
        chunk = [prompts[i] for i in idxs]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        gen = model.generate(
            **enc,
            max_new_tokens=args.horizon * 8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        in_len = enc["input_ids"].size(1)
        for row in gen:
            text = tokenizer.decode(row[in_len:], skip_special_tokens=True)
            unconst.append(parse_numbers(text, args.horizon))
    del model
    torch.cuda.empty_cache()
    unconst_valid = [len(r) == args.horizon for r in unconst]
    unconst_parse = float(np.mean(unconst_valid))
    unconst_mse = float(
        np.mean(
            [
                (np.asarray(unconst[i]) - futures[unconst_subset[i]]) ** 2
                for i in range(len(unconst_subset))
                if unconst_valid[i]
            ]
        )
    )

    # oracle expert MSE on the same windows
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    err = np.zeros((n_total, len(experts)))
    for j, expert in enumerate(experts):
        for i in range(n_total):
            err[i, j] = float(np.mean((expert.predict(contexts[i], args.horizon) - futures[i]) ** 2))
    oracle_mse = float(np.mean(err.min(axis=1)))

    # bootstrap CI
    rng = np.random.default_rng(0)
    def boot(values: np.ndarray, n_boot: int = 500) -> tuple[float, float]:
        means = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    recog_ci = boot((probe_pred == kind_ids).astype(float))
    constr_err_vec = np.asarray(
        [(np.asarray(constr[i]) - futures[i]) ** 2 for i in range(n_total) if constr_valid[i]]
    )
    constr_ci = boot(constr_err_vec) if len(constr_err_vec) else (float("nan"), float("nan"))

    report = {
        "n_windows": n_total,
        "n_per_kind": n_per_kind,
        "recognition_accuracy": recog_acc,
        "recognition_ci95": list(recog_ci),
        "constrained": {
            "parse_rate": float(np.mean(constr_valid)),
            "mse": constr_mse,
            "mse_ci95": list(constr_ci),
        },
        "unconstrained": {
            "parse_rate": unconst_parse,
            "mse": unconst_mse,
            "n_windows": int(len(unconst_subset)),
        },
        "oracle_expert_mse": oracle_mse,
        "status": "done",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
