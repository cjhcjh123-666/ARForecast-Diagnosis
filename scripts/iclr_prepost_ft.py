"""ICLR: pre- vs post-finetuning representation analysis.

Loads the frozen 8B backbone, extracts last-layer hidden states on the labeled
synthetic-dynamics windows; then loads a LoRA adapter saved by E1 (finetuned on
a real forecasting task), extracts hidden states on the SAME windows, and
reports:
  - recognition accuracy (linear probe) before vs after finetuning
  - representation shift: mean abs delta and cosine similarity of the linear
    probe directions, plus raw hidden-state CKA
  - per-kind accuracy before/after

Question: does forecasting finetuning (which makes the LLM a better forecaster)
preserve, erode, or sharpen the temporal-structure recognition representation?
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

from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from scripts.run_frozen_probe import _load_model, extract_last_hidden

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


def cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two feature matrices (n, d)."""
    def center(k):
        k = k - k.mean(axis=0, keepdims=True)
        return k - k.mean(axis=1, keepdims=True) + k.mean()
    xx = center(x @ x.T)
    yy = center(y @ y.T)
    hsic = float((xx * yy).sum())
    n1 = float((xx * xx).sum()) ** 0.5
    n2 = float((yy * yy).sum()) ** 0.5
    return hsic / (n1 * n2 + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B")
    parser.add_argument("--model-tag", default="qwen3_8b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-per-kind", type=int, default=150)
    parser.add_argument("--n-train-per-kind", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--softmax-iters", type=int, default=2000)
    parser.add_argument("--adapter", type=str,
                        default="results/icassp/ettm1/qwen3_8b_lora/seed7/lora_adapter")
    parser.add_argument("--adapter-tag", default="ettm1_lora_seed7")
    parser.add_argument("--out", type=Path, default=Path("results/iclr/prepost_ft/summary.json"))
    args = parser.parse_args()

    contexts, _, kind_ids = build_labeled_windows(
        KINDS, args.context_len, args.horizon, args.n_per_kind, args.seed
    )
    n = args.n_per_kind
    n_train = args.n_train_per_kind
    train_idx = np.concatenate([np.arange(k * n, k * n + n_train) for k in range(5)])
    test_idx = np.concatenate([np.arange(k * n + n_train, (k + 1) * n) for k in range(5)])
    train_label, test_label = kind_ids[train_idx], kind_ids[test_idx]

    # frozen hidden states
    model, tokenizer = _load_model(args.model_path, args.device, False)
    h_frozen = extract_last_hidden(model, tokenizer, contexts, args.device)
    del model
    torch.cuda.empty_cache()

    # post-finetuning hidden states (same base + LoRA adapter)
    from peft import PeftModel
    model2, tokenizer2 = _load_model(args.model_path, args.device, False)
    model2 = PeftModel.from_pretrained(model2, args.adapter, is_trainable=False)
    model2.eval()
    h_ft = extract_last_hidden(model2, tokenizer2, contexts, args.device)
    del model2
    torch.cuda.empty_cache()

    # recognition accuracy pre/post
    def acc(h):
        _, pred = softmax_predict(h[test_idx], *softmax_regression(h[train_idx], train_label, n_iter=args.softmax_iters))
        return float(accuracy(pred, test_label)), pred

    acc_frozen, pred_frozen = acc(h_frozen)
    acc_ft, pred_ft = acc(h_ft)

    # representation shift
    delta = float(np.mean(np.abs(h_ft - h_frozen)))
    cka_val = cka(h_frozen, h_ft)
    # probe direction similarity
    W_frozen = softmax_regression(h_frozen[train_idx], train_label, n_iter=args.softmax_iters)[0]
    W_ft = softmax_regression(h_ft[train_idx], train_label, n_iter=args.softmax_iters)[0]
    Wf, Wt = W_frozen / (np.linalg.norm(W_frozen, axis=1, keepdims=True) + 1e-12), W_ft / (np.linalg.norm(W_ft, axis=1, keepdims=True) + 1e-12)
    probe_cos = float(np.mean(np.diag(Wf @ Wt.T)))

    report = {
        "model": args.model_tag, "adapter": args.adapter_tag, "seed": args.seed,
        "context_len": args.context_len, "horizon": args.horizon,
        "frozen": {"recognition_acc": acc_frozen,
                   "per_kind": {KINDS[k]: float(accuracy(pred_frozen[test_label == k], test_label[test_label == k])) for k in range(5)}},
        "post_ft": {"recognition_acc": acc_ft,
                    "per_kind": {KINDS[k]: float(accuracy(pred_ft[test_label == k], test_label[test_label == k])) for k in range(5)}},
        "representation_shift": {
            "mean_abs_delta": delta,
            "linear_cka": cka_val,
            "probe_direction_cos": probe_cos,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
