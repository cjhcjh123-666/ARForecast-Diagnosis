"""ICLR: layer-wise linear probe of temporal-structure recognition.

Extracts frozen hidden states from *every* transformer layer (not just the
last) on the labeled synthetic-dynamics windows, and fits a linear probe per
layer for pretrained vs random same-architecture controls.  Optionally loads a
LoRA adapter (from E1 finetuning) to compare pre-FT vs post-FT recognition per
layer.

This answers: where does language pretraining put temporal structure, and does
forecasting finetuning preserve/erode it?
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
from scripts.run_frozen_probe import _load_model, _prompts

KINDS = ["trend", "periodic", "local", "mixture", "regime"]


@torch.no_grad()
def extract_all_layers_hidden(
    model, tokenizer, contexts: np.ndarray, device: str, batch_size: int = 32
) -> np.ndarray:
    """Return (n_layers, n_windows, hidden) final-token hidden states."""
    prompts = _prompts(contexts)
    all_layers = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(
            chunk, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        out = model(**encoded, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states)  # (L+1, B, S, H) includes embeddings
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        row_idx = torch.arange(len(chunk), device=device)
        per_layer = hidden[:, row_idx, lengths].float().cpu().numpy()  # (L+1, B, H)
        all_layers.append(per_layer)
    return np.concatenate(all_layers, axis=1)  # (L+1, N, H)


def probe_all_layers(h_all: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                     train_label: np.ndarray, test_label: np.ndarray,
                     softmax_iters: int) -> dict:
    """Fit a linear probe per layer; return per-layer accuracy + per-kind."""
    n_layers = h_all.shape[0]
    per_layer = {}
    for layer in range(n_layers):
        tr_h = h_all[layer, train_idx]
        te_h = h_all[layer, test_idx]
        _, pred = softmax_predict(te_h, *softmax_regression(tr_h, train_label, n_iter=softmax_iters))
        acc = accuracy(pred, test_label)
        per_kind = {
            KINDS[k]: accuracy(pred[test_label == k], test_label[test_label == k])
            for k in range(len(KINDS))
        }
        per_layer[str(layer)] = {"accuracy": float(acc), "per_kind": per_kind}
    return per_layer


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
    parser.add_argument("--adapter-path", type=Path, default=None,
                        help="LoRA adapter dir (PeftModel) for post-FT comparison")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/iclr/layerwise"))
    args = parser.parse_args()

    contexts, _, kind_ids = build_labeled_windows(
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

    model, tokenizer = _load_model(
        args.model_path, args.device, args.random_init, random_init_mode="reinit"
    )

    if args.adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
        model.eval()

    print(f"[layerwise] extracting {model.config.num_hidden_layers + 1} layers")
    h_all = extract_all_layers_hidden(model, tokenizer, contexts, args.device)
    print(f"[layerwise] hidden shape (layers, windows, dim) = {h_all.shape}")

    per_layer = probe_all_layers(
        h_all, train_idx, test_idx, train_label, test_label, args.softmax_iters
    )

    results = {
        "model": args.model_tag,
        "model_path": str(args.model_path),
        "adapter": str(args.adapter_path) if args.adapter_path else None,
        "random_init": args.random_init,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "seed": args.seed,
        "n_layers": int(h_all.shape[0]),
        "per_layer_accuracy": {k: v["accuracy"] for k, v in per_layer.items()},
        "per_layer": per_layer,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["per_layer_accuracy"], indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
