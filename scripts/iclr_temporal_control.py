"""ICLR item-2: temporal-order control.

Q: does the probe recognize temporal dynamics, or just per-process numeric
distributions (histogram shape)?

Conditions (Qwen3-8B, matched random 8B, feature baseline; same windows/seeds):
  - Original:      current temporal order
  - Shuffled:      per-window random permutation of the time order (train+test)
  - Marginal-only: order-free features (quantiles, mean/std, skew, kurtosis,
                   min/max/range) -- no temporal ordering at all
Compare clean-family recognition and clean -> mixture/regime OOD routing.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _load_model, extract_last_hidden
KINDS = ["trend", "periodic", "local", "mixture", "regime"]

def marginal_features(x):
    q = np.quantile(x, [0.1,0.25,0.5,0.75,0.9])
    s = x.std()
    return np.concatenate([q, [x.mean(), s, (x.mean()/ (s+1e-9)),
                               np.mean((x-x.mean())**3)/(s**3+1e-9),
                               np.mean((x-x.mean())**4)/(s**4+1e-9)-3,
                               x.min(), x.max(), x.max()-x.min()]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B")
    ap.add_argument("--model-tag", default="qwen3_8b")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-per-kind", type=int, default=150)
    ap.add_argument("--n-train-per-kind", type=int, default=90)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/temporal_control/qwen3_8b.json"))
    a = ap.parse_args()
    n = a.n_per_kind; n_tr = a.n_train_per_kind
    contexts, futures, kind_ids = build_labeled_windows(KINDS, 64, 16, n, a.seed)
    train_idx = np.concatenate([np.arange(k*n, k*n+n_tr) for k in range(5)])
    test_idx = np.concatenate([np.arange(k*n+n_tr, (k+1)*n) for k in range(5)])
    tr_label, te_label = kind_ids[train_idx], kind_ids[test_idx]

    rng = np.random.default_rng(0)
    shuffled = np.stack([rng.permutation(w) for w in contexts])
    marginal = np.stack([marginal_features(w) for w in contexts])

    def probe(X, kind):
        _, p = softmax_predict(X[test_idx], *softmax_regression(X[train_idx], tr_label, n_iter=2000))
        return accuracy(p, te_label)
    def route(X):
        hard = np.isin(te_label, [3,4])
        clean = tr_label < 3
        _, p = softmax_predict(X[test_idx][hard], *softmax_regression(X[train_idx][clean], tr_label[clean], n_iter=2000))
        err = np.zeros((int(hard.sum()), 3))
        h_idx = test_idx[hard]
        experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
        for j,e in enumerate(experts):
            for i,w in enumerate(h_idx):
                err[i,j] = float(np.mean((e.predict(contexts[w],16)-futures[w])**2))
        oracle = err.argmin(axis=1)
        return accuracy(p, oracle), float(np.mean(err[np.arange(len(p)),p])), err.min(axis=1).mean()

    rep = {"model": a.model_tag, "seed": a.seed}
    # feature baselines (order-free marginal vs temporal features)
    feats_te = np.stack([temporal_features(x) for x in contexts])
    feats_sh = np.stack([temporal_features(x) for x in shuffled])
    rep["feature_temporal"] = {"recog": probe(feats_te,"f"),
                               "route_acc": route(feats_te)[0]}
    rep["feature_shuffled"] = {"recog": probe(feats_sh,"f"),
                               "route_acc": route(feats_sh)[0]}
    rep["marginal_only"] = {"recog": probe(marginal,"m"),
                            "route_acc": route(marginal)[0]}

    # LLM: original (cached), shuffled (extract), marginal (n/a)
    for init in ["pretrained", "random"]:
        cache_dir = REPO_ROOT/("results/icassp/probe/cache" if a.model_tag=="qwen3_8b" else f"results/iclr/probe/{a.model_tag}/seed{a.seed}/cache")
        h_orig = np.load(cache_dir/f"train_h_{init}_seed{a.seed}.npy")   # 450, 90/kind
        te_orig = np.load(cache_dir/f"test_h_{init}_seed{a.seed}.npy")   # 300, 60/kind
        # rebuild ORIGINAL window order: [kind0: 90tr+60te, kind1: ..., ...] (750)
        X_orig = np.concatenate([
            np.concatenate([h_orig[k*90:(k+1)*90], te_orig[k*60:(k+1)*60]])
            for k in range(5)
        ], axis=0)
        model, tok = _load_model(a.model_path, a.device, init=="random")
        X_sh = extract_last_hidden(model, tok, shuffled, a.device)
        del model; torch.cuda.empty_cache()
        rep[f"llm_{init}"] = {
            "original": {"recog": probe(X_orig,"h"), "route_acc": route(X_orig)[0], "route_mse": route(X_orig)[1]},
            "shuffled": {"recog": probe(X_sh,"h"), "route_acc": route(X_sh)[0], "route_mse": route(X_sh)[1]},
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"saved={a.out}")

if __name__ == "__main__":
    main()
