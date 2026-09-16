"""Linear probes over frozen representations (protocol-locked, see 02_protocol_lock.md)."""
from __future__ import annotations

import numpy as np

from analysis.linear_probe import softmax_predict, softmax_regression
from .statistics import balanced_accuracy, macro_f1, normalized_mae, r2_score


def split_indices(y, seed: int, target_index: int, frac: float = 0.6):
    """Per-target stratified split; depends only on (target, seed) so it is identical across models."""
    y = np.asarray(y)
    rng = np.random.default_rng(10000 + 100 * int(seed) + int(target_index))
    uniq = np.unique(y)
    if len(uniq) > 0.5 * len(y):
        # continuous (regression) target: plain random split, no stratification
        idx = rng.permutation(len(y))
        n_tr = int(round(frac * len(y)))
        return np.asarray(sorted(idx[:n_tr]), dtype=int), np.asarray(sorted(idx[n_tr:]), dtype=int)
    tr, te = [], []
    for cls in uniq:
        idx = np.where(y == cls)[0].copy()
        rng.shuffle(idx)
        n_tr = int(round(frac * len(idx)))
        n_tr = min(max(n_tr, 1), len(idx) - 1) if len(idx) > 1 else len(idx)
        tr.extend(idx[:n_tr].tolist())
        te.extend(idx[n_tr:].tolist())
    return np.asarray(sorted(tr), dtype=int), np.asarray(sorted(te), dtype=int)


def ridge_fit_predict(Xtr, ytr, Xte, alpha: float = 1.0):
    """Closed-form ridge on standardized features (protocol lock: alpha = 1.0)."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd; Zt = (Xte - mu) / sd
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (ytr - ytr.mean()))
    return Zt @ w + ytr.mean()


def run_probe(H, y, target_index: int, kind: str, seed: int, frac: float = 0.6):
    """Returns dict(metric -> value) for one (representation, target, seed)."""
    y = np.asarray(y)
    tr, te = split_indices(y, seed, target_index, frac)
    Xtr, Xte = H[tr], H[te]
    if kind == "regression":
        pred = ridge_fit_predict(Xtr, y[tr].astype(float), Xte)
        return dict(r2=r2_score(y[te], pred), nmae=normalized_mae(y[te], pred),
                    n_train=len(tr), n_test=len(te))
    n_classes = int(max(y.max() + 1, 2))
    W, mu, sd = softmax_regression(Xtr, y[tr], n_iter=2000)
    _, pred = softmax_predict(Xte, W, mu, sd)
    return dict(balanced_accuracy=balanced_accuracy(pred, y[te], n_classes),
                macro_f1=macro_f1(pred, y[te]), n_train=len(tr), n_test=len(te))


def run_pair_probe(H, pairs, labels, target_index: int, kind: str, seed: int, frac: float = 0.6):
    """Relation probe: features are concat(h_a, h_b) for the pairs."""
    X = np.concatenate([H[pairs[:, 0]], H[pairs[:, 1]]], axis=1)
    return run_probe(X, labels, target_index, kind, seed, frac)


def presence_signature_metric(pred_bits, true_bits):
    """Per-label balanced accuracy + exact-signature accuracy for the compositional probe."""
    out = {}
    for j in range(true_bits.shape[1]):
        out[f"ba_label{j}"] = balanced_accuracy(pred_bits[:, j], true_bits[:, j], 2)
    out["exact_signature"] = float((pred_bits == true_bits).all(1).mean())
    out["mean_ba"] = float(np.mean([v for k, v in out.items() if k.startswith("ba_label")]))
    return out
