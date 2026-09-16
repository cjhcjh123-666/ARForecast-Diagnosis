"""Statistics helpers shared by the abstraction probes (paired, window-level)."""
from __future__ import annotations

import numpy as np


def balanced_accuracy(pred, true, n_classes=None):
    pred = np.asarray(pred); true = np.asarray(true)
    classes = np.unique(true) if n_classes is None else np.arange(n_classes)
    recs = []
    for c in classes:
        m = true == c
        if m.sum():
            recs.append(float((pred[m] == c).mean()))
    return float(np.mean(recs)) if recs else float("nan")


def macro_f1(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    classes = np.unique(true)
    f1s = []
    for c in classes:
        tp = float(((pred == c) & (true == c)).sum())
        fp = float(((pred == c) & (true != c)).sum())
        fn = float(((pred != c) & (true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def r2_score(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def normalized_mae(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    return float(np.abs(y - pred).mean() / (y.std() + 1e-12))


def paired_bootstrap(deltas, n=10000, seed=0, alpha=0.05):
    """Percentile CI of the mean of paired per-window deltas."""
    d = np.asarray(deltas, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return dict(mean=float(d.mean()), lo=float(np.percentile(bs, 100 * alpha / 2)),
                hi=float(np.percentile(bs, 100 * (1 - alpha / 2))), n=int(len(d)))


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    m = ok.sum()
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = idx[rank]
        prev = min(prev, p[ok][i] * m / (rank + 1))
        q[ok][i] = prev
    return q
