"""Level-2 probes: pairwise relations (R1-R7) and compositional generalisation (C1-C3).

Features: concat(h_a, h_b) of the frozen final-layer representations (protocol lock).
Probe: single linear readout (softmax_regression, 2000 iters) for classification.
Splits: windows are split first (60/40, seeded), pairs are then formed inside each split, so no
window can appear in both train and test of the same task.

Outputs one CSV per (model, seed, init):
  results/temporal_abstraction/relations/level2_<model>_<init>_s<seed>.csv
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression
from temporal_abstraction.statistics import balanced_accuracy
from temporal_abstraction.primitive_targets import PRESENCE_SIGNATURE

POOL = REPO / "results/temporal_abstraction/primitives"
REPS = REPO / "results/temporal_abstraction/reps"
OUT = REPO / "results/temporal_abstraction/relations"
SEEN = ["TREND", "PERIODIC", "LOCAL", "TP"]
UNSEEN = ["TL", "TPL"]
SIG = {"TREND": (1, 0, 0), "PERIODIC": (0, 1, 0), "LOCAL": (0, 0, 1), "TP": (1, 1, 0),
       "TL": (1, 0, 1), "TPL": (1, 1, 1), "REGIME": (1, 1, 0)}


def load_pool(seed):
    lat = [json.loads(l) for l in open(POOL / f"pool_s{seed}.jsonl")]
    idx = {}
    for i, r in enumerate(lat):
        idx.setdefault(r["block"], []).append(i)
    return lat, idx


def split_windows(ids, rng, frac=0.6):
    ids = np.asarray(ids); ids = ids[rng.permutation(len(ids))]
    n = int(round(frac * len(ids)))
    return ids[:n], ids[n:]


def pairs_r1(lat, idx, rng, n):          # stronger trend
    out = []
    pool = list(idx["TREND"])
    while len(out) < n:
        a, b = rng.choice(pool, 2, replace=False)
        da, db = abs(lat[a]["slope"]), abs(lat[b]["slope"])
        if abs(da - db) < 0.03:
            continue
        out.append((a, b, int(da > db)))
    return out


def pairs_r2(lat, idx, rng, n):          # which series is periodic
    a_pool, b_pool = list(idx["PERIODIC"]), list(idx["APERIODIC"])
    out = []
    while len(out) < n:
        a, b = rng.choice(a_pool), rng.choice(b_pool)
        if rng.random() < 0.5:
            out.append((a, b, 1))
        else:
            out.append((b, a, 0))
    return out


def pairs_r3(lat, idx, rng, n):          # higher volatility
    pool = list(idx["LOCAL"])
    out = []
    while len(out) < n:
        a, b = rng.choice(pool, 2, replace=False)
        if lat[a]["sigma"] == lat[b]["sigma"]:
            continue
        out.append((a, b, int(lat[a]["sigma"] > lat[b]["sigma"])))
    return out


def pairs_r4(lat, idx, rng, n):          # same family?
    fams = ["TREND", "PERIODIC", "LOCAL"]
    out = []
    while len(out) < n:
        if rng.random() < 0.5:
            f = str(rng.choice(fams)); a, b = rng.choice(idx[f], 2, replace=False); y = 1
        else:
            f1, f2 = rng.choice(fams, 2, replace=False)
            a, b, y = rng.choice(idx[f1]), rng.choice(idx[f2]), 0
        out.append((a, b, int(y)))
    return out


def pairs_r5(lat, idx, rng, n):          # same dominant period?
    pool = list(idx["PERIODIC"])
    out = []
    while len(out) < n:
        a, b = rng.choice(pool, 2, replace=False)
        out.append((a, b, int(lat[a]["period"] == lat[b]["period"])))
    return out


def pairs_r6(lat, idx, rng, n):          # which series is anomalous
    out = []
    while len(out) < n:
        a, b = rng.choice(idx["ANOM"]), rng.choice(idx["ANOM_CLEAN"])
        if rng.random() < 0.5:
            out.append((a, b, 1))
        else:
            out.append((b, a, 0))
    return out


def pairs_r7(lat, idx, rng, n):          # same latent class (family + similar parameters)?
    fams = ["TREND", "PERIODIC", "LOCAL"]
    out = []
    while len(out) < n:
        if rng.random() < 0.5:
            f = str(rng.choice(fams)); a, b = rng.choice(idx[f], 2, replace=False); y = 1
        else:
            f1, f2 = rng.choice(fams, 2, replace=False)
            a, b, y = rng.choice(idx[f1]), rng.choice(idx[f2]), 0
        out.append((a, b, int(y)))
    return out


RELATIONS = [("R1_stronger_trend", pairs_r1), ("R2_periodicity_present", pairs_r2),
             ("R3_higher_volatility", pairs_r3), ("R4_same_family", pairs_r4),
             ("R5_same_period", pairs_r5), ("R6_anomaly_vs_clean", pairs_r6),
             ("R7_same_latent_class", pairs_r7)]


def fit_eval(X, y, tr, te, seed):
    W, mu, sd = softmax_regression(X[tr], y[tr], n_iter=2000)
    _, pred = softmax_predict(X[te], W, mu, sd)
    return balanced_accuracy(pred, y[te], int(max(y.max() + 1, 2))), len(tr), len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--n-pairs", type=int, default=1200)
    ap.add_argument("--layer", default="final")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rep_f = REPS / f"{a.model_key}_{a.init}_s{a.seed}.npz"
    if not rep_f.is_file():
        raise SystemExit(f"missing {rep_f}")
    z = np.load(rep_f)
    layer_ids = z["layer_ids"].tolist()
    lid = layer_ids[-1] if a.layer == "final" else layer_ids[0]
    H = z[f"h_L{lid}"].astype(np.float32)
    lat, idx = load_pool(a.seed)
    rng = np.random.default_rng(10000 + a.seed)
    rows = []
    # ---------------- relations ----------------
    for name, builder in RELATIONS:
        tr_pool, te_pool = {}, {}
        for blk in set(idx):
            tr_pool[blk], te_pool[blk] = split_windows(idx[blk], np.random.default_rng(hash((name, blk, a.seed)) % (2**31)))
        def build(tag):
            sub_idx = tr_pool if tag == "train" else te_pool
            sub = {k: v for k, v in sub_idx.items()}
            # reuse the builder on the restricted pool
            return builder(lat, sub, np.random.default_rng(hash((name, tag, a.seed)) % (2**31)), a.n_pairs // 2)
        tr = build("train"); te = build("test")
        Xtr = np.concatenate([H[[p[0] for p in tr]], H[[p[1] for p in tr]]], axis=1)
        ytr = np.asarray([p[2] for p in tr])
        Xte = np.concatenate([H[[p[0] for p in te]], H[[p[1] for p in te]]], axis=1)
        yte = np.asarray([p[2] for p in te])
        ba, ntr, nte = fit_eval(np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte]),
                                np.arange(len(ytr)), np.arange(len(ytr), len(ytr) + len(yte)), a.seed)
        rows.append(dict(model=a.model_key, init=a.init, seed=a.seed, task=name, family="relation",
                         n_train=ntr, n_test=nte, metric="balanced_accuracy", score=round(ba, 4)))
        print(f"[{a.model_key}/{a.init}/s{a.seed}] {name} BA={ba:.3f}", flush=True)
    # ---------------- composition (presence signature) ----------------
    tr_ids = np.concatenate([np.asarray(idx[b]) for b in SEEN])
    rng2 = np.random.default_rng(10 + a.seed)
    tr_ids = tr_ids[rng2.permutation(len(tr_ids))]
    tr_ids = tr_ids[: int(0.75 * len(tr_ids))]
    te_ids = np.concatenate([np.asarray(idx[b]) for b in UNSEEN])
    if len(te_ids):
        Ytr = np.asarray([SIG[lat[i]["block"]] for i in tr_ids])
        Yte = np.asarray([SIG[lat[i]["block"]] for i in te_ids])
        for j, lab in enumerate(["trend_present", "periodic_present", "local_present"]):
            X = H[tr_ids]
            W, mu, sd = softmax_regression(X, Ytr[:, j], n_iter=2000)
            _, pred = softmax_predict(H[te_ids], W, mu, sd)
            ba = balanced_accuracy(pred, Yte[:, j], 2)
            rows.append(dict(model=a.model_key, init=a.init, seed=a.seed, task=f"C1_signature_{lab}",
                             family="composition_unseen", n_train=len(tr_ids), n_test=len(te_ids),
                             metric="balanced_accuracy", score=round(ba, 4)))
            print(f"[{a.model_key}/{a.init}/s{a.seed}] C1 {lab} BA={ba:.3f}", flush=True)
        # exact signature (all three heads)
        preds = []
        for j in range(3):
            W, mu, sd = softmax_regression(H[tr_ids], Ytr[:, j], n_iter=2000)
            _, p = softmax_predict(H[te_ids], W, mu, sd)
            preds.append(p)
        P = np.stack(preds, 1)
        rows.append(dict(model=a.model_key, init=a.init, seed=a.seed, task="C1_exact_signature",
                         family="composition_unseen", n_train=len(tr_ids), n_test=len(te_ids),
                         metric="exact_match", score=round(float((P == Yte).all(1).mean()), 4)))
    # arrangement probe C3: TP (simultaneous) vs REGIME (sequential)
    a_ids, b_ids = np.asarray(idx["TP"]), np.asarray(idx["REGIME"])
    rng3 = np.random.default_rng(20 + a.seed)
    a_i = a_ids[rng3.permutation(len(a_ids))]; b_i = b_ids[rng3.permutation(len(b_ids))]
    n = min(len(a_i), len(b_i))
    Xa, Xb = H[a_i[:n]], H[b_i[:n]]
    X = np.concatenate([Xa, Xb]); y = np.asarray([0] * n + [1] * n)
    m = rng3.permutation(len(y)); X, y = X[m], y[m]
    cut = int(0.6 * len(y))
    ba, ntr, nte = fit_eval(X, y, np.arange(cut), np.arange(cut, len(y)), a.seed)
    rows.append(dict(model=a.model_key, init=a.init, seed=a.seed, task="C3_arrangement_tp_vs_regime",
                     family="composition_arrangement", n_train=ntr, n_test=nte,
                     metric="balanced_accuracy", score=round(ba, 4)))
    print(f"[{a.model_key}/{a.init}/s{a.seed}] C3 arrangement BA={ba:.3f}", flush=True)
    out = OUT / f"level2_{a.model_key}_{a.init}_s{a.seed}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out.name, flush=True)


if __name__ == "__main__":
    main()
