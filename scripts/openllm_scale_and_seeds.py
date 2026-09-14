"""Model-scale sweep (Qwen3 0.6B/1.7B/8B) and the 10-seed headline replication for Qwen3-8B.

Uses the stored router-relevant features (clean270 + bal3/bal3n, seeds 7..97) from
qwen_feats/ and qwen_feats_10s/, i.e. the same protocol as the main suite:
train the linear router on the 270 clean primitive windows, evaluate balanced accuracy
on the frozen 120-window balanced OOD set (40/40/40) against the future-defined oracle.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression

OUT = REPO / "results/open_llm_suite/raw"
SHORT = ["qwen3_0.6b", "qwen3_1.7b", "qwen3_8b_base"]
PARAMS = {"qwen3_0.6b": "0.6B", "qwen3_1.7b": "1.7B", "qwen3_8b_base": "8.2B"}
HIDDEN = {"qwen3_0.6b": 1024, "qwen3_1.7b": 2048, "qwen3_8b_base": 4096}


def balacc(pred, oracle):
    cm = np.zeros((3, 3), int)
    for t, p in zip(oracle, pred):
        if t < 3 and p < 3:
            cm[t, p] += 1
    rec = np.array([cm[i, i] / (cm[i].sum() + 1e-12) for i in range(3)])
    return float(rec.mean())


def feat_path(model, init, seed):
    for d in ["qwen_feats", "qwen_feats_10s"]:
        p = REPO / d / ("%s_%s_s%d.npz" % (model, init, seed))
        if p.is_file():
            return p
    return None


def one(model, init, seed, tag="bal3"):
    p = feat_path(model, init, seed)
    if p is None:
        return None
    z = np.load(p)
    H, Ho = z["h_clean"], z["h_" + tag]
    y, oy = z["clean_family"], z[tag + "_oracle"]
    _, pred = softmax_predict(Ho, *softmax_regression(H, y, n_iter=2000))
    return balacc(pred, oy)


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)  # rng must persist across resamples
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(d.mean()), float(np.std(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    rows = []
    print("== scale sweep (family-label bal3, delta = pretrained - random) ==")
    for model in SHORT:
        ds = []
        for seed in [7, 17, 27]:
            p, r = one(model, "pretrained", seed), one(model, "random", seed)
            if p is None or r is None:
                print("missing features for", model, seed); continue
            ds.append(p - r)
            print("  %-14s s%-3d P=%.3f R=%.3f d=%+.3f" % (model, seed, p, r, p - r))
        if ds:
            m, sd, lo, hi = boot(ds)
            rows.append(dict(model=model, params=PARAMS[model], hidden=HIDDEN[model], n_seeds=len(ds),
                             delta=round(m, 4), std=round(sd, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             per_seed=";".join("%+.3f" % x for x in ds)))
    with open(OUT / "scale_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote scale_sweep.csv")

    print("\n== 10-seed headline replication, Qwen3-8B (family-label bal3) ==")
    rows2 = []
    for tag in ["bal3", "bal3n"]:
        ds, per = [], []
        for seed in [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]:
            p, r = one("qwen3_8b_base", "pretrained", seed, tag), one("qwen3_8b_base", "random", seed, tag)
            if p is None or r is None:
                continue
            ds.append(p - r); per.append((seed, p, r))
        m, sd, lo, hi = boot(ds)
        print("  %-5s n=%d delta=%+.4f (sd %.4f) 95%%CI [%+.4f,%+.4f]" % (tag, len(ds), m, sd, lo, hi))
        for s, p, r in per:
            print("        s%-3d P=%.3f R=%.3f d=%+.3f" % (s, p, r, p - r))
        rows2.append(dict(test=tag, model="qwen3_8b_base", n_seeds=len(ds), delta=round(m, 4), std=round(sd, 4),
                          ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                          per_seed=";".join("s%d:%+.3f" % (s, p - r) for s, p, r in per)))
    with open(OUT / "ten_seed_headline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0].keys())); w.writeheader(); w.writerows(rows2)
    print("wrote ten_seed_headline.csv")


if __name__ == "__main__":
    main()
