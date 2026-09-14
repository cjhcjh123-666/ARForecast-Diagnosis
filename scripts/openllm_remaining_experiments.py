"""Remaining robustness experiments for the open-LLM suite (CPU, from stored features).

Produces (results/open_llm_suite/raw/):
  tableB_all.csv              5-expert vs 3-expert decision-definition sensitivity, all LMs
  router_capacity_all.csv     linear vs MLP64 router (is the gain a linear-separability artefact?)
  dimension_matching.csv      PCA / random-projection down to 384-2048 dims (dimension-confound control)
  pooling_sensitivity.csv     last-non-pad token vs mean pooling

Both pretrained and random are kept so every reported quantity is the PAIRED delta.
"""
from __future__ import annotations
import csv, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.openllm_suite import mlp_router
from scripts.tsfm_expertbank_stability import TrendSeasonalExpert, ARExpert

KINDS = ["trend", "periodic", "local", "mixture", "regime"]
SEEDS = [int(x) for x in os.environ.get("OPENLLM_SEEDS", "7,17,27").split(",") if x]
INITS = [x for x in os.environ.get("OPENLLM_INITS", "pretrained,random").split(",") if x]
# Random projection keeps the geometry at any width; PCA is only a genuine truncation
# while dim < rank(X_train) (=270 training windows), above that every dim gives the SAME
# rank-270 projection, so those rows are dropped as degenerate.
DIMS_RP = [64, 128, 256, 384, 512, 768, 1024, 2048]
DIMS_PCA = [32, 64, 128, 256]
SUFFIX = os.environ.get("OPENLLM_OUT_SUFFIX", "")
FF = REPO / "results/open_llm_suite/features_full"
FEAT = REPO / "results/open_llm_suite/features"
OUT = REPO / "results/open_llm_suite/raw"
import os
MODELS = os.environ.get("OPENLLM_MODELS", "").split(",") if os.environ.get("OPENLLM_MODELS") else [
    "qwen3_8b_base", "llama31_8b", "llama32_3b", "gemma2_9b", "gemma2_2b",
    "mistral_7b_v03", "deepseek_llm_7b", "olmo2_7b", "olmo2_13b", "deepseek_v2_lite"]
ORIG = [TrendExpert(), PeriodicExpert(), LocalExpert()]
EXP5 = [TrendExpert(), TrendSeasonalExpert(), PeriodicExpert(), LocalExpert(), ARExpert(10, ridge=1.0)]
E5FAM = np.array([0, 0, 1, 2, 2])


def balacc(pred, oracle):
    cm = np.zeros((3, 3), int)
    for t, p in zip(oracle, pred):
        if t < 3 and p < 3:
            cm[t, p] += 1
    rec = np.array([cm[i, i] / (cm[i].sum() + 1e-12) for i in range(3)])
    return float(rec.mean())


def errs(ctx, fut, experts):
    E = np.zeros((len(ctx), len(experts)))
    for j, e in enumerate(experts):
        for i in range(len(ctx)):
            E[i, j] = float(np.mean((e.predict(ctx[i], 16) - fut[i]) ** 2))
    return E


def load_feats(model, init, seed):
    """consistent clean750 + bal3/bal3n features for one (model, init, seed)."""
    f = FF / ("full_%s_%s_s%d.npz" % (model, init, seed))
    if f.is_file():
        z = np.load(f)
        return dict(h_clean=z["h_clean"], kind=z["clean_kind"], h_bal3=z["h_bal3"], h_bal3n=z["h_bal3n"],
                    bal3_oracle=z["bal3_oracle"], bal3n_oracle=z["bal3n_oracle"],
                    h_clean_mean=z["h_clean_mean"] if "h_clean_mean" in z else None,
                    h_bal3_mean=z["h_bal3_mean"] if "h_bal3_mean" in z else None)
    fc = FEAT / ("pw_%s_%s_s%d.npz" % (model, init, seed))
    fo = FEAT / ("ood_%s_%s_s%d.npz" % (model, init, seed))
    if fc.is_file() and fo.is_file():
        zo = np.load(fo)
        return dict(h_clean=np.load(fc)["H"], kind=None, h_bal3=zo["h_bal3"], h_bal3n=zo["h_bal3n"],
                    bal3_oracle=zo["bal3_oracle"], bal3n_oracle=zo["bal3n_oracle"],
                    h_clean_mean=None, h_bal3_mean=None)
    return None


def windows(seed):
    b3 = np.load(REPO / ("results/iclr/e4_balanced3/windows/bal3_s%d.npz" % seed))
    b3n = np.load(REPO / ("results/iclr/e4_balanced3_natural/windows/bal3n_s%d.npz" % seed))
    return {"bal3": (b3["ctx"], b3["fut"]), "bal3n": (b3n["ctx"], b3n["fut"])}


def main():
    rowsB, rowsR, rowsD, rowsP = [], [], [], []
    for model in MODELS:
        for seed in SEEDS:
            ctx, fut, kk = build_labeled_windows(KINDS, 64, 16, 150, seed)
            clean270 = np.concatenate([np.arange(k * 150, k * 150 + 90) for k in range(3)])
            tr5 = np.concatenate([np.arange(k * 150, k * 150 + 90) for k in range(5)])
            te5 = np.concatenate([np.arange(k * 150 + 90, k * 150 + 150) for k in range(5)])
            W = windows(seed)
            Ec = errs(ctx[clean270], fut[clean270], ORIG)
            oclean = Ec.argmin(1)
            fam5 = {}
            for tag in W:
                cctx, cfut = W[tag]
                fam5[tag] = E5FAM[errs(cctx, cfut, EXP5).argmin(1)]
            for init in INITS:
                d = load_feats(model, init, seed)
                if d is None:
                    print("missing", model, init, seed, flush=True); continue
                H = d["h_clean"]; kind = d["kind"] if d["kind"] is not None else kk
                Wf = softmax_regression(H[clean270], kind[clean270], n_iter=2000)
                Wo = softmax_regression(H[clean270], oclean, n_iter=2000)
                for tag in ["bal3", "bal3n"]:
                    Hb = d["h_" + tag]; bor = d[tag + "_oracle"]
                    _, pf = softmax_predict(Hb, *Wf)
                    _, po = softmax_predict(Hb, *Wo)
                    cctx, cfut = W[tag]
                    E3 = errs(cctx, cfut, ORIG)
                    marg = E3[np.arange(len(cctx)), np.argsort(E3, 1)[:, 1]] / (E3.min(1) + 1e-9)
                    rowsB.append(dict(model=model, init=init, seed=seed, test=tag,
                                      family3=round(balacc(pf, bor), 4), oracle3=round(balacc(po, bor), 4),
                                      family5=round(balacc(pf, fam5[tag]), 4),
                                      margin_mean=round(float(marg.mean()), 4), n_windows=len(bor)))
                Hb = d["h_bal3"]; bor = d["bal3_oracle"]
                _, pf = softmax_predict(Hb, *Wf)
                pm = mlp_router(H[clean270], kind[clean270], Hb, seed=seed)
                rowsR.append(dict(model=model, init=init, seed=seed,
                                  linear=round(balacc(pf, bor), 4), mlp64=round(balacc(pm, bor), 4)))
                Xtr = H[clean270]
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                Xtrn = (Xtr - mu) / sd; Xben = (Hb - mu) / sd
                # reference row: no projection at all (must reproduce the runner's number)
                _, pf_ref = softmax_predict(Hb, *softmax_regression(H[clean270], kind[clean270], n_iter=2000))
                rowsD.append(dict(model=model, init=init, seed=seed, proj="full", dim=int(Xtr.shape[1]),
                                  ba=round(balacc(pf_ref, bor), 4)))
                rng = np.random.default_rng(seed)
                for dim in DIMS_RP:
                    if dim >= Xtr.shape[1]:
                        continue
                    Rp = rng.normal(0, 1, (Xtr.shape[1], dim)) / np.sqrt(dim)
                    _, pr = softmax_predict(Xben @ Rp, *softmax_regression(Xtrn @ Rp, kind[clean270], n_iter=2000))
                    rowsD.append(dict(model=model, init=init, seed=seed, proj="randproj", dim=dim,
                                      ba=round(balacc(pr, bor), 4)))
                U, S, Vt = np.linalg.svd(Xtrn, full_matrices=False)
                for dim in DIMS_PCA:
                    if dim > len(S):
                        continue
                    _, pp = softmax_predict(Xben @ Vt[:dim].T, *softmax_regression(Xtrn @ Vt[:dim].T, kind[clean270], n_iter=2000))
                    rowsD.append(dict(model=model, init=init, seed=seed, proj="pca", dim=dim,
                                      ba=round(balacc(pp, bor), 4)))
                if d["h_clean_mean"] is not None and d["h_bal3_mean"] is not None:
                    for pool, Hc, Hb2 in [("last", H, Hb), ("mean", d["h_clean_mean"], d["h_bal3_mean"])]:
                        Wc = softmax_regression(Hc[tr5], kind[tr5], n_iter=2000)
                        _, pc = softmax_predict(Hc[te5], *Wc)
                        _, prr = softmax_predict(Hb2, *softmax_regression(Hc[clean270], kind[clean270], n_iter=2000))
                        rowsP.append(dict(model=model, init=init, seed=seed, pooling=pool,
                                          clean_acc=round(float(np.mean(pc == kind[te5])), 4),
                                          family_ba_bal3=round(balacc(prr, bor), 4)))
                print("[%s/%s/s%d] done" % (model, init, seed), flush=True)
    for name, rows in [("tableB_all%s.csv" % SUFFIX, rowsB), ("router_capacity_all%s.csv" % SUFFIX, rowsR),
                       ("dimension_matching%s.csv" % SUFFIX, rowsD), ("pooling_sensitivity%s.csv" % SUFFIX, rowsP)]:
        if not rows:
            print("no rows for", name); continue
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", OUT / name, len(rows), "rows")
    print("\n== 5-expert (family5) vs 3-expert, bal3 ==")
    for model in MODELS:
        def g(kk_, init):
            return [r[kk_] for r in rowsB if r["model"] == model and r["test"] == "bal3" and r["init"] == init]
        a, b, c, dd = g("family3", "pretrained"), g("family3", "random"), g("family5", "pretrained"), g("family5", "random")
        if a and c:
            print("%-18s 3exp delta=%+.3f  5exp delta=%+.3f" % (model, np.mean(a) - np.mean(b), np.mean(c) - np.mean(dd)))
    print("\n== router capacity (bal3 delta) ==")
    for model in MODELS:
        def h(kk_, init):
            return [r[kk_] for r in rowsR if r["model"] == model and r["init"] == init]
        if h("linear", "pretrained"):
            print("%-18s linear=%+.3f  MLP64=%+.3f" % (model, np.mean(h("linear", "pretrained")) - np.mean(h("linear", "random")),
                                                       np.mean(h("mlp64", "pretrained")) - np.mean(h("mlp64", "random"))))
    print("\n== dimension matching (bal3 delta, pretrained - random) ==")
    for model in MODELS:
        for proj in ["full", "randproj", "pca"]:
            out = []
            dims = sorted({r["dim"] for r in rowsD if r["model"] == model and r["proj"] == proj})
            for dim in dims:
                p = [r["ba"] for r in rowsD if r["model"] == model and r["proj"] == proj and r["dim"] == dim and r["init"] == "pretrained"]
                q = [r["ba"] for r in rowsD if r["model"] == model and r["proj"] == proj and r["dim"] == dim and r["init"] == "random"]
                if p and q:
                    out.append("%d:%+.3f" % (dim, np.mean(p) - np.mean(q)))
            if out:
                print("%-18s %-8s %s" % (model, proj, " ".join(out)))
    print("\n== pooling (means over seeds) ==")
    for model in MODELS:
        for pool in ["last", "mean"]:
            p = [r["family_ba_bal3"] for r in rowsP if r["model"] == model and r["pooling"] == pool and r["init"] == "pretrained"]
            q = [r["family_ba_bal3"] for r in rowsP if r["model"] == model and r["pooling"] == pool and r["init"] == "random"]
            cp = [r["clean_acc"] for r in rowsP if r["model"] == model and r["pooling"] == pool and r["init"] == "pretrained"]
            if p:
                print("%-18s %-5s routing P=%.3f R=%.3f delta=%+.3f | clean P=%.3f" %
                      (model, pool, np.mean(p), np.mean(q), np.mean(p) - np.mean(q), np.mean(cp)))


if __name__ == "__main__":
    main()
