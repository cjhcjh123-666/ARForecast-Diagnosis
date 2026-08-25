"""ICLR item-4: paired bootstrap CIs for DIFFERENCES on identical windows.

Computes stratified (seed x family) bootstrap 95% CIs for:
  - recognition:  pretrained - random            (per-window correct 0/1)
  - OOD hit rate: pretrained - random, pretrained - feature
  - routing MSE:  pretrained - random, pretrained - feature
  - generation:   LM mse - baseline mse          (persistence / best-fixed /
                  context-only router), per parsed window
Positive difference => pretrained/LM better.  Stratified resampling keeps the
seed x family structure of the evaluation set.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from scripts.iclr_bootstrap_window import per_window_outcomes, _cache_dir
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS = ["trend", "periodic", "local", "mixture", "regime"]

def strat_boot_ci(diffs, strata, n_boot=2000, seed=0):
    """diffs: per-window paired difference; strata: per-window stratum id."""
    diffs = np.asarray(diffs, float); strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = {s: np.where(strata == s)[0] for s in np.unique(strata)}
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups.values()])
        means[b] = np.mean(diffs[idx])
    return {"mean": float(np.mean(diffs)), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(len(diffs))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--gen-json", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    # ---- recognition + routing: per-window paired outcomes ----
    recog_diff, rnd_strat, rt = [], [], []
    for seed in seeds:
        o = per_window_outcomes(a.model, seed)
        n_tr = len(o["recog_pre"]); n_rt = len(o["route_pre_acc"])
        recog_diff += (o["recog_pre"] - o["recog_rnd"]).tolist()
        rnd_strat += [f"recog-{seed}"] * n_tr
        rt.append((o, seed, n_rt))
    rep = {"model": a.model, "seeds": seeds, "resampling": "stratified (seed x family)", "n_boot": 2000}
    rep["recognition_pre_minus_random"] = strat_boot_ci(recog_diff, rnd_strat)

    for o, seed, n in rt:
        for key, name in [("route_pre_acc","pre"),("route_rnd_acc","rnd"),("route_feat_acc","feat"),
                          ("route_pre_mse","pre_mse"),("route_rnd_mse","rnd_mse"),("route_feat_mse","feat_mse")]:
            pass
    # pool routing across seeds
    hit = {"pre": [], "rnd": [], "feat": []}
    mse = {"pre": [], "rnd": [], "feat": []}
    strat = []
    for o, seed, n in rt:
        for k in hit: hit[k] += o[f"route_{k}_acc"].tolist()
        for k in mse: mse[k] += o[f"route_{k}_mse"].tolist()
        strat += [f"route-{seed}"] * n
    strat = np.array(strat)
    for k in ["pre", "rnd", "feat"]:
        hit[k] = np.array(hit[k]); mse[k] = np.array(mse[k])
    rep["ood_hit_pre_minus_random"] = strat_boot_ci(hit["pre"] - hit["rnd"], strat)
    rep["ood_hit_pre_minus_feature"] = strat_boot_ci(hit["pre"] - hit["feat"], strat)
    rep["routing_mse_pre_minus_random"] = strat_boot_ci(mse["pre"] - mse["rnd"], strat)
    rep["routing_mse_pre_minus_feature"] = strat_boot_ci(mse["pre"] - mse["feat"], strat)

    # ---- generation: LM vs context-only baselines (per parsed window) ----
    if a.gen_json and a.gen_json.is_file():
        g = json.loads(a.gen_json.read_text())
        for base in ["persistence_mse", "best_fixed_mse", "context_router_mse"]:
            d, s = [], []
            for seed in seeds:
                r = g.get(str(seed), g.get(seed))
                if not r or not r.get(base): continue
                lm = np.array(r["lm_mse"]); bs = np.array(r[base])
                d += (lm - bs).tolist()   # positive => LM worse
                s += [f"gen-{seed}"] * len(lm)
            if d:
                rep[f"generation_lm_minus_{base}"] = strat_boot_ci(d, s)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=2, default=float))
    print(f"saved={a.out}")

if __name__ == "__main__":
    main()
