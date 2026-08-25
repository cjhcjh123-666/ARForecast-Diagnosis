"""ICLR item-5: real-data routing over 3 router-training seeds.

Router is trained on synthetic clean dynamics at seeds {7,17,27} and applied to
FIXED real test windows (seed-7 real sampling, cached hidden states for 8B).
Reports per dataset: LM vs feature router MSE (mean over seeds), window-level
bootstrap CI (pooled over seed x window), and paired LM - feature difference.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS = ["trend", "periodic", "local", "mixture", "regime"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("results/iclr/multi_dataset_router/qwen3_8b"))
    ap.add_argument("--model-tag", default="qwen3_8b")
    ap.add_argument("--router-seeds", default="7,17,27")
    ap.add_argument("--out", type=Path, default=Path("results/iclr/realdata_3seeds/qwen3_8b.json"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.router_seeds.split(",")]
    wdir = a.run_dir / f"windows_{a.model_tag}_s7"
    hdir = a.run_dir / f"hidden_{a.model_tag}_s7"
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]

    # synthetic router training material per seed
    syn = {}
    for s in seeds:
        c, _, kk = build_labeled_windows(KINDS, 64, 16, 150, s)
        tr = np.concatenate([np.arange(k*150, k*150+90) for k in range(3)])
        clean = kk[tr]
        feats = np.stack([temporal_features(x) for x in c[tr]])
        cache = REPO_ROOT/f"results/icassp/probe_{'probe' if s==7 else f'probe_seed{s}'}"
        # correct: seed7 -> probe/cache, others -> probe_seedX/cache
        cache_dir = REPO_ROOT/("results/icassp/probe/cache" if s==7 else f"results/icassp/probe_seed{s}/cache")
        syn[s] = {
            "h_pre": np.load(cache_dir/f"train_h_pretrained_seed{s}.npy")[tr],
            "h_rnd": np.load(cache_dir/f"train_h_random_seed{s}.npy")[tr],
            "feat": feats, "label": clean,
        }

    report = {"model": a.model_tag, "router_seeds": seeds, "datasets": {}}
    for wp in sorted(wdir.glob("*_windows.npz")):
        ds = wp.name.replace("_windows.npz","")
        z = np.load(wp); ctx, fut = z["ctx"], z["fut"]
        h_pre = np.load(hdir/f"{ds}_pretrained.npy")
        h_rnd = np.load(hdir/f"{ds}_random.npy")
        feats = np.stack([temporal_features(x) for x in ctx])
        err = np.zeros((len(ctx), 3))
        for j,e in enumerate(experts):
            for i in range(len(ctx)):
                err[i,j] = float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
        per_seed = {"lm": [], "feat": []}
        for s in seeds:
            _, p_pre = softmax_predict(h_pre, *softmax_regression(syn[s]["h_pre"], syn[s]["label"], n_iter=2000))
            _, p_rnd = softmax_predict(h_rnd, *softmax_regression(syn[s]["h_rnd"], syn[s]["label"], n_iter=2000))
            _, p_feat = softmax_predict(feats, *softmax_regression(syn[s]["feat"], syn[s]["label"], n_iter=2000))
            per_seed["lm"].append(err[np.arange(len(ctx)), p_pre].tolist())
            per_seed["feat"].append(err[np.arange(len(ctx)), p_feat].tolist())
        # pooled (seed x window)
        lm = np.concatenate([np.array(v) for v in per_seed["lm"]])
        ft = np.concatenate([np.array(v) for v in per_seed["feat"]])
        def ci(x):
            m = np.mean(x); rng = np.random.default_rng(0)
            ms = np.array([np.mean(rng.choice(x, len(x), replace=True)) for _ in range(2000)])
            return m, np.percentile(ms,2.5), np.percentile(ms,97.5)
        lm_m, lm_lo, lm_hi = ci(lm); ft_m, ft_lo, ft_hi = ci(ft)
        d = lm - ft; d_m, d_lo, d_hi = ci(d)
        report["datasets"][ds] = {
            "lm_mse": [round(np.mean(v),4) for v in per_seed["lm"]],
            "feat_mse": [round(np.mean(v),4) for v in per_seed["feat"]],
            "lm_pooled": {"mean": round(lm_m,4), "ci": [round(lm_lo,4), round(lm_hi,4)]},
            "feat_pooled": {"mean": round(ft_m,4), "ci": [round(ft_lo,4), round(ft_hi,4)]},
            "lm_minus_feat": {"mean": round(d_m,4), "ci": [round(d_lo,4), round(d_hi,4)]},
        }
        print(f"{ds:12s} lm={lm_m:.4f}[{lm_lo:.3f},{lm_hi:.3f}] feat={ft_m:.4f} diff={d_m:+.4f}[{d_lo:+.3f},{d_hi:+.3f}]")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={a.out}")

if __name__ == "__main__":
    main()
