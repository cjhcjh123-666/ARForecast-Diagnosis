"""ICLR P0-2: complete numerical baselines on 15 real datasets.

For each dataset (official 8B, seed-7 windows): Persistence | Best Fixed |
Feature Router | Qwen Router | Oracle.  Answer: does Qwen recognition actually
improve forecasting over context-only references, not just over the synthetic-
trained feature router?
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("results/iclr/multi_dataset_router/qwen3_8b_official"))
    ap.add_argument("--probe-cache", type=Path, default=Path("results/iclr/probe_official/qwen3_8b/seed7/cache"))
    ap.add_argument("--out", type=Path, default=Path("results/iclr/realdata_full_baselines/qwen3_8b_official.json"))
    a=ap.parse_args()
    wdir=a.run_dir/"windows_qwen3_8b_official_s7"; hdir=a.run_dir/"hidden_qwen3_8b_official_s7"
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]; names=["trend","periodic","local"]
    # synthetic router training material (official 8B, seed7)
    c,_,kk=build_labeled_windows(KINDS,64,16,150,7)
    tr=np.concatenate([np.arange(k*150,k*150+90) for k in range(3)]); clean=kk[tr]
    Htr=np.load(a.probe_cache/"train_h_pretrained_seed7.npy")[tr]
    feats_tr=np.stack([temporal_features(x) for x in c[tr]])
    report={"model":"official Qwen3-8B-Base","datasets":{}}
    for wp in sorted(wdir.glob("*_windows.npz")):
        ds=wp.name.replace("_windows.npz","")
        z=np.load(wp); ctx,fut=z["ctx"],z["fut"]
        h=np.load(hdir/f"{ds}_pretrained.npy")
        err=np.zeros((len(ctx),3))
        for j,e in enumerate(experts):
            for i in range(len(ctx)): err[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
        pers=np.array([float(np.mean((np.full(16,ctx[i,-1])-fut[i])**2)) for i in range(len(ctx))])
        bf=int(err.mean(axis=0).argmin()); bestf=err[:,bf]
        oracle=err.min(axis=1)
        te_feats=np.stack([temporal_features(x) for x in ctx])
        _,p_feat=softmax_predict(te_feats,*softmax_regression(feats_tr,clean,n_iter=2000))
        _,p_q=softmax_predict(h,*softmax_regression(Htr,clean,n_iter=2000))
        q=err[np.arange(len(ctx)),p_q]; feat=err[np.arange(len(ctx)),p_feat]
        report["datasets"][ds]={
            "persistence":round(float(pers.mean()),4),"best_fixed":round(float(bestf.mean()),4),
            "best_fixed_name":names[bf],"feature_router":round(float(feat.mean()),4),
            "qwen_router":round(float(q.mean()),4),"oracle":round(float(oracle.mean()),4)}
        print(f"{ds:12s} persist={pers.mean():.4f} bestfixed={bestf.mean():.4f}({names[bf]}) feat={feat.mean():.4f} qwen={q.mean():.4f} oracle={oracle.mean():.4f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
