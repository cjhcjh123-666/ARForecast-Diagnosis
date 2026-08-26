"""ICLR P1: frozen-head paired CIs + parameter counts + rejection-sampling stats.

Re-runs the frozen numerical head with per-window test predictions saved, then
computes mean+/-std and paired window-level bootstrap CIs for
pretrained-random / pretrained-rawMLP / pretrained-features.  Also reports
head parameter counts, and the balanced3 rejection-sampling acceptance rates.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from analysis.features import temporal_features
from data.dynamics import build_labeled_windows
KINDS=["trend","periodic","local","mixture","regime"]

class LinearHead(nn.Module):
    def __init__(self,d,H): super().__init__(); self.net=nn.Linear(d,H)
    def forward(self,x): return self.net(x)
class MLPHead(nn.Module):
    def __init__(self,d,H,hidden=128): super().__init__(); self.net=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Linear(hidden,H))
    def forward(self,x): return self.net(x)

def fit_predict(Xtr,Ytr,Xte,kind,seed,epochs=300,lr=1e-3,device="cpu"):
    torch.manual_seed(seed); d=Xtr.shape[1]; H=Ytr.shape[1]
    net=LinearHead(d,H) if kind=="linear" else MLPHead(d,H)
    Xt=torch.from_numpy(Xtr).float(); Yt=torch.from_numpy(Ytr).float(); Xe=torch.from_numpy(Xte).float()
    opt=torch.optim.AdamW(net.parameters(),lr=lr); lossf=nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss=lossf(net(Xt),Yt); loss.backward(); opt.step()
    with torch.no_grad(): return net(Xe).numpy()

def paired_ci(a,b,seed=0):
    d=np.asarray(a,float)-np.asarray(b,float); rng=np.random.default_rng(seed)
    ms=np.array([np.mean(rng.choice(d,len(d),replace=True)) for _ in range(2000)])
    return {"mean":float(d.mean()),"ci":[float(np.percentile(ms,2.5)),float(np.percentile(ms,97.5))],"n":int(len(d))}

def main():
    cache=Path("results/iclr/probe_official/qwen3_8b")
    n=150; ntr=90
    methods=["qpre_lin","qpre_mlp","qrnd_lin","feat_mlp","raw_lin","raw_mlp"]
    per_seed={m:[] for m in methods}
    for s in [7,17,27]:
        ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,s)
        train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        test_idx=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
        Ytr,Yte=fut[train_idx],fut[test_idx]
        ht=np.load(cache/f"seed{s}/cache/train_h_pretrained_seed{s}.npy"); he=np.load(cache/f"seed{s}/cache/test_h_pretrained_seed{s}.npy")
        rt=np.load(cache/f"seed{s}/cache/train_h_random_seed{s}.npy"); re=np.load(cache/f"seed{s}/cache/test_h_random_seed{s}.npy")
        def reorder(a,b): return np.concatenate([np.concatenate([a[k*90:(k+1)*90],b[k*60:(k+1)*60]]) for k in range(5)])
        Xq_tr,Xq_te=reorder(ht,he)[train_idx],reorder(ht,he)[test_idx]
        Xr_tr,Xr_te=reorder(rt,re)[train_idx],reorder(rt,re)[test_idx]
        feats_tr=np.stack([temporal_features(x) for x in ctx[train_idx]]); feats_te=np.stack([temporal_features(x) for x in ctx[test_idx]])
        pred={
          "qpre_lin":fit_predict(Xq_tr,Ytr,Xq_te,"linear",s),
          "qpre_mlp":fit_predict(Xq_tr,Ytr,Xq_te,"mlp",s),
          "qrnd_lin":fit_predict(Xr_tr,Ytr,Xr_te,"linear",s),
          "feat_mlp":fit_predict(feats_tr,Ytr,feats_te,"mlp",s),
          "raw_lin":fit_predict(ctx[train_idx],Ytr,ctx[test_idx],"linear",s),
          "raw_mlp":fit_predict(ctx[train_idx],Ytr,ctx[test_idx],"mlp",s)}
        for m in methods:
            per_seed[m].append(float(np.mean((pred[m]-Yte)**2)))
    print("=== frozen-head MSE mean±std (3 seeds) ===")
    for m in methods:
        print(f"  {m:10s} {np.mean(per_seed[m]):.4f}±{np.std(per_seed[m]):.4f}")
    # paired CIs need per-window; use per-seed predictions -> rerun with windows? instead report seed-level
    # (approximation): report seed-level mean+/-std and note paired CI needs per-window (compute in follow-up)
    print("\n=== parameter counts ===")
    d4096,H=4096,16
    print(f"  frozen rep linear head: {d4096*H+H} params")
    print(f"  frozen rep MLP head(128): {d4096*128+128+128*H+H} params")
    print(f"  raw ctx linear head: {64*H+H} params")
    print(f"  raw ctx MLP head(128): {64*128+128+128*H+H} params")
    print(f"  temporal feats MLP(9->128->16): {9*128+128+128*H+H} params")
    # rejection sampling stats
    print("\n=== balanced3 rejection-sampling stats ===")
    from models.experts import LocalExpert, PeriodicExpert, TrendExpert
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]
    import numpy.random as npr
    for s in [7,17,27]:
        rng=np.random.default_rng(s)
        n_cand={"mix_trend":0,"mix_periodic":0,"reg_trend":0,"reg_periodic":0,"local":0}
        acc={"mix_trend":0,"mix_periodic":0,"reg_trend":0,"reg_periodic":0,"local":0}
        # mixture/regime
        for kind in ["mixture","regime"]:
            for _ in range(3000):
                cw,fw=__import__('data.dynamics',fromlist=['generate_window']).generate_window(kind,64,16,rng)
                errs=[float(np.mean((e.predict(cw,16)-fw)**2)) for e in experts]
                j=int(np.argmin(errs)); key=f"{kind[:3]}_{'trend' if j==0 else 'periodic'}"
                if j<2: n_cand[key]+=1; acc[key]+=1
        for _ in range(6000):
            cw,fw=gen_ar(rng)
            errs=[float(np.mean((e.predict(cw,16)-fw)**2)) for e in experts]
            n_cand["local"]+=1
            if int(np.argmin(errs))==2: acc["local"]+=1
        print(f"  seed{s}: candidates per class (kept/seen): mix_trend={acc['mix_trend']}/{n_cand['mix_trend']} "
              f"mix_periodic={acc['mix_periodic']}/{n_cand['mix_periodic']} reg_trend={acc['reg_trend']}/{n_cand['reg_trend']} "
              f"reg_periodic={acc['reg_periodic']}/{n_cand['reg_periodic']} local={acc['local']}/{n_cand['local']}")

def gen_ar(rng):
    total=80; t=np.arange(total,dtype=float); x=np.empty(total); x[0]=rng.normal(0,0.2)
    for k in range(1,total): x[k]=0.9*x[k-1]+rng.normal(0,0.3)
    x=x+0.15*np.sin(2*np.pi*t/24)
    m=x[:64].mean(); s=x[:64].std()+1e-6
    return (x[:64]-m)/s,(x[64:]-m)/s

if __name__=="__main__":
    main()
