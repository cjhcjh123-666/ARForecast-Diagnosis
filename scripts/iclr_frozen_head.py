"""ICLR P0-1: frozen representation -> numerical head (forecast).

Closes the Recognize-Don't-Generate loop: the SAME frozen hidden state used for
recognition is fed to a small trainable numerical head (linear / 2-layer MLP)
that regresses the H=16 future directly (MSE).  Compare:
  - frozen pretrained Qwen3-8B-Base + head
  - frozen random Qwen3-8B + head
  - temporal handcrafted features + head
  - raw-context MLP (no LLM)
All heads are trained only on the clean train windows and evaluated on the same
held-out synthetic windows used by the generation experiment.  If the frozen
pretrained representation + simple head forecasts well while native decoding
does not, the information is present and usable through a numerical interface.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from analysis.features import temporal_features
from data.dynamics import build_labeled_windows
KINDS=["trend","periodic","local","mixture","regime"]

def train_head(Xtr, Ytr, Xte, Yte, kind, seed, epochs=300, lr=1e-3, hidden=128, device="cpu"):
    torch.manual_seed(seed)
    d = Xtr.shape[1]; H = Ytr.shape[1]
    if kind=="linear":
        net = nn.Sequential(nn.Linear(d, H))
    elif kind=="mlp":
        net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, H))
    else:
        raise ValueError(kind)
    net = net.to(device)
    Xtr_t=torch.from_numpy(Xtr).float().to(device); Ytr_t=torch.from_numpy(Ytr).float().to(device)
    Xte_t=torch.from_numpy(Xte).float().to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr)
    lossf=nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad(); loss=lossf(net(Xtr_t),Ytr_t); loss.backward(); opt.step()
    with torch.no_grad(): pred=net(Xte_t).cpu().numpy()
    return float(np.mean((pred-Yte)**2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("results/iclr/probe_official/qwen3_8b"))
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--out", type=Path, default=Path("results/iclr/frozen_head/qwen3_8b_official.json"))
    a=ap.parse_args()
    seeds=[int(s) for s in a.seeds.split(",")]
    n=150; ntr=90
    report={"model":"official Qwen3-8B-Base","per_seed":{}}
    for s in seeds:
        ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,s)
        train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        test_idx=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
        Ytr=fut[train_idx]; Yte=fut[test_idx]
        # hidden in ORIGINAL window order
        h_tr=np.load(a.cache_root/f"seed{s}/cache/train_h_pretrained_seed{s}.npy")
        h_te=np.load(a.cache_root/f"seed{s}/cache/test_h_pretrained_seed{s}.npy")
        hr_tr=np.load(a.cache_root/f"seed{s}/cache/train_h_random_seed{s}.npy")
        hr_te=np.load(a.cache_root/f"seed{s}/cache/test_h_random_seed{s}.npy")
        def reorder(Htr,Hte): return np.concatenate([np.concatenate([Htr[k*90:(k+1)*90],Hte[k*60:(k+1)*60]]) for k in range(5)])
        Xq=reorder(h_tr,h_te); Xr=reorder(hr_tr,hr_te)
        feats=np.stack([temporal_features(x) for x in ctx])
        row={}
        row["frozen_pretrained_linear"]=train_head(Xq[train_idx],Ytr,Xq[test_idx],Yte,"linear",s)
        row["frozen_pretrained_mlp"]=train_head(Xq[train_idx],Ytr,Xq[test_idx],Yte,"mlp",s)
        row["frozen_random_linear"]=train_head(Xr[train_idx],Ytr,Xr[test_idx],Yte,"linear",s)
        row["frozen_random_mlp"]=train_head(Xr[train_idx],Ytr,Xr[test_idx],Yte,"mlp",s)
        row["temporal_features_mlp"]=train_head(feats[train_idx],Ytr,feats[test_idx],Yte,"mlp",s)
        row["raw_context_mlp"]=train_head(ctx[train_idx],Ytr,ctx[test_idx],Yte,"mlp",s)
        report["per_seed"][str(s)]=row
        print(f"[seed {s}] pre-lin={row['frozen_pretrained_linear']:.4f} pre-mlp={row['frozen_pretrained_mlp']:.4f} "
              f"rnd-lin={row['frozen_random_linear']:.4f} feat-mlp={row['temporal_features_mlp']:.4f} raw-mlp={row['raw_context_mlp']:.4f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
