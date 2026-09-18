"""Final consistent metrics for open LMs from features_full (CPU only).

Computes A (clean 5-way), B (linear + MLP readout), C (family-label routing),
C2 (oracle-label routing; clean contexts rebuilt deterministically), R (MLP64 router).
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
SEEDS=[7,17,27]; MODELS=["gemma2_2b","gemma2_9b","llama32_3b"]
FF=Path("results/open_llm_suite/features_full"); OUT=Path("results/open_llm_suite/raw"); OUT.mkdir(parents=True,exist_ok=True)
ORIG=[TrendExpert(),PeriodicExpert(),LocalExpert()]
def balacc(pred,oracle):
    cm=np.zeros((3,3),int)
    for t,p in zip(oracle,pred):
        if t<3 and p<3: cm[t,p]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    return float(rec.mean())
def mlp_clf(Xtr,ytr,Xte,seed,h=64,steps=600,lr=1e-3,wd=1e-2):
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(Xtr.shape[1],h),nn.ReLU(),nn.Linear(h,int(ytr.max())+1)).double()
    Xt=torch.from_numpy(Xtr).double(); yt=torch.from_numpy(ytr).long()
    mu=Xt.mean(0); sd=Xt.std(0)+1e-9
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    Xn=(Xt-mu)/sd
    for _ in range(steps):
        opt.zero_grad(); nn.functional.cross_entropy(net(Xn),yt).backward(); opt.step()
    with torch.no_grad(): return net((torch.from_numpy(Xte).double()-mu)/sd).argmax(1).numpy()
def head_linear(Xtr,Ytr,Xte,seed,wd=0.01,epochs=300):
    torch.manual_seed(seed); net=nn.Linear(Xtr.shape[1],Ytr.shape[1])
    Xt=torch.from_numpy(Xtr).float(); Yt=torch.from_numpy(Ytr).float()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(); nn.functional.mse_loss(net(Xt),Yt).backward(); opt.step()
    with torch.no_grad(): return net(torch.from_numpy(Xte).float()).numpy(), Xtr.shape[1]*Ytr.shape[1]+Ytr.shape[1]
def head_mlp(Xtr,Ytr,Xte,seed,h=256,steps=400,lr=1e-3,wd=1e-4):
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(Xtr.shape[1],h),nn.ReLU(),nn.Linear(h,Ytr.shape[1]))
    Xt=torch.from_numpy(Xtr).float(); Yt=torch.from_numpy(Ytr).float()
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); nn.functional.mse_loss(net(Xt),Yt).backward(); opt.step()
    with torch.no_grad(): return net(torch.from_numpy(Xte).float()).numpy(), sum(p.numel() for p in net.parameters())
rows=[]
for model in MODELS:
    for seed in SEEDS:
        n=150;ntr=90
        ctx,fut_ref,kk=build_labeled_windows(KINDS,64,16,n,seed)
        tr=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        te=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
        clean270=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
        Ec=np.zeros((len(clean270),3))
        for j,e in enumerate(ORIG):
            for i,idx in enumerate(clean270): Ec[i,j]=float(np.mean((e.predict(ctx[idx],16)-fut_ref[idx])**2))
        oclean=Ec.argmin(1)
        for init in ["pretrained","random"]:
            f=FF/f"full_{model}_{init}_s{seed}.npz"
            if not f.is_file(): print("missing",f); continue
            z=np.load(f); H=z["h_clean"]; kind=z["clean_kind"]; fut=z["fut"]
            r={"model":model,"init":init,"seed":seed}
            _,p=softmax_predict(H[te],*softmax_regression(H[tr],kind[tr],n_iter=2000))
            r["A_recog"]=round(float((p==kind[te]).mean()),4)
            pred,hp=head_linear(H[tr],fut[tr],H[te],seed)
            r["B_linear"]=round(float(np.mean((pred-fut[te])**2)),4); r["B_head_params"]=hp
            predm,hpm=head_mlp(H[tr],fut[tr],H[te],seed)
            r["B_mlp"]=round(float(np.mean((predm-fut[te])**2)),4); r["B_mlp_params"]=hpm
            W,mu,sd=softmax_regression(H[clean270],kind[clean270],n_iter=2000)
            Wo,muo,sdo=softmax_regression(H[clean270],oclean,n_iter=2000)
            for tag,hk,ok in [("bal3","h_bal3","bal3_oracle"),("bal3n","h_bal3n","bal3n_oracle")]:
                Hb=z[hk]; bor=z[ok]
                _,pf=softmax_predict(Hb,W,mu,sd)
                _,po=softmax_predict(Hb,Wo,muo,sdo)
                pm=mlp_clf(H[clean270],kind[clean270],Hb,seed)
                r[f"C_family_{tag}"]=round(balacc(pf,bor),4)
                r[f"C2_oracle_{tag}"]=round(balacc(po,bor),4)
                r[f"R_mlp_{tag}"]=round(balacc(pm,bor),4)
            rows.append(r)
        print(f"[{model}/s{seed}] done",flush=True)
f=OUT/"metrics_from_features_full.csv"
fields=sorted({k for r in rows for k in r})
with open(f,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
    for r in rows: w.writerow(r)
print("wrote",f,len(rows),"rows")
print(f"\n{'model':<12}{'init':<11}{'A':>7}{'B_lin':>8}{'B_mlp':>8}{'fam3':>7}{'fam3n':>7}{'or3':>7}{'MLP3':>7}")
for model in MODELS:
    for init in ["pretrained","random"]:
        def m(k):
            v=[r.get(k) for r in rows if r["model"]==model and r["init"]==init and r.get(k) is not None]
            return float(np.mean(v)) if v else float('nan')
        print(f"{model:<12}{init:<11}{m('A_recog'):7.3f}{m('B_linear'):8.3f}{m('B_mlp'):8.3f}{m('C_family_bal3'):7.3f}{m('C_family_bal3n'):7.3f}{m('C2_oracle_bal3'):7.3f}{m('R_mlp_bal3'):7.3f}")
