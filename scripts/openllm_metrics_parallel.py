"""Parallel (multiprocessing) recomputation of open-LM metrics from features_full."""
from __future__ import annotations
import csv, sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
REPO="/9950backfile/chenjiahui/ARForecast-Diagnosis"
sys.path.insert(0,REPO)
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import torch, torch.nn as nn
torch.set_num_threads(1)
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
SEEDS=[7,17,27]; MODELS=["gemma2_2b","gemma2_9b","llama32_3b"]
FF=Path(REPO)/"results/open_llm_suite/features_full"; OUT=Path(REPO)/"results/open_llm_suite/raw"
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
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd); Xn=(Xt-mu)/sd
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
def job(args):
    model,seed=args
    n=150;ntr=90
    ctx,fut_ref,kk=build_labeled_windows(KINDS,64,16,n,seed)
    tr=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
    te=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
    clean270=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
    Ec=np.zeros((len(clean270),3))
    for j,e in enumerate(ORIG):
        for i,idx in enumerate(clean270): Ec[i,j]=float(np.mean((e.predict(ctx[idx],16)-fut_ref[idx])**2))
    oclean=Ec.argmin(1)
    out=[]
    for init in ["pretrained","random"]:
        f=FF/f"full_{model}_{init}_s{seed}.npz"
        if not f.is_file(): continue
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
            _,pf=softmax_predict(Hb,W,mu,sd); _,po=softmax_predict(Hb,Wo,muo,sdo)
            pm=mlp_clf(H[clean270],kind[clean270],Hb,seed)
            r[f"C_family_{tag}"]=round(balacc(pf,bor),4)
            r[f"C2_oracle_{tag}"]=round(balacc(po,bor),4)
            r[f"R_mlp_{tag}"]=round(balacc(pm,bor),4)
        out.append(r)
    return out
if __name__=="__main__":
    jobs=[(m,s) for m in MODELS for s in SEEDS]
    with Pool(min(9,len(jobs))) as pool:
        res=[r for sub in pool.map(job,jobs) for r in sub]
    f=OUT/"metrics_from_features_full.csv"
    fields=sorted({k for r in res for k in r})
    with open(f,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        for r in res: w.writerow(r)
    print("wrote",f,len(res),"rows")
    for model in MODELS:
        for init in ["pretrained","random"]:
            def m(k):
                v=[r[k] for r in res if r["model"]==model and r["init"]==init and r.get(k) is not None]
                return float(np.mean(v)) if v else float('nan')
            print(f"{model:<12}{init:<11} A={m('A_recog'):.4f} B_lin={m('B_linear'):.4f} B_mlp={m('B_mlp'):.4f} fam3={m('C_family_bal3'):.4f} fam3n={m('C_family_bal3n'):.4f} or3={m('C2_oracle_bal3'):.4f} MLP3={m('R_mlp_bal3'):.4f}")
