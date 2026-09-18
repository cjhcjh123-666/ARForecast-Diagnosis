"""Table B for open LMs (CPU, from stored clean + OOD features).

Per (model, init, seed):
  family-label 3-expert balanced acc (bal3/bal3n)
  oracle-label 3-expert balanced acc
  family-label router vs 5-expert winner-family
  MLP64 router (recomputed)
  margin bins (best/second of 3 experts) -> pre/rand accuracy
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.tsfm_expertbank_stability import TrendSeasonalExpert, ARExpert
KINDS=["trend","periodic","local","mixture","regime"]
ORIG=[TrendExpert(),PeriodicExpert(),LocalExpert()]
EXP5=[TrendExpert(),TrendSeasonalExpert(),PeriodicExpert(),LocalExpert(),ARExpert(10,ridge=1.0)]
E5FAM=np.array([0,0,1,2,2])
SEEDS=[7,17,27]
MODELS=["gemma2_2b","gemma2_9b","llama32_3b"]
FEAT=Path("results/open_llm_suite/features")
OUT=Path("results/open_llm_suite/raw"); OUT.mkdir(parents=True,exist_ok=True)

def balacc(pred,oracle):
    cm=np.zeros((3,3),int)
    for t,p in zip(oracle,pred):
        if t<3 and p<3: cm[t,p]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    return float(rec.mean())

def errs(ctx,fut,exp):
    E=np.zeros((len(ctx),len(exp)))
    for j,e in enumerate(exp):
        for i in range(len(ctx)): E[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
    return E

rows=[]
for model in MODELS:
    for seed in SEEDS:
        n=150;ntr=90
        ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,seed)
        clean270=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
        Ec=errs(ctx[clean270],fut[clean270],ORIG); oclean=Ec.argmin(1)
        for init in ["pretrained","random"]:
            fc=FEAT/f"pw_{model}_{init}_s{seed}.npz"; fo=FEAT/f"ood_{model}_{init}_s{seed}.npz"
            if not (fc.is_file() and fo.is_file()): print("missing",fc,fo); continue
            H=np.load(fc)["H"]; Hc=H[clean270]
            zo=np.load(fo)
            W,mu,sd=softmax_regression(Hc,kk[clean270],n_iter=2000)
            Wo,muo,sdo=softmax_regression(Hc,oclean,n_iter=2000)
            for tag,key in [("bal3","h_bal3"),("bal3n","h_bal3n")]:
                Hb=zo[key]; bor=zo[f"{tag}_oracle"]
                z=np.load(f"results/iclr/e4_balanced3/windows/bal3_s{seed}.npz") if tag=="bal3" else np.load(f"results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz")
                bctx,bfut=z["ctx"],z["fut"]
                _,pf=softmax_predict(Hb,W,mu,sd)
                _,po=softmax_predict(Hb,Wo,muo,sdo)
                E5=errs(bctx,bfut,EXP5); fam5=E5FAM[E5.argmin(1)]
                E3=errs(bctx,bfut,ORIG)
                marg=E3[np.arange(len(bctx)),np.argsort(E3,1)[:,1]]/(E3.min(1)+1e-9)
                rows.append(dict(model=model,init=init,seed=seed,test=tag,
                                 family3=round(balacc(pf,bor),4),
                                 oracle3=round(balacc(po,bor),4),
                                 family5=round(balacc(pf,fam5),4),
                                 margin_mean=round(float(marg.mean()),4)))
            print(f"[{model}/{init}/s{seed}] done",flush=True)
f=OUT/"tableB_openllm.csv"
fields=sorted({k for r in rows for k in r})
with open(f,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
    for r in rows: w.writerow(r)
print("wrote",f,len(rows),"rows")
# summary
for model in MODELS:
    for init in ["pretrained","random"]:
        def m(k):
            v=[r[k] for r in rows if r["model"]==model and r["init"]==init and r["test"]=="bal3"]
            return np.mean(v) if v else float('nan')
        print(f"{model:<12}{init:<11} family3={m('family3'):.3f} oracle3={m('oracle3'):.3f} family5={m('family5'):.3f}")
