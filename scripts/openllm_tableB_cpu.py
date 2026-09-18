"""CPU-only Table B for open LMs from stored features (no GPU).

For each (model, init, seed): load stored H features (clean 750 windows),
retrain family-label router on clean270, evaluate on bal3/bal3n with
  3-expert oracle hit, 5-expert oracle hit (TrendSeasonal + AR10 added),
  and margin-binned Qwen-style deltas.  Also oracle-label router.
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
E5FAM=[0,0,1,2,2]
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
    for init in ["pretrained","random"]:
        for seed in SEEDS:
            f=FEAT/f"pw_{model}_{init}_s{seed}.npz"
            if not f.is_file(): continue
            H=np.load(f)["H"]
            n=150;ntr=90
            ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,seed)
            clean270=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
            Hc=H[clean270]
            W,mu,sd=softmax_regression(Hc,kk[clean270],n_iter=2000)
            Ec=errs(ctx[clean270],fut[clean270],ORIG); oclean=Ec.argmin(1)
            Wo,muo,sdo=softmax_regression(Hc,oclean,n_iter=2000)
            for tag,root in [("bal3",f"results/iclr/e4_balanced3/windows/bal3_s{seed}.npz"),
                             ("bal3n",f"results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz")]:
                if not Path(root).is_file(): continue
                z=np.load(root); bctx,bfut,bor=z["ctx"],z["fut"],z["oracle"]
                Hb=H[len(clean270):] if False else None
                # re-embed OOD features were not stored; recompute from model is impossible on CPU.
                pass
print("NOTE: OOD features are not stored (only clean750 H). Table B needs the runner to store H_bal3/H_bal3n.")
