"""Native (direct) numerical generation for open LMs: parse rate + MSE vs references."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts.openllm_suite import resolve_path, load_lm, generate_native
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]
OUT=REPO/"results/open_llm_suite/native"; OUT.mkdir(parents=True,exist_ok=True)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-key",required=True); ap.add_argument("--model-id",required=True)
    ap.add_argument("--path",default=None); ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--inits",default="pretrained,random"); ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--windows",type=int,default=40)
    a=ap.parse_args(); path=resolve_path(a.model_id,local=a.path)
    n=150;ntr=90
    ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,a.seed)
    per=max(1,a.windows//5)
    sel=np.concatenate([np.arange(k*n+ntr,k*n+ntr+per) for k in range(5)])[:a.windows]
    te=sel
    tr=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
    # references
    E=np.zeros((len(te),3))
    for j,e in enumerate(EXPERTS):
        for i,idx in enumerate(te): E[i,j]=float(np.mean((e.predict(ctx[idx],16)-fut[idx])**2))
    pers=np.array([float(np.mean((np.full(16,ctx[i,-1])-fut[i])**2)) for i in te])
    bf=int(E.mean(0).argmin()); best=E[:,bf]; oracle=E.min(1)
    ftr=np.stack([temporal_features(x) for x in ctx[tr]])
    feats=np.stack([temporal_features(x) for x in ctx[te]])
    _,pf=softmax_predict(feats,*softmax_regression(ftr,kk[tr],n_iter=2000))
    rows=[]
    for init in [x for x in a.inits.split(",") if x]:
        model,tok=load_lm(path,init,a.seed,a.device)
        try:
            preds,parses,raws=generate_native(model,tok,ctx[te],a.device)
        finally:
            del model; import torch; torch.cuda.empty_cache()
        ok=parses==1
        mse=float(np.mean([np.mean((preds[i]-fut[te][i])**2) for i in range(len(te)) if ok[i]])) if ok.any() else float("nan")
        rows.append(dict(model=a.model_key,init=init,seed=a.seed,n=len(te),parse_rate=round(float(ok.mean()),4),
                         native_mse=round(mse,4) if mse==mse else "",
                         oracle_mse=round(float(oracle.mean()),4),persistence_mse=round(float(pers.mean()),4),
                         best_fixed_mse=round(float(best.mean()),4),context_router_mse=round(float(E[np.arange(len(te)),pf].mean()),4)))
        np.savez(OUT/f"native_{a.model_key}_{init}_s{a.seed}.npz",pred=preds,parse=parses,fut=fut[te],raw=np.array(raws,dtype=object))
        print(rows[-1],flush=True)
    f=OUT/"native_local.csv"
    new=not f.exists()
    with open(f,"a",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        for r in rows: w.writerow(r)
if __name__=="__main__": main()
