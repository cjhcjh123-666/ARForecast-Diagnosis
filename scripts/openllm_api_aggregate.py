"""Aggregate API-generated forecasts (resumable inputs) into paper-ready CSVs.

For each (model, protocol, seed): parse rate; native MSE over parsed windows;
persistence / best-fixed / context-router / oracle MSE on the SAME windows;
ratios. Partial JSONL is fine (only completed windows counted).
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from data.dynamics import build_labeled_windows
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]
RAW=REPO/"results/open_llm_suite/api_raw"
OUT=REPO/"results/open_llm_suite"

def windows(seed,n=40):
    ctx,fut,kk=build_labeled_windows(KINDS,64,16,150,seed)
    n150=150;ntr=90
    te=np.concatenate([np.arange(k*n150+ntr,(k+1)*n150) for k in range(5)])
    return ctx[te][:n],fut[te][:n],kk[te][:n]

def refs(seed,n=40):
    ctx,fut,kind=windows(seed,n)
    clean_ctx,_,clean_kind=build_labeled_windows(KINDS,64,16,150,seed)
    tr=np.concatenate([np.arange(k*150,k*150+90) for k in range(3)])
    Htr=np.stack([temporal_features(x) for x in clean_ctx[tr]])
    E=np.zeros((len(ctx),3))
    for j,e in enumerate(EXPERTS):
        for i in range(len(ctx)): E[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
    pers=np.array([float(np.mean((np.full(16,ctx[i,-1])-fut[i])**2)) for i in range(len(ctx))])
    bf=int(E.mean(0).argmin()); best=E[:,bf]; oracle=E.min(1)
    te=np.stack([temporal_features(x) for x in ctx])
    _,pred=softmax_predict(te,*softmax_regression(Htr,clean_kind[tr],n_iter=2000))
    ctxrouter=E[np.arange(len(ctx)),pred]
    return fut,dict(persistence=pers,best_fixed=best,oracle=oracle,context_router=ctxrouter)

def main():
    rows=[]
    for jf in sorted(RAW.glob("*/*.jsonl")):
        model=jf.parent.name.replace("__","/")
        proto,seed=jf.stem.split("_s"); seed=int(seed)
        recs=[json.loads(l) for l in jf.read_text(errors="ignore").splitlines() if l.strip()]
        done=[r for r in recs if r.get("ok")]
        if not done: continue
        fut,ref=refs(seed,len(windows(seed)[0]))
        preds=[]; idx=[]
        for r in done:
            if r.get("parse_ok") and r.get("parsed") is not None:
                preds.append(np.asarray(r["parsed"],float)); idx.append(r["window_id"])
        n=len(done); npar=len(preds)
        mse=np.mean([np.mean((preds[k]-fut[idx[k]])**2) for k in range(npar)]) if npar else float("nan")
        row=dict(model=model,protocol=proto,seed=seed,n_windows=n,parse_rate=round(npar/max(n,1),4),
                 native_mse=round(float(mse),4) if mse==mse else "",
                 persistence_mse=round(float(ref["persistence"][idx].mean()),4) if npar else "",
                 best_fixed_mse=round(float(ref["best_fixed"][idx].mean()),4) if npar else "",
                 context_router_mse=round(float(ref["context_router"][idx].mean()),4) if npar else "",
                 oracle_mse=round(float(ref["oracle"][idx].mean()),4) if npar else "")
        if npar and mse==mse and row["oracle_mse"]:
            row["native_over_oracle"]=round(float(mse)/row["oracle_mse"],3)
            row["native_over_persistence"]=round(float(mse)/row["persistence_mse"],3)
            row["native_over_bestfixed"]=round(float(mse)/row["best_fixed_mse"],3)
            row["native_over_context_router"]=round(float(mse)/row["context_router_mse"],3)
        rows.append(row)
    if rows:
        f=OUT/("api_generation_"+rows[0]["protocol"]+".csv")
        fields=sorted({k for r in rows for k in r})
        with open(f,"w",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
            for r in rows: w.writerow(r)
        print("wrote",f,len(rows),"rows")
        for r in rows[:15]: print({k:r[k] for k in ["model","protocol","parse_rate","native_mse","native_over_oracle"] if k in r})
if __name__=="__main__": main()
