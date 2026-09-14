"""Table D: real-world zero-shot routing attribution (CPU, from stored features)."""
from __future__ import annotations
import csv, math, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
RF=REPO/"results/open_llm_suite/realworld_features"
OUT=REPO/"results/open_llm_suite/raw"
WD=REPO/"results/iclr/multi_dataset_router/qwen3_8b_official/windows_qwen3_8b_official_s7"
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]
rng=np.random.default_rng(0)
# shared feature-router training material (synthetic clean270)
ctx0,_,kk0=build_labeled_windows(KINDS,64,16,150,7)
C270=np.concatenate([np.arange(k*150,k*150+90) for k in range(3)])
FTR=np.stack([temporal_features(x) for x in ctx0[C270]]); KTR=kk0[C270]
def errs(ctx,fut):
    E=np.zeros((len(ctx),3))
    for j,e in enumerate(EXPERTS):
        for i in range(len(ctx)): E[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
    return E
def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); q=np.empty(n)
    prev=1.0
    for rank in range(n-1,-1,-1):
        i=o[rank]; prev=min(prev,p[i]*n/(rank+1)); q[i]=prev
    return q
def parse_model(fname):
    core=fname[len("rw_"):] if fname.startswith("rw_") else fname
    for init in ["pretrained","random"]:
        if core.endswith("_"+init+".npz"): return core[:-len(init)-4-1]
    return None
models=sorted({m for m in (parse_model(f.name) for f in RF.glob("rw_*_pretrained.npz")) if m})
rows=[]; boot_rows=[]
for model in models:
    # The clean-primitive router depends only on (model, init); training it once and
    # reusing it across the 15 datasets is numerically identical and ~15x faster.
    routers={}
    c270=np.concatenate([np.arange(k*150,k*150+90) for k in range(3)])
    for f_ in sorted(WD.glob("*_windows.npz")):
        ds=f_.name.replace("_windows.npz","")
        z=np.load(f_); ctx,fut=z["ctx"],z["fut"]
        E=errs(ctx,fut)
        pers=np.array([float(np.mean((np.full(16,ctx[i,-1])-fut[i])**2)) for i in range(len(ctx))])
        bf=int(E.mean(0).argmin()); best=E[:,bf]; oracle=E.min(1)
        feats=np.stack([temporal_features(x) for x in ctx])
        _,pf=softmax_predict(feats,*softmax_regression(FTR,KTR,n_iter=2000))
        route={}
        for init in ["pretrained","random"]:
            fc=RF/f"rw_{model}_{init}.npz"; fd=RF/f"rw_{model}_{init}_{ds}.npz"
            if not (fc.is_file() and fd.is_file()): continue
            if init not in routers:
                zc=np.load(fc); H=zc["h_clean"]; kind=zc["clean_kind"]
                routers[init]=softmax_regression(H[c270],kind[c270],n_iter=2000)
            h=np.load(fd)["h"]
            _,pred=softmax_predict(h,*routers[init])
            route[init]=E[np.arange(len(ctx)),pred]
        rec={"model":model,"dataset":ds,"n":len(ctx),
             "persistence":round(float(pers.mean()),4),"best_fixed":round(float(best.mean()),4),
             "oracle":round(float(oracle.mean()),4),"feature_router":round(float(np.mean(E[np.arange(len(ctx)),pf])),4)}
        if "pretrained" in route: rec["pretrained"]=round(float(route["pretrained"].mean()),4)
        if "random" in route: rec["random"]=round(float(route["random"].mean()),4)
        rows.append(rec)
        if "pretrained" in route and "random" in route:
            for a,b,name in [(route["pretrained"],route["random"],"P-R"),(route["pretrained"],E[np.arange(len(ctx)),pf],"P-feature")]:
                d=np.asarray(a)-np.asarray(b)
                bs=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(2000)])
                boot_rows.append(dict(model=model,dataset=ds,contrast=name,delta=round(float(d.mean()),4),
                                      lo=round(float(np.percentile(bs,2.5)),4),hi=round(float(np.percentile(bs,97.5)),4)))
    print("done",model,flush=True)
# BH-FDR per contrast
for name in ["P-R","P-feature"]:
    sub=[r for r in boot_rows if r["contrast"]==name]
    if not sub: continue
    # two-sided nominal p approximated by whether CI crosses 0 -> use normal approx on bootstrap sd
    ps=[]
    for r in sub:
        sd=(r["hi"]-r["lo"])/3.92
        z=abs(r["delta"])/(sd+1e-12); ps.append(float(2*(1-0.5*(1+math.erf(z/math.sqrt(2))))))
    qs=bh(ps)
    for r,q in zip(sub,qs): r["q_value"]=round(float(q),4)
f=OUT/"tableD_realworld.csv"
fields=["model","dataset","n","pretrained","random","feature_router","persistence","best_fixed","oracle"]
with open(f,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
    for r in rows: w.writerow({k:r.get(k,"") for k in fields})
f2=OUT/"tableD_realworld_stats.csv"
f2f=["model","dataset","contrast","delta","lo","hi","q_value"]
with open(f2,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=f2f); w.writeheader()
    for r in boot_rows: w.writerow({k:r.get(k,"") for k in f2f})
print("wrote",f,"and",f2)
for model in models:
    sub=[r for r in rows if r["model"]==model]
    if not sub: continue
    p=np.mean([r.get("pretrained",float("nan")) for r in sub]); rr=np.mean([r.get("random",float("nan")) for r in sub])
    ft=np.mean([r["feature_router"] for r in sub]); orc=np.mean([r["oracle"] for r in sub])
    wins=sum(1 for r in sub if r.get("pretrained",9e9)<r.get("random",9e9))
    print(f"{model:<14} n_ds={len(sub)} mean P={p:.4f} R={rr:.4f} feature={ft:.4f} oracle={orc:.4f} | P<R on {wins}/{len(sub)}")
