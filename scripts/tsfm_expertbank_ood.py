"""OOD oracle robustness under an expanded 5-expert bank (CPU, stored preds).

For each OOD window, the winner expert is recomputed over 5 experts:
  trend  : Trend, TrendSeasonal
  periodic: Periodic
  local  : AR(5), AR(10)
Winner FAMILY = family of the argmin expert. Reports:
  - winner-family flip rate 3-expert -> 5-expert oracle
  - per-model balanced accuracy of the family-label router against the
    5-expert winner-family labels (3-class), pretrained | random | delta
    (same windows as bal3/bal3n; per-seed then mean over seeds)
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.tsfm_expertbank_stability import TrendSeasonalExpert, ARExpert, ORIG

SEEDS=[7,17,27]
MODELS=["qwen3_8b_base","chronos_t5-small","chronos_t5-base","chronos_bolt-small",
        "timesfm_2.5-200m","moment-1-large","moirai-1.1-R-small"]
FAM=["trend","periodic","local"]
EXP5=[TrendExpert(),TrendSeasonalExpert(),PeriodicExpert(),LocalExpert(),ARExpert(10,ridge=1.0)]
E5FAM=[0,0,1,2,2]

def balacc(pred,oracle):
    n=len(pred); cm=np.zeros((3,3),int)
    for t,pp in zip(oracle,pred):
        if t<3 and pp<3: cm[t,pp]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    return float(rec.mean())

rows=[]
out=[]
for tag in ["bal3","bal3n"]:
    for s in SEEDS:
        z=np.load(f"results/iclr/e4_balanced3/windows/{tag}_s{s}.npz") if tag=="bal3" else np.load(f"results/iclr/e4_balanced3_natural/windows/{tag}_s{s}.npz")
        ctx,fut,orac3=z["ctx"],z["fut"],z["oracle"]
        n=len(ctx)
        # 3-expert and 5-expert winner family
        e3=np.zeros((n,3)); e5=np.zeros((n,5))
        for i in range(n):
            for j,e in enumerate(ORIG): e3[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
            for j,e in enumerate(EXP5): e5[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
        fam5=np.array(E5FAM)[e5.argmin(1)]
        flip=float(np.mean(fam5!=orac3))
        print(f"[{tag}/s{s}] winner-family flip 3->5 expert oracle: {flip:.3f}")
        for m in MODELS:
            for ini in ["pretrained","random"]:
                zz=np.load(f"results/iclr/tsfm_deliver_v2/perwindow/pw_{m}_{ini}_s{s}.npz")
                pred=zz[f"c_{tag}_pred"]
                acc=balacc(pred,fam5)
                rows.append({"model":m,"init":ini,"seed":s,"test_set":tag,"n":n,"acc_vs_5exp_oracle":round(acc,4)})
        out.append((tag,s,flip))
# summary deltas (mean over seeds) vs 3-expert oracle (recompute from metrics.csv for reference)
def mean_delta(key):
    d={}
    for m in MODELS:
        vals=[]
        for ini in ["pretrained","random"]:
            vv=[r[key] for r in rows if r["model"]==m and r["init"]==ini]
            vals.append(float(np.mean(vv)) if vv else float("nan"))
        d[m]=vals
    return d
print("\n=== balanced accuracy vs 5-expert OOD oracle family (mean over seeds; P | R | delta) ===")
D=mean_delta("acc_vs_5exp_oracle")
for m in MODELS:
    print(f"{m:<20} {D[m][0]:.3f} | {D[m][1]:.3f} | {D[m][0]-D[m][1]:+.3f}")
# save rows
with open("results/iclr/tsfm_deliver_v2/expertbank_ood_router_acc.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved expertbank_ood_router_acc.csv")
