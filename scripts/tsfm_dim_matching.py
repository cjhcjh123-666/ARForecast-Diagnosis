"""Dimension-matched Qwen routing (family-label protocol).

For each seed: fit PCA or seeded random projection on the CLEAN 270 train
features only, project clean+OOD, retrain the family-label softmax router at
that dimension, evaluate on bal3/bal3n. Reports pretrained|random|delta per
(dim, method), mean over seeds.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression
SEEDS=[7,17,27]
DIMS=[384,512,768,1024,2048,4096]
def balacc(pred,oracle):
    n=len(pred);cm=np.zeros((3,3),int)
    for t,pp in zip(oracle,pred):
        if t<3 and pp<3: cm[t,pp]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    return float(rec.mean())

def reduce(Xtr,Xte,d,method,seed):
    n=Xtr.shape[0]
    if d>=Xtr.shape[1]: return Xtr,Xte
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sd; Zt=(Xte-mu)/sd
    if method=="pca":
        U,S,Vt=np.linalg.svd(Z,full_matrices=False)
        P=Vt[:d].T
    else:
        rng=np.random.default_rng(seed)
        P=rng.normal(0,1.0/np.sqrt(d),(Xtr.shape[1],d))
    return Z@P, Zt@P

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--featdir", type=Path, default=Path("qwen_feats"))
    ap.add_argument("--methods", default="pca,rp")
    a=ap.parse_args(); methods=[m for m in a.methods.split(",") if m]
    print("dim method model P_bal3 R_bal3 dP_bal3 P_bal3n R_bal3n dP_bal3n")
    for d in DIMS:
        for method in methods:
            acc={"pretrained":[[],[]],"random":[[],[]]}  # [bal3,bal3n]
            for s in SEEDS:
                for ini in acc:
                    z=np.load(a.featdir/f"qwen3_8b_base_{ini}_s{s}.npz")
                    Hc,Hb3,Hb3n=z["h_clean"],z["h_bal3"],z["h_bal3n"]
                    fam=z["clean_family"]
                    Hcr,Hb3r=reduce(Hc,Hb3,d,method,seed=s)
                    Hcn,Hb3nr=reduce(Hc,Hb3n,d,method,seed=s)
                    _,p3=softmax_predict(Hb3r,*softmax_regression(Hcr,fam,n_iter=2000))
                    _,p3n=softmax_predict(Hb3nr,*softmax_regression(Hcr,fam,n_iter=2000))
                    acc[ini][0].append(balacc(p3,z["bal3_oracle"]))
                    acc[ini][1].append(balacc(p3n,z["bal3n_oracle"]))
            P3=np.mean(acc["pretrained"][0]);R3=np.mean(acc["random"][0])
            P3n=np.mean(acc["pretrained"][1]);R3n=np.mean(acc["random"][1])
            print(f"{d:5d} {method:4s} qwen {P3:.3f} {R3:.3f} {P3-R3:+.3f}   {P3n:.3f} {R3n:.3f} {P3n-R3n:+.3f}")
if __name__=="__main__":
    main()
