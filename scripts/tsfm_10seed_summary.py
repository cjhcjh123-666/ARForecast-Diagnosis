"""10-seed Qwen3-8B headline routing summary (family-label protocol).

Uses stored features: seeds 7,17,27 from qwen_feats/, seeds 37..97 from
qwen_feats_10s/. Reports per-seed balanced acc (bal3 & bal3n), seed-wise delta,
mean+-std, and a 10000-resample seed bootstrap 95% CI of the mean delta.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.linear_probe import softmax_predict, softmax_regression
SEEDS=[7,17,27,37,47,57,67,77,87,97]
def balacc(pred,oracle):
    cm=np.zeros((3,3),int)
    for t,p in zip(oracle,pred):
        if t<3 and p<3: cm[t,p]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    return float(rec.mean())
def main():
    rows=[]
    for s in SEEDS:
        for ini in ["pretrained","random"]:
            src="qwen_feats_10s" if s in [37,47,57,67,77,87,97] else "qwen_feats"
            z=np.load(f"{src}/qwen3_8b_base_{ini}_s{s}.npz")
            Hc,Hb3,Hb3n=z["h_clean"],z["h_bal3"],z["h_bal3n"]; fam=z["clean_family"]
            _,p3=softmax_predict(Hb3,*softmax_regression(Hc,fam,n_iter=2000))
            _,p3n=softmax_predict(Hb3n,*softmax_regression(Hc,fam,n_iter=2000))
            rows.append((s,ini,"bal3",balacc(p3,z["bal3_oracle"])))
            rows.append((s,ini,"bal3n",balacc(p3n,z["bal3n_oracle"])))
    # per seed delta
    print("seed  dP_bal3  dP_bal3n")
    d3=[];d3n=[]
    for s in SEEDS:
        pb3=next(r[3] for r in rows if r[0]==s and r[1]=="pretrained" and r[2]=="bal3")
        rb3=next(r[3] for r in rows if r[0]==s and r[1]=="random" and r[2]=="bal3")
        pb3n=next(r[3] for r in rows if r[0]==s and r[1]=="pretrained" and r[2]=="bal3n")
        rb3n=next(r[3] for r in rows if r[0]==s and r[1]=="random" and r[2]=="bal3n")
        d3.append(pb3-rb3); d3n.append(pb3n-rb3n)
        print(f"{s:4d}  {pb3-rb3:+.4f}   {pb3n-rb3n:+.4f}")
    d3=np.array(d3); d3n=np.array(d3n)
    def ci(d):
        rng=np.random.default_rng(0); bs=np.array([np.mean(rng.choice(d,len(d),replace=True)) for _ in range(10000)])
        return np.mean(d),np.std(d),np.percentile(bs,2.5),np.percentile(bs,97.5)
    m,sd,lo,hi=ci(d3); m2,sd2,lo2,hi2=ci(d3n)
    print(f"\nbal3 : mean {m:.4f} +- {sd:.4f} (seed std) | seed-bootstrap 95% CI [{lo:.4f},{hi:.4f}]")
    print(f"bal3n: mean {m2:.4f} +- {sd2:.4f} | CI [{lo2:.4f},{hi2:.4f}]")
    np.save("results/iclr/tsfm_deliver_v2/seed10_delta.npy", np.stack([d3,d3n]))
if __name__=="__main__":
    main()
