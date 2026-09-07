"""Qwen size sweep on routing (family-label protocol, native dims)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.linear_probe import softmax_predict, softmax_regression
SEEDS=[7,17,27]
SIZES={"qwen3_0.6b":"qwen3_0.6b","qwen3_1.7b":"qwen3_1.7b","qwen3_8b_base":"qwen3_8b_base"}
DIM={"qwen3_0.6b":1024,"qwen3_1.7b":2048,"qwen3_8b_base":4096}
def balacc(pred,oracle):
    cm=np.zeros((3,3),int)
    for t,pp in zip(oracle,pred):
        if t<3 and pp<3: cm[t,pp]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    return float(rec.mean())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--featdir",type=Path,default=Path("qwen_feats")); a=ap.parse_args()
    print("model   dim  P_bal3 R_bal3 dP_bal3  P_bal3n R_bal3n dP_bal3n")
    for tag in SIZES:
        acc={"pretrained":[[],[]],"random":[[],[]]}
        for s in SEEDS:
            for ini in acc:
                z=np.load(a.featdir/f"{tag}_{ini}_s{s}.npz")
                Hc,Hb3,Hb3n=z["h_clean"],z["h_bal3"],z["h_bal3n"]; fam=z["clean_family"]
                _,p3=softmax_predict(Hb3,*softmax_regression(Hc,fam,n_iter=2000))
                _,p3n=softmax_predict(Hb3n,*softmax_regression(Hc,fam,n_iter=2000))
                acc[ini][0].append(balacc(p3,z["bal3_oracle"])); acc[ini][1].append(balacc(p3n,z["bal3n_oracle"]))
        P3=np.mean(acc["pretrained"][0]);R3=np.mean(acc["random"][0]);P3n=np.mean(acc["pretrained"][1]);R3n=np.mean(acc["random"][1])
        print(f"{tag:<15} {DIM[tag]:5d}  {P3:.3f} {R3:.3f} {P3-R3:+.3f}    {P3n:.3f} {R3n:.3f} {P3n-R3n:+.3f}")
if __name__=="__main__": main()
