"""Linear vs 2-layer MLP family-label router on stored Qwen features.

Does a nonlinear router let random features close the gap (i.e., is the Qwen
advantage merely 'linear accessibility')? MLP: hidden 64/128, ReLU, strong-ish
wd=1e-2, Adam lr=1e-3, full-batch 600 steps (same clean270 train, no OOD use).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
SIZES={"qwen3_0.6b":1024,"qwen3_1.7b":2048,"qwen3_8b_base":4096}
SEEDS=[7,17,27]
def balacc(pred,oracle):
    cm=np.zeros((3,3),int)
    for t,p in zip(oracle,pred):
        if t<3 and p<3: cm[t,p]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    return float(rec.mean())
def mlp_probe(Xtr,ytr,Xte,h=64,steps=600,wd=1e-2,seed=0,lr=1e-3):
    torch.manual_seed(seed)
    d=Xtr.shape[1]; net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,3))
    Xt=torch.from_numpy(Xtr).double(); yt=torch.from_numpy(ytr).long()
    # standardize by train stats
    mu=Xt.mean(0); sd=Xt.std(0)+1e-9
    Xn=(Xt-mu)/sd; Xe=(torch.from_numpy(Xte).double()-mu)/sd
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    net=net.double()
    for _ in range(steps):
        opt.zero_grad(); l=nn.functional.cross_entropy(net(Xn),yt); l.backward(); opt.step()
    with torch.no_grad(): return net(Xe).argmax(1).numpy()
def main():
    print("model   head   P_bal3 R_bal3 dP_bal3")
    for tag,d in SIZES.items():
        accL={"pretrained":[],"random":[]}; accM={"pretrained":[],"random":[]}
        for s in SEEDS:
            for ini in ["pretrained","random"]:
                z=np.load(f"qwen_feats/{tag}_{ini}_s{s}.npz")
                Hc,Hb3=z["h_clean"],z["h_bal3"]; fam=z["clean_family"]
                # linear
                from analysis.linear_probe import softmax_predict,softmax_regression
                _,p=softmax_predict(Hb3,*softmax_regression(Hc,fam,n_iter=2000))
                accL[ini].append(balacc(p,z["bal3_oracle"]))
                # mlp
                pm=mlp_probe(Hc,fam,Hb3,h=64,seed=s)
                accM[ini].append(balacc(pm,z["bal3_oracle"]))
        for name,acc in [("linear",accL),("mlp64",accM)]:
            P=np.mean(acc["pretrained"]);R=np.mean(acc["random"])
            print(f"{tag:<14} {name:<7} {P:.3f} {R:.3f} {P-R:+.3f}")
if __name__=="__main__": main()
