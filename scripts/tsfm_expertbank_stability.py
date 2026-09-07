"""Expert-bank robustness + oracle margin/stability (CPU-only, uses stored
per-window router predictions + window ctx/fut; no model re-run).

A. Expert-bank robustness (scheme-1): expand bank -> 5 experts with family tags
   {trend: [Trend, TrendSeasonal], periodic: [Periodic], local: [AR5, AR10]};
   each predicted family serves its validation-best member (chosen on the 270
   clean windows). Recompute routed MSE per model/init/seed and the
   pretrained-random delta on bal3/bal3n, compare to the 3-expert bank.
B. Margin/stability proxy: for each OOD window, margin = second_best/best of the
   ORIGINAL 3 experts; bin windows and report Qwen pretrained/random routing
   accuracy and their difference; also family-vs-oracle agreement on clean270.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert

KINDS=["trend","periodic","local","mixture","regime"]
SEEDS=[7,17,27]
MODELS=["qwen3_8b_base","chronos_t5-small","chronos_t5-base","chronos_bolt-small",
        "timesfm_2.5-200m","moment-1-large","moirai-1.1-R-small"]
OUT=Path("results/iclr/tsfm_deliver_v2")

def period_of(x):
    x=x-x.mean(); best=(12,0.0)
    for p in range(6,40):
        t=np.arange(len(x)); c=np.abs(np.sum(x*np.exp(-2j*np.pi*t/p)))
        if c>best[1]: best=(p,float(c))
    return best[0]

class TrendSeasonalExpert:
    family="trend"
    def predict(self, ctx, H):
        p=period_of(ctx); t=np.arange(len(ctx))
        A=np.stack([np.ones_like(t),t,np.sin(2*np.pi*t/p),np.cos(2*np.pi*t/p)],1)
        beta=np.linalg.lstsq(A,ctx,rcond=None)[0]
        tf=np.arange(len(ctx),len(ctx)+H)
        Af=np.stack([np.ones_like(tf),tf,np.sin(2*np.pi*tf/p),np.cos(2*np.pi*tf/p)],1)
        return Af@beta

class ARExpert:
    def __init__(self,k,ridge=1.0): self.k=k; self.ridge=ridge; self.family="local"
    def predict(self, ctx, H):
        k=self.k; n=len(ctx)
        X=[];y=[]
        for t in range(k,n):
            X.append(ctx[t-k:t]); y.append(ctx[t])
        X=np.array(X); y=np.array(y)
        Xt=np.concatenate([X,np.ones((len(X),1))],1)
        A=Xt.T@Xt; b=Xt.T@y
        beta=np.linalg.solve(A+self.ridge*np.eye(A.shape[0]),b)
        state=list(ctx[-k:]); out=[]
        for _ in range(H):
            v=beta[:-1]@np.array(state[::-1])+beta[-1]
            out.append(v); state.append(v); state.pop(0)
        return np.array(out)

TREND_E=[TrendExpert(),TrendSeasonalExpert()]
PER_E=[PeriodicExpert()]
LOC_E=[LocalExpert(),ARExpert(10,ridge=1.0)]
BANK={"trend":TREND_E,"periodic":PER_E,"local":LOC_E}
ORIG=[TrendExpert(),PeriodicExpert(),LocalExpert()]

def err3(exp,ctx,fut):
    return np.array([float(np.mean((e.predict(ctx,16)-fut)**2)) for e in exp])

def load_pred(model,init,seed,tag):
    z=np.load(OUT/"perwindow"/f"pw_{model}_{init}_s{seed}.npz")
    return z[f"c_{tag}_pred"], z[f"c_{tag}_oracle"], z[f"c_{tag}_err"]

# validation-best member per family on the 270 clean windows
def val_best_member():
    best={}
    for s in SEEDS:
        ctx,fut,kk=build_labeled_windows(KINDS,64,16,150,s)
        clean=np.concatenate([np.arange(k*150,k*150+90) for k in range(3)])
        for f in ["trend","periodic","local"]:
            fammask=kk[clean]==["trend","periodic","local"].index(f)
            idx=clean[fammask]
            if len(idx)==0: continue
            scores={type(e).__name__:0.0 for e in BANK[f]}
            cnt=0
            for i in idx:
                for e in BANK[f]:
                    scores[type(e).__name__]+=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
                cnt+=1
            pick=min(scores,key=lambda k:scores[k]/max(cnt,1))
            best.setdefault(f,[]).append(pick)
    # majority pick per family
    out={}
    for f in best:
        names=best[f]; out[f]=max(set(names),key=names.count)
    return out

def main():
    vbest=val_best_member()
    print("validation-best member per family (clean windows):",vbest)
    # precompute clean270 margin/agreement by family vs oracle (original 3)
    clean_aggr=[]
    for s in SEEDS:
        ctx,fut,kk=build_labeled_windows(KINDS,64,16,150,s)
        clean=np.concatenate([np.arange(k*150,k*150+90) for k in range(3)])
        fam=kk[clean]; agree=[]; margin=[]
        for i in clean:
            e=err3(ORIG,ctx[i],fut[i]); o=int(np.argmin(e))
            order=e[np.argsort(e)]; margin.append(order[1]/(order[0]+1e-9))
            agree.append(int(fam[i]==o) if False else None)
        # recompute per window
        cl=list(clean)
        for j,i in enumerate(cl):
            e=err3(ORIG,ctx[i],fut[i]); o=int(np.argmin(e))
            agree[j]=int(fam[j]==o)
        clean_aggr.append((agree,margin))
    # report agreement by margin bin (clean270)
    print("\n=== clean270 family-vs-oracle agreement vs margin (original 3 experts) ===")
    bins=[(1.0,1.05),(1.05,1.2),(1.2,1.5),(1.5,1e9)]
    for lo,hi in bins:
        tot=0; ok=0
        for (agree,margin) in clean_aggr:
            for a,m in zip(agree,margin):
                if lo<=m<hi: tot+=1; ok+=a
        if tot: print(f"  margin[{lo:.2f},{hi:.2f}): n={tot:4d} agreement={ok/tot:.3f}")
    # OOD: qwen pretrained/random routing vs margin bin (bal3)
    print("\n=== bal3: Qwen routing acc by oracle margin bin (family-label router) ===")
    for lo,hi in bins:
        rows=[]
        for s in SEEDS:
            z=np.load(f"results/iclr/e4_balanced3/windows/bal3_s{s}.npz"); ctx,fut,orac=z["ctx"],z["fut"],z["oracle"]
            eo=np.zeros((len(ctx),3))
            for j,e in enumerate(ORIG):
                for i in range(len(ctx)): eo[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
            marg=eo[np.arange(len(ctx)),np.argsort(eo,1)[:,1]]/(eo.min(1)+1e-9)
            sel=np.where((marg>=lo)&(marg<hi))[0]
            if len(sel)==0: continue
            for ini in ["pretrained","random"]:
                pred,_o,_e=load_pred("qwen3_8b_base",ini,s,"bal3")
                rows.append((ini,len(sel),float(np.mean(pred[sel]==orac[sel]))))
        if rows:
            for ini in ["pretrained","random"]:
                v=[r[2] for r in rows if r[0]==ini]
                if v: print(f"  margin[{lo:.2f},{hi:.2f}) {ini:<11} acc={np.mean(v):.3f}")
    # expert-bank robustness: routed MSE under expanded bank (serving val-best member of predicted family)
    print("\n=== routed MSE: original 3-expert vs expanded 5-expert bank (bal3+bal3n) ===")
    hdr=f"{'model':<20}{'init':<10}{'orig3 bal3':>12}{'exp5 bal3':>12}{'orig3 bal3n':>12}{'exp5 bal3n':>12}"
    print(hdr)
    res={}
    for m in MODELS:
        for ini in ["pretrained","random"]:
            o3=[];e5=[];o3n=[];e5n=[]
            for s in SEEDS:
                for tag,oacc,oaccn in [("bal3",o3,o3n),("bal3n",o3n,None)]:
                    pass
            # bal3
            for s in SEEDS:
                pred,orac,err=load_pred(m,ini,s,"bal3")
                z=np.load(f"results/iclr/e4_balanced3/windows/bal3_s{s}.npz"); ctx,fut=z["ctx"],z["fut"]
                o3.append(float(np.mean(err[np.arange(len(pred)),pred])))
                # expanded: serve val-best member of predicted family
                mse5=[]
                fam_names=["trend","periodic","local"]
                for i in range(len(pred)):
                    fam=fam_names[pred[i]]
                    serv=next(e for e in BANK[fam] if type(e).__name__==vbest[fam])
                    mse5.append(float(np.mean((serv.predict(ctx[i],16)-fut[i])**2)))
                e5.append(float(np.mean(mse5)))
                # bal3n
                predn,oracn,errn=load_pred(m,ini,s,"bal3n")
                zn=np.load(f"results/iclr/e4_balanced3_natural/windows/bal3n_s{s}.npz"); ctxn,futn=zn["ctx"],zn["fut"]
                o3n.append(float(np.mean(errn[np.arange(len(predn)),predn])))
                mse5n=[]
                for i in range(len(predn)):
                    fam=fam_names[predn[i]]
                    serv=next(e for e in BANK[fam] if type(e).__name__==vbest[fam])
                    mse5n.append(float(np.mean((serv.predict(ctxn[i],16)-futn[i])**2)))
                e5n.append(float(np.mean(mse5n)))
            res[(m,ini)]={"o3":float(np.mean(o3)),"e5":float(np.mean(e5)),
                          "o3n":float(np.mean(o3n)),"e5n":float(np.mean(e5n))}
            print(f"{m:<20}{ini:<10}{res[(m,ini)]['o3']:>12.4f}{res[(m,ini)]['e5']:>12.4f}"
                  f"{res[(m,ini)]['o3n']:>12.4f}{res[(m,ini)]['e5n']:>12.4f}")
    print("\n=== pretrained - random routed-MSE delta: orig3 vs exp5 (bal3) ===")
    for m in MODELS:
        d3=res[(m,"pretrained")]["o3"]-res[(m,"random")]["o3"]
        d5=res[(m,"pretrained")]["e5"]-res[(m,"random")]["e5"]
        print(f"{m:<20} orig3 {d3:+.4f}  exp5 {d5:+.4f}")

if __name__=="__main__":
    main()
