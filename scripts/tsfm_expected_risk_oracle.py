"""Expected-risk / majority-future oracle on the 270 clean windows.

Replays data.dynamics.generate_window with the SAME rng draw order to recover
each clean window's latent params (contexts verified bit-identical), then
resamples K=50 futures per window from that latent and computes:
  expected-risk winner  = argmin_j mean_k MSE(f_j(ctx), fut_k)
  majority winner       = argmax_j P(j wins a single future)
  stability             = max_j P(j wins)
  entropy(3)
Saves clean270 labels to disk for later router eval; also prints agreement
of single-future/family vs expected-risk/majority and its dependence on
stability (the decisive "single-future oracle is noisy near ties" evidence).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from data.dynamics import _standardize
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]
KINDS=["trend","periodic","local","mixture","regime"]
SEEDS=[7,17,27]; K=50
OUT=Path("results/iclr/tsfm_deliver_v2/expected_risk"); OUT.mkdir(parents=True,exist_ok=True)

def replay_clean(seed):
    """Replay build_labeled_windows; capture params; verify ctx identical."""
    from data.dynamics import build_labeled_windows, generate_window as _gw
    rng=np.random.default_rng(seed)
    ctx0,_,kk=build_labeled_windows(KINDS,64,16,150,seed)
    ctxs=[]; params=[]; fams=[]; wids=[]
    n_per=150; w=0
    for kind_id,kind in enumerate(["trend","periodic","local","mixture","regime"]):
        for _ in range(n_per):
            p={}
            if kind=="trend":
                slope=rng.uniform(0.03,0.18); noise=0.05
                x=slope*np.arange(80.0)+rng.normal(0.0,noise,80); p={"kind":"trend","slope":slope,"noise":noise,"x_full":x}
            elif kind=="periodic":
                period=float(rng.choice([12,24])); amp=rng.uniform(0.5,1.5); phase=rng.uniform(0.0,2*np.pi); noise=0.05
                t=np.arange(80.0)
                x=amp*np.sin(2*np.pi*t/period+phase)+rng.normal(0.0,noise,80)
                p={"kind":"periodic","period":period,"amp":amp,"phase":phase,"noise":noise,"x_full":x}
            elif kind=="local":
                phi=rng.uniform(0.7,0.95); sigma=0.15
                x=np.empty(80); x[0]=rng.normal(0.0,sigma)
                for t in range(1,80): x[t]=phi*x[t-1]+rng.normal(0.0,sigma)
                p={"kind":"local","phi":phi,"sigma":sigma,"x_full":x}
            elif kind=="mixture":
                slope=rng.uniform(0.01,0.06); period=float(rng.choice([12,24])); amp=rng.uniform(0.4,1.2)
                t=np.arange(80.0)
                x=slope*t+amp*np.sin(2*np.pi*t/period)+rng.normal(0.0,0.05,80)
                p={"kind":"mixture","slope":slope,"period":period,"amp":amp,"noise":0.05,"x_full":x}
            elif kind=="regime":
                switch=int(64*0.6); x=np.empty(80)
                x[:switch]=0.08*np.arange(switch)+rng.normal(0.0,0.05,switch)
                x[switch:]=np.sin(2*np.pi*np.arange(switch,80)/12.0)+rng.normal(0.0,0.05,80-switch)
                p={"kind":"regime","x_full":x}
            x=_standardize(p["x_full"],64)
            ctxs.append(x[:64]); params.append(p); fams.append(kind_id); wids.append(w); w+=1
    ctxs=np.stack(ctxs); fams=np.array(fams)
    # verify bit-identical contexts
    md=np.abs(ctxs-ctx0).max()
    print(f"[seed {seed}] replay ctx max|diff| = {md:.2e}")
    assert md<1e-12, "replay mismatch!"
    return ctxs,params,fams

def resample_futures(p, K, seed, ctxmean, ctxstd):
    """K futures (K,16) in standardized space from latent p, given ctx stats."""
    kind=p["kind"]; t64=np.arange(64,80)
    rng=np.random.default_rng(seed)
    if kind in ("trend","periodic","mixture"):
        total=80
        if kind=="trend": sig=np.stack([p["slope"]*np.arange(80.0)]*K)
        elif kind=="periodic":
            base=p["amp"]*np.sin(2*np.pi*np.arange(80.0)/p["period"]+p["phase"])
            sig=np.stack([base]*K)
        else:
            base=p["slope"]*np.arange(80.0)+p["amp"]*np.sin(2*np.pi*np.arange(80.0)/p["period"])
            sig=np.stack([base]*K)
        noise=rng.normal(0.0,p.get("noise",0.05),(K,total))
        full=sig+noise
        return (full[:,64:]-ctxmean)/ctxstd
    if kind=="local":
        phi=p["phi"]; sigma=p["sigma"]; xlast=p["x_full"][63]
        fut=np.zeros((K,16))
        for k in range(K):
            x=xlast
            for h in range(16):
                x=phi*x+rng.normal(0.0,sigma)
                fut[k,h]=x
        return (fut-ctxmean)/ctxstd
    raise ValueError(kind)

def err_of(e, ctx, fut):
    return float(np.mean((e.predict(ctx,16)-fut)**2))

def main():
    report={}
    for seed in SEEDS:
        ctxs,params,fams=replay_clean(seed)
        n=150; ntr=90
        clean=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
        fam=fams[clean]; P=[params[i] for i in clean]
        res=[]  # per clean window dict
        for i in range(len(clean)):
            c=ctxs[clean[i]]; p=P[i]; cm=c.mean(); cs=c.std()+1e-6
            fs=resample_futures(p,K,1000*seed+i,cm,cs)
            errs=np.zeros((K,3))
            for j,e in enumerate(EXPERTS):
                for kk in range(K):
                    errs[kk,j]=err_of(e,c,fs[kk])
            wincount=errs.argmin(1)
            pj=np.array([np.mean(wincount==j) for j in range(3)])
            exp_winner=int(np.argmin(errs.mean(0)))
            maj_winner=int(np.argmax(pj))
            res.append({"family":int(fam[i]),"exp_winner":int(exp_winner),"maj_winner":int(maj_winner),
                        "p_win":[float(v) for v in pj],"stability":float(pj.max()),
                        "entropy":float(-np.sum(pj*np.log(pj+1e-12))) if pj.min()>0 else float(-np.sum(pj*np.log(pj+1e-12))),
                        "single_oracle":None})
        fam=np.array([r["family"] for r in res]); expw=np.array([r["exp_winner"] for r in res])
        majw=np.array([r["maj_winner"] for r in res]); stab=np.array([r["stability"] for r in res])
        # compare with the single-future oracle actually realized (compute on the real future)
        # we don't have stored clean futures here; recompute single-future winner from the true future:
        from data.dynamics import build_labeled_windows
        _,fut0,_=build_labeled_windows(KINDS,64,16,150,seed)
        sing=np.array([int(np.argmin([err_of(e,ctxs[clean[i]],fut0[clean[i]]) for e in EXPERTS])) for i in range(len(clean))])
        a_fam_sing=float(np.mean(fam==sing)); a_fam_exp=float(np.mean(fam==expw)); a_fam_maj=float(np.mean(fam==majw))
        a_exp_maj=float(np.mean(expw==majw)); a_sing_exp=float(np.mean(sing==expw))
        report[str(seed)]={"n_clean":len(clean),
            "agree_family_single":round(a_fam_sing,4),"agree_family_expected":round(a_fam_exp,4),
            "agree_family_majority":round(a_fam_maj,4),"agree_single_expected":round(a_sing_exp,4),
            "mean_stability":round(float(stab.mean()),4),
            "n_unstable_le0.6":int(np.sum(stab<=0.6)),"n_stable_ge0.8":int(np.sum(stab>=0.8)),
            "agreement_by_stability":[]}
        for lo,hi in [(0.0,0.6),(0.6,0.8),(0.8,1.0)]:
            m=(stab>=lo)&(stab<hi if hi<1.0 else stab<=hi)
            if m.sum():
                report[str(seed)]["agreement_by_stability"].append(
                    {"bin":[lo,hi],"n":int(m.sum()),
                     "family_vs_single":round(float(np.mean(fam[m]==sing[m])),3),
                     "family_vs_expected":round(float(np.mean(fam[m]==expw[m])),3)})
        print(f"[seed {seed}] fam-vs-single {a_fam_sing:.3f} | fam-vs-expected {a_fam_exp:.3f} | "
              f"single-vs-expected {a_sing_exp:.3f} | mean stability {stab.mean():.3f}")
        np.savez(OUT/f"clean_expected_risk_s{seed}.npz", family=fam, exp_winner=expw, maj_winner=majw,
                 single_winner=sing, stability=stab)
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("saved ->",OUT)
if __name__=="__main__":
    main()
