"""ICLR E4: balanced OOD test (trend vs periodic oracle classes).

The synthetic mixture/regime OOD windows are 87% trend-oracle; local-oracle
windows are structurally absent.  We rejection-sample mixture/regime windows so
the oracle is balanced between trend and periodic (e.g. 60 each), then compare
official-8B / random-8B / marginal-only / feature router / constant-trend /
best-fixed on balanced accuracy, macro-F1, per-class recall, and MSE.

Router is trained ONLY on the clean synthetic kinds (no OOD tuning).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows, generate_window
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _load_model, extract_last_hidden
KINDS=["trend","periodic","local","mixture","regime"]

def marginal_features(x):
    q=np.quantile(x,[0.1,0.25,0.5,0.75,0.9]); s=x.std()
    return np.concatenate([q,[x.mean(),s,(x.mean()/(s+1e-9)),
        np.mean((x-x.mean())**3)/(s**3+1e-9),np.mean((x-x.mean())**4)/(s**4+1e-9)-3,
        x.min(),x.max(),x.max()-x.min()]])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/9950backfile/chenjiahui/ARForecast-Diagnosis/models/qwen3-8b-base")
    ap.add_argument("--cache-root", type=Path, default=Path("results/iclr/probe_official/qwen3_8b"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/e4_balanced/qwen3_8b_official.json"))
    a=ap.parse_args()
    seeds=[int(s) for s in a.seeds.split(",")]
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]; names=["trend","periodic","local"]
    report={"model":"official Qwen3-8B-Base","per_class":a.per_class,"per_seed":{}}

    for s in seeds:
        # clean training (probe) material
        c,_,kk=build_labeled_windows(KINDS,64,16,150,s)
        n=150; ntr=90
        train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        tr_label=kk[train_idx]; clean=tr_label<3
        Htr=np.load(a.cache_root/f"seed{s}/cache/train_h_pretrained_seed{s}.npy")
        Hrtr=np.load(a.cache_root/f"seed{s}/cache/train_h_random_seed{s}.npy")
        feats_tr=np.stack([temporal_features(x) for x in c[train_idx][clean]])
        marg_tr=np.stack([marginal_features(x) for x in c[train_idx][clean]])

        # generate many OOD windows, pick balanced trend/periodic-oracle set
        rng=np.random.default_rng(s)
        ctx=[]; fut=[]; oracle=[]
        for kind in ["mixture","regime"]:
            for _ in range(3000):
                cw,fw=generate_window(kind,64,16,rng)
                errs=[]
                for e in experts:
                    p=e.predict(cw,16); errs.append(float(np.mean((p-fw)**2)))
                ctx.append(cw); fut.append(fw); oracle.append(int(np.argmin(errs)))
        ctx=np.stack(ctx); fut=np.stack(fut); oracle=np.array(oracle)
        # reject to balanced trend/periodic
        keep=np.concatenate([np.where(oracle==0)[0][:a.per_class], np.where(oracle==1)[0][:a.per_class]])
        ctx=ctx[keep]; fut=fut[keep]; oracle=oracle[keep]
        print(f"[seed {s}] balanced OOD set: {len(ctx)} windows, oracle counts T/P = {int((oracle==0).sum())}/{int((oracle==1).sum())}")

        # hidden extraction
        model,tok=_load_model(a.model_path,a.device,False)
        Hq=extract_last_hidden(model,tok,ctx,a.device); del model; torch.cuda.empty_cache()
        model,tok=_load_model(a.model_path,a.device,True)
        Hr=extract_last_hidden(model,tok,ctx,a.device); del model; torch.cuda.empty_cache()
        te_feats=np.stack([temporal_features(x) for x in ctx])
        te_marg=np.stack([marginal_features(x) for x in ctx])
        _,pq=softmax_predict(Hq,*softmax_regression(Htr[clean],tr_label[clean],n_iter=2000))
        _,pr=softmax_predict(Hr,*softmax_regression(Hrtr[clean],tr_label[clean],n_iter=2000))
        _,pm=softmax_predict(te_marg,*softmax_regression(marg_tr,tr_label[clean],n_iter=2000))
        _,pf=softmax_predict(te_feats,*softmax_regression(feats_tr,tr_label[clean],n_iter=2000))
        err=np.zeros((len(ctx),3))
        for j,e in enumerate(experts):
            for i in range(len(ctx)): err[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
        const=np.zeros(len(ctx),int)
        def metrics(p):
            cm=np.zeros((2,3),int)
            for t,pp in zip(oracle,p): cm[t,pp]+=1
            recall=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(2)])
            bal=float(recall.mean())
            prec=np.array([cm[i,i]/(cm[:,i].sum()+1e-12) if cm[:,i].sum()>0 else 0 for i in range(2)])
            f1=np.mean([2*p*r/(p+r+1e-12) for p,r in zip(prec,recall)])
            return {"mse":float(np.mean(err[np.arange(len(p)),p])),"balanced_acc":bal,"macro_f1":float(f1),
                    "recall":{names[k]:float(recall[k]) for k in range(2)}}
        row={"oracle_counts":{names[k]:int((oracle==k).sum()) for k in range(2)}}
        for tag,p in [("qwen",pq),("rnd",pr),("marg",pm),("feat",pf),("trend_const",const)]:
            row[tag]=metrics(p)
        row["oracle_mse"]=float(np.mean(err.min(axis=1)))
        report["per_seed"][str(s)]=row
        print(f"    qwen balacc={row['qwen']['balanced_acc']:.3f} mse={row['qwen']['mse']:.3f} | const mse={row['trend_const']['mse']:.3f} | oracle={row['oracle_mse']:.3f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
