"""ICLR E4 re-audit: full balanced-metric diagnostics on official Qwen3-8B-Base.

For each seed (7/17/27) on the existing mixture/regime OOD test windows:
  - oracle-label counts
  - confusion matrix of each router (official8B / random8B / marginal-only /
    feature router) vs oracle expert
  - per-class recall (per oracle class), balanced accuracy, macro-F1
  - per-oracle-class routing MSE
  - constant-trend and test-best-fixed baselines
  - Qwen routing MSE minus best-fixed MSE, paired CI
Per-seed values are reported (not just pooled).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS = ["trend", "periodic", "local", "mixture", "regime"]

def marginal_features(x):
    q=np.quantile(x,[0.1,0.25,0.5,0.75,0.9]); s=x.std()
    return np.concatenate([q,[x.mean(),s,(x.mean()/(s+1e-9)),
        np.mean((x-x.mean())**3)/(s**3+1e-9),np.mean((x-x.mean())**4)/(s**4+1e-9)-3,
        x.min(),x.max(),x.max()-x.min()]])

def balanced_metrics(pred, oracle, n_cls=3):
    cm = np.zeros((n_cls, n_cls), int)
    for t, p in zip(oracle, pred): cm[t, p] += 1
    recall = np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(n_cls)])
    prec = np.array([cm[:,i][i]/(cm[:,i].sum()+1e-12) for i in range(n_cls)])
    f1 = 2*prec*recall/(prec+recall+1e-12)
    balanced_acc = float(recall.mean())
    macro_f1 = float(f1.mean())
    return cm, recall, prec, f1, balanced_acc, macro_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("results/iclr/probe_official/qwen3_8b"))
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--out", type=Path, default=Path("results/iclr/e4_audit/qwen3_8b_official.json"))
    a = ap.parse_args()
    seeds=[int(s) for s in a.seeds.split(",")]
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]
    names=["trend","periodic","local"]
    report={"model":"official Qwen3-8B-Base","seeds":seeds,"per_seed":{},"pooled":{}}

    pool={k:[] for k in ["qwen","rnd","marg","feat","trend_const","best_fixed","oracle"]}
    for s in seeds:
        contexts,futures,kind_ids=build_labeled_windows(KINDS,64,16,150,s)
        n=150; ntr=90
        train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        test_idx=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
        tr_label,te_label=kind_ids[train_idx],kind_ids[test_idx]
        hard=np.isin(te_label,[3,4]); h_idx=test_idx[hard]
        # expert errors / oracle
        err=np.zeros((len(h_idx),3))
        for j,e in enumerate(experts):
            for i,w in enumerate(h_idx):
                err[i,j]=float(np.mean((e.predict(contexts[w],16)-futures[w])**2))
        oracle=err.argmin(axis=1)
        clean=tr_label<3
        # features
        feats_tr=np.stack([temporal_features(x) for x in contexts[train_idx][clean]])
        marg_tr=np.stack([marginal_features(x) for x in contexts[train_idx][clean]])
        te_feats=np.stack([temporal_features(x) for x in contexts[test_idx[hard]]])
        te_marg=np.stack([marginal_features(x) for x in contexts[test_idx[hard]]])
        # hidden
        h_tr=np.load(a.cache_root/f"seed{s}/cache/train_h_pretrained_seed{s}.npy")
        h_te=np.load(a.cache_root/f"seed{s}/cache/test_h_pretrained_seed{s}.npy")
        hr_tr=np.load(a.cache_root/f"seed{s}/cache/train_h_random_seed{s}.npy")
        hr_te=np.load(a.cache_root/f"seed{s}/cache/test_h_random_seed{s}.npy")
        # X in original window order
        def reorder(Htr,Hte):
            return np.concatenate([np.concatenate([Htr[k*90:(k+1)*90],Hte[k*60:(k+1)*60]]) for k in range(5)])
        Xq=reorder(h_tr,h_te); Xr=reorder(hr_tr,hr_te)
        _,pq=softmax_predict(Xq[test_idx][hard],*softmax_regression(Xq[train_idx][clean],tr_label[clean],n_iter=2000))
        _,pr=softmax_predict(Xr[test_idx][hard],*softmax_regression(Xr[train_idx][clean],tr_label[clean],n_iter=2000))
        _,pm=softmax_predict(te_marg,*softmax_regression(marg_tr,tr_label[clean],n_iter=2000))
        _,pf=softmax_predict(te_feats,*softmax_regression(feats_tr,tr_label[clean],n_iter=2000))
        const=0*np.ones(len(oracle),int)  # always trend
        bf=int(err.mean(axis=0).argmin()); bestfixed=bf*np.ones(len(oracle),int)
        mse=lambda p: float(np.mean(err[np.arange(len(p)),p]))
        row={"oracle_counts":{names[k]:int((oracle==k).sum()) for k in range(3)}}
        for tag,p in [("qwen",pq),("rnd",pr),("marg",pm),("feat",pf),("trend_const",const),("best_fixed",bestfixed)]:
            cm,rec,_,_,ba,mf=balanced_metrics(p,oracle)
            row[tag]={"mse":mse(p),"balanced_acc":ba,"macro_f1":mf,
                      "per_class_recall":{names[k]:float(rec[k]) for k in range(3)},
                      "cm":cm.tolist()}
        row["oracle_mse"]=mse(oracle)
        report["per_seed"][str(s)]=row
        # paired Qwen - best-fixed MSE
        q_err=err[np.arange(len(pq)),pq]; b_err=err[np.arange(len(pq)),bestfixed]
        d=q_err-b_err; rng=np.random.default_rng(0)
        ds=np.array([np.mean(rng.choice(d,len(d),replace=True)) for _ in range(2000)])
        row["qwen_minus_bestfixed"]={"mean":float(d.mean()),"ci":[(float(np.percentile(ds,2.5))),(float(np.percentile(ds,97.5)))]}
        for k in pool:
            if k=="oracle": pool[k].append(err.min(axis=1).tolist())
            elif k=="trend_const": pool[k].append(err[:,0].tolist())
            elif k=="best_fixed": pool[k].append(err[:,bf].tolist())
            else:
                p=locals()["p"+{"qwen":"q","rnd":"r","marg":"m","feat":"f"}[k]]
                pool[k].append(err[np.arange(len(p)),p].tolist())
    # pooled (window-level)
    for k in ["qwen","rnd","marg","feat","trend_const","best_fixed","oracle"]:
        vals=np.concatenate(pool[k]); rng=np.random.default_rng(0)
        ms=np.array([np.mean(rng.choice(vals,len(vals),replace=True)) for _ in range(2000)])
        report["pooled"][k+"_mse"]={"mean":float(vals.mean()),"ci":[float(np.percentile(ms,2.5)),float(np.percentile(ms,97.5))]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
