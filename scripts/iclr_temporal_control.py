"""ICLR P0-1: temporal-order control across seeds (official Qwen3-8B-Base).

Conditions (same windows/seeds, official 8B pre/random + feature baselines):
  - Original, Shuffled (per-window random permutation), Marginal-only (no order)
Report clean recognition and OOD routing per seed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
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
    ap.add_argument("--model-tag", default="qwen3_8b_official")
    ap.add_argument("--cache-root", type=Path, default=Path("results/iclr/probe_official/qwen3_8b"))
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--n-per-kind", type=int, default=150)
    ap.add_argument("--n-train-per-kind", type=int, default=90)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/temporal_control/qwen3_8b_official.json"))
    a=ap.parse_args()
    n=a.n_per_kind; ntr=a.n_train_per_kind
    rep={"model":a.model_tag,"seeds":[int(x) for x in a.seeds.split(",")]}
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]
    for sd in [int(x) for x in a.seeds.split(",")]:
        contexts,futures,kind_ids=build_labeled_windows(KINDS,64,16,n,sd)
        train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        test_idx=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
        tr_label,te_label=kind_ids[train_idx],kind_ids[test_idx]
        rng=np.random.default_rng(0)
        shuffled=np.stack([rng.permutation(w) for w in contexts])
        marginal=np.stack([marginal_features(w) for w in contexts])
        def probe(X):
            _,p=softmax_predict(X[test_idx],*softmax_regression(X[train_idx],tr_label,n_iter=2000))
            return accuracy(p,te_label)
        def route(X):
            hard=np.isin(te_label,[3,4]); clean=tr_label<3
            _,p=softmax_predict(X[test_idx][hard],*softmax_regression(X[train_idx][clean],tr_label[clean],n_iter=2000))
            err=np.zeros((int(hard.sum()),3)); h_idx=test_idx[hard]
            for j,e in enumerate(experts):
                for i,w in enumerate(h_idx): err[i,j]=float(np.mean((e.predict(contexts[w],16)-futures[w])**2))
            return accuracy(p,err.argmin(axis=1))
        srep={}
        fte=np.stack([temporal_features(x) for x in contexts]); fsh=np.stack([temporal_features(x) for x in shuffled])
        srep["feature_temporal"]={"recog":probe(fte),"route":route(fte)}
        srep["feature_shuffled"]={"recog":probe(fsh),"route":route(fsh)}
        srep["marginal_only"]={"recog":probe(marginal),"route":route(marginal)}
        for init in ["pretrained","random"]:
            cd=a.cache_root/f"seed{sd}/cache"
            htr=np.load(cd/f"train_h_{init}_seed{sd}.npy"); hte=np.load(cd/f"test_h_{init}_seed{sd}.npy")
            Xo=np.concatenate([np.concatenate([htr[k*90:(k+1)*90],hte[k*60:(k+1)*60]]) for k in range(5)])
            model,tok=_load_model(a.model_path,a.device,init=="random")
            Xsh=extract_last_hidden(model,tok,shuffled,a.device); del model; torch.cuda.empty_cache()
            srep[f"llm_{init}"]={"original_recog":probe(Xo),"original_route":route(Xo),
                                 "shuffled_recog":probe(Xsh),"shuffled_route":route(Xsh)}
        rep[f"seed{sd}"]=srep
        print(f"[seed {sd}] llm_pre orig_recog={srep['llm_pretrained']['original_recog']:.3f} shuffle={srep['llm_pretrained']['shuffled_recog']:.3f} | marginal_recog={srep['marginal_only']['recog']:.3f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(rep,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
