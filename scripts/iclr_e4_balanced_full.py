"""ICLR P0-1/P0-5: balanced OOD test with ALL baselines + paired MSE CIs.

Same balanced trend/periodic-oracle OOD windows (rejection-sampled, saved to
npz) evaluated by: official Qwen3-8B-Base (pre/rnd), MLP, PatchTST, Chronos-T5
(small, base), marginal-only, temporal features, constant-trend, best-fixed.
Reports per seed + pooled balanced acc, macro-F1, recall T/P, routed MSE, and
window-level paired CI of Qwen MSE minus each baseline.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT/"third_party/chronos-forecasting/src"))
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows, generate_window
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _load_model, extract_last_hidden
from scripts.iclr_learned_router import PatchTSTRouter, MLPRouter, train_eval
KINDS=["trend","periodic","local","mixture","regime"]

def marginal_features(x):
    q=np.quantile(x,[0.1,0.25,0.5,0.75,0.9]); s=x.std()
    return np.concatenate([q,[x.mean(),s,(x.mean()/(s+1e-9)),
        np.mean((x-x.mean())**3)/(s**3+1e-9),np.mean((x-x.mean())**4)/(s**4+1e-9)-3,
        x.min(),x.max(),x.max()-x.min()]])

def metrics(pred, oracle, err):
    cm=np.zeros((2,3),int)
    for t,pp in zip(oracle,pred): cm[t,pp]+=1
    recall=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(2)])
    prec=np.array([cm[i,i]/(cm[:,i].sum()+1e-12) if cm[:,i].sum()>0 else 0 for i in range(2)])
    f1=float(np.mean([2*p*r/(p+r+1e-12) for p,r in zip(prec,recall)]))
    return {"mse":float(np.mean(err[np.arange(len(pred)),pred])),
            "balanced_acc":float(recall.mean()),"macro_f1":f1,
            "recall_trend":float(recall[0]),"recall_periodic":float(recall[1])}

def paired_ci(a,b,seed=0,n_boot=2000):
    d=np.asarray(a,float)-np.asarray(b,float); rng=np.random.default_rng(seed)
    ms=np.array([np.mean(rng.choice(d,len(d),replace=True)) for _ in range(n_boot)])
    return {"mean":float(d.mean()),"ci":[float(np.percentile(ms,2.5)),float(np.percentile(ms,97.5))],"n":int(len(d))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/9950backfile/chenjiahui/ARForecast-Diagnosis/models/qwen3-8b-base")
    ap.add_argument("--cache-root", type=Path, default=Path("results/iclr/probe_official/qwen3_8b"))
    ap.add_argument("--chronos-root", type=Path, default=Path("/public/chenjiahui/SAFER-TS/code/model_cache"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/e4_balanced_full/qwen3_8b_official.json"))
    a=ap.parse_args()
    seeds=[int(s) for s in a.seeds.split(",")]
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]; names=["trend","periodic","local"]
    wdir=Path("results/iclr/e4_balanced_full/windows"); wdir.mkdir(parents=True,exist_ok=True)
    report={"model":"official Qwen3-8B-Base","per_class":a.per_class,"per_seed":{}}

    for s in seeds:
        # clean training material
        c,_,kk=build_labeled_windows(KINDS,64,16,150,s)
        n=150; ntr=90
        train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        tr_label=kk[train_idx]; clean=tr_label<3
        Htr=np.load(a.cache_root/f"seed{s}/cache/train_h_pretrained_seed{s}.npy")
        Hrtr=np.load(a.cache_root/f"seed{s}/cache/train_h_random_seed{s}.npy")
        feats_tr=np.stack([temporal_features(x) for x in c[train_idx][clean]])
        marg_tr=np.stack([marginal_features(x) for x in c[train_idx][clean]])
        tr_x=torch.from_numpy(c[train_idx][clean]).float()
        tr_y=torch.from_numpy(tr_label[clean]).long()

        # balanced OOD windows (save/load)
        wpath=wdir/f"bal_s{s}.npz"
        if wpath.is_file():
            z=np.load(wpath); ctx,fut,oracle=z["ctx"],z["fut"],z["oracle"]
        else:
            rng=np.random.default_rng(s); ctx,fut,oracle=[],[],[]
            for kind in ["mixture","regime"]:
                for _ in range(3000):
                    cw,fw=generate_window(kind,64,16,rng)
                    errs=[float(np.mean((e.predict(cw,16)-fw)**2)) for e in experts]
                    ctx.append(cw); fut.append(fw); oracle.append(int(np.argmin(errs)))
            ctx=np.stack(ctx); fut=np.stack(fut); oracle=np.array(oracle)
            keep=np.concatenate([np.where(oracle==0)[0][:a.per_class],np.where(oracle==1)[0][:a.per_class]])
            ctx,fut,oracle=ctx[keep],fut[keep],oracle[keep]
            np.savez(wpath,ctx=ctx,fut=fut,oracle=oracle)
        print(f"[seed {s}] balanced {len(ctx)} windows (T/P={int((oracle==0).sum())}/{int((oracle==1).sum())})")

        # representations
        model,tok=_load_model(a.model_path,a.device,False)
        Hq=extract_last_hidden(model,tok,ctx,a.device); del model; torch.cuda.empty_cache()
        model,tok=_load_model(a.model_path,a.device,True)
        Hr=extract_last_hidden(model,tok,ctx,a.device); del model; torch.cuda.empty_cache()
        te_feats=np.stack([temporal_features(x) for x in ctx])
        te_marg=np.stack([marginal_features(x) for x in ctx])
        err=np.zeros((len(ctx),3))
        for j,e in enumerate(experts):
            for i in range(len(ctx)): err[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))

        # routers
        _,pq=softmax_predict(Hq,*softmax_regression(Htr[clean],tr_label[clean],n_iter=2000))
        _,pr=softmax_predict(Hr,*softmax_regression(Hrtr[clean],tr_label[clean],n_iter=2000))
        _,pm=softmax_predict(te_marg,*softmax_regression(marg_tr,tr_label[clean],n_iter=2000))
        _,pf=softmax_predict(te_feats,*softmax_regression(feats_tr,tr_label[clean],n_iter=2000))
        # MLP / PatchTST
        mlp=MLPRouter(64,seed=s); pmlp,_,_=train_eval(mlp,tr_x,tr_y,torch.from_numpy(ctx).float(),100,1e-3,64,a.device,s)
        pat=PatchTSTRouter(64,16,8,seed=s); ppat,_,_=train_eval(pat,tr_x,tr_y,torch.from_numpy(ctx).float(),100,1e-3,64,a.device,s)
        # Chronos T5 small + base
        from transformers import AutoConfig, AutoModelForSeq2SeqLM
        from chronos import ChronosConfig
        Hch={}; Hch_tr={}; pch={}
        for tag,mpath in [("chronos_t5_small",a.chronos_root/"amazon__chronos-t5-small"),
                          ("chronos_t5_base",a.chronos_root/"amazon__chronos-t5-base")]:
            cfg=ChronosConfig(**{k:v for k,v in AutoConfig.from_pretrained(str(mpath)).chronos_config.items() if k in ChronosConfig.__dataclass_fields__})
            tokz=cfg.create_tokenizer(); m=AutoModelForSeq2SeqLM.from_pretrained(str(mpath),torch_dtype=torch.bfloat16).to(a.device).eval()
            def emb_chronos(X):
                out=[]
                with torch.no_grad():
                    for st in range(0,len(X),64):
                        b=torch.from_numpy(X[st:st+64]).float()
                        tids,attn,_=tokz.context_input_transform(b.cpu()); tids=tids.to(a.device); attn=attn.to(a.device)
                        enc=m.encoder(input_ids=tids,attention_mask=attn)
                        last=attn.sum(1)-1; idx=torch.arange(len(b),device=a.device)
                        out.append(enc.last_hidden_state[idx,last].float().cpu().numpy())
                return np.concatenate(out,axis=0)
            Hch[tag]=emb_chronos(ctx); Hch_tr[tag]=emb_chronos(c)
            del m; torch.cuda.empty_cache()
            _,pch[tag]=softmax_predict(Hch[tag],*softmax_regression(Hch_tr[tag][train_idx][clean],tr_label[clean],n_iter=2000))

        row={"oracle_counts":{"trend":int((oracle==0).sum()),"periodic":int((oracle==1).sum())},
             "oracle_mse":float(np.mean(err.min(axis=1)))}
        preds={}
        for tag,p in [("qwen",pq),("random",pr),("marginal",pm),("features",pf),
                      ("mlp",pmlp),("patchtst",ppat),
                      ("chronos_t5_small",pch["chronos_t5_small"]),("chronos_t5_base",pch["chronos_t5_base"]),
                      ("constant_trend",np.zeros(len(ctx),int))]:
            preds[tag]=p; row[tag]=metrics(p,oracle,err)
        # paired CI: qwen - each
        q_err=err[np.arange(len(ctx)),pq]
        row["qwen_minus"]={tag:paired_ci(q_err,err[np.arange(len(ctx)),p]) for tag,p in preds.items()}
        report["per_seed"][str(s)]=row
        print(f"    qwen balacc={row['qwen']['balanced_acc']:.3f} mse={row['qwen']['mse']:.3f} | "
              f"mlp={row['mlp']['balanced_acc']:.3f} patchtst={row['patchtst']['balanced_acc']:.3f} "
              f"chronos_s={row['chronos_t5_small']['balanced_acc']:.3f} const={row['constant_trend']['mse']:.3f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
