"""ICLR P1: 3-class balanced OOD with a NATURAL local composition + Chronos.

Local-oracle windows come from periodic+AR (periodic amp 0.4/24 + AR(1) 0.9,
sigma 0.28) -- a genuinely unseen composition where the local expert wins
naturally (~54%), instead of heavy rejection sampling.  trend/periodic oracle
windows come from mixture.  Adds Chronos-T5-base to the comparison.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
sys.path.insert(0,str(REPO_ROOT/"third_party/chronos-forecasting/src"))
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

def gen_per_ar(rng):
    total=80; t=np.arange(total,dtype=float); x=np.empty(total); x[0]=rng.normal(0,0.2)
    for k in range(1,total): x[k]=0.9*x[k-1]+rng.normal(0,0.28)
    x=x+0.4*np.sin(2*np.pi*t/24)
    m=x[:64].mean(); s=x[:64].std()+1e-6
    return (x[:64]-m)/s,(x[64:]-m)/s

def metrics(pred,oracle,err):
    n_cls=3; cm=np.zeros((n_cls,n_cls),int)
    for t,p in zip(oracle,pred): cm[t,p]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(n_cls)])
    prec=np.array([cm[:,i][i]/(cm[:,i].sum()+1e-12) if cm[:,i].sum()>0 else 0 for i in range(n_cls)])
    f1=float(np.mean([2*p*r/(p+r+1e-12) for p,r in zip(prec,rec)]))
    return {"mse":float(np.mean(err[np.arange(len(pred)),pred])),"balanced_acc":float(rec.mean()),
            "macro_f1":f1,"recall":[float(r) for r in rec]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/9950backfile/chenjiahui/ARForecast-Diagnosis/models/qwen3-8b-base")
    ap.add_argument("--cache-root", type=Path, default=Path("results/iclr/probe_official/qwen3_8b"))
    ap.add_argument("--chronos-root", type=Path, default=Path("/public/chenjiahui/SAFER-TS/code/model_cache"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/e4_balanced3_natural/qwen3_8b_official.json"))
    a=ap.parse_args()
    seeds=[int(s) for s in a.seeds.split(",")]
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]
    wdir=Path("results/iclr/e4_balanced3_natural/windows"); wdir.mkdir(parents=True,exist_ok=True)
    report={"model":"official Qwen3-8B-Base","per_class":a.per_class,"local_source":"periodic+AR","per_seed":{}}
    for s in seeds:
        c,_,kk=build_labeled_windows(KINDS,64,16,150,s)
        n=150; ntr=90
        train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        tr_label=kk[train_idx]; clean=tr_label<3
        Htr=np.load(a.cache_root/f"seed{s}/cache/train_h_pretrained_seed{s}.npy")
        Hrtr=np.load(a.cache_root/f"seed{s}/cache/train_h_random_seed{s}.npy")
        feats_tr=np.stack([temporal_features(x) for x in c[train_idx][clean]])
        marg_tr=np.stack([marginal_features(x) for x in c[train_idx][clean]])
        tr_x=torch.from_numpy(c[train_idx][clean]).float(); tr_y=torch.from_numpy(tr_label[clean]).long()
        wpath=wdir/f"bal3n_s{s}.npz"
        if wpath.is_file():
            z=np.load(wpath); ctx,fut,oracle=z["ctx"],z["fut"],z["oracle"]
        else:
            rng=np.random.default_rng(s); ctx,fut,oracle=[],[],[]
            for _ in range(4000):  # mixture -> T/P oracle
                cw,fw=generate_window("mixture",64,16,rng)
                errs=[float(np.mean((e.predict(cw,16)-fw)**2)) for e in experts]
                j=int(np.argmin(errs))
                if j<2: ctx.append(cw); fut.append(fw); oracle.append(j)
            for _ in range(4000):  # periodic+AR -> local-oracle (natural)
                cw,fw=gen_per_ar(rng)
                errs=[float(np.mean((e.predict(cw,16)-fw)**2)) for e in experts]
                if int(np.argmin(errs))==2: ctx.append(cw); fut.append(fw); oracle.append(2)
            ctx=np.stack(ctx); fut=np.stack(fut); oracle=np.array(oracle)
            keep=np.concatenate([np.where(oracle==k)[0][:a.per_class] for k in range(3)])
            ctx,fut,oracle=ctx[keep],fut[keep],oracle[keep]
            np.savez(wpath,ctx=ctx,fut=fut,oracle=oracle)
        print(f"[seed {s}] balanced3-natural {len(ctx)} (T/P/L={int((oracle==0).sum())}/{int((oracle==1).sum())}/{int((oracle==2).sum())})")
        model,tok=_load_model(a.model_path,a.device,False)
        Hq=extract_last_hidden(model,tok,ctx,a.device); del model; torch.cuda.empty_cache()
        model,tok=_load_model(a.model_path,a.device,True)
        Hr=extract_last_hidden(model,tok,ctx,a.device); del model; torch.cuda.empty_cache()
        te_feats=np.stack([temporal_features(x) for x in ctx]); te_marg=np.stack([marginal_features(x) for x in ctx])
        err=np.zeros((len(ctx),3))
        for j,e in enumerate(experts):
            for i in range(len(ctx)): err[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
        preds={}
        _,preds["qwen"]=softmax_predict(Hq,*softmax_regression(Htr[clean],tr_label[clean],n_iter=2000))
        _,preds["random"]=softmax_predict(Hr,*softmax_regression(Hrtr[clean],tr_label[clean],n_iter=2000))
        _,preds["marginal"]=softmax_predict(te_marg,*softmax_regression(marg_tr,tr_label[clean],n_iter=2000))
        _,preds["features"]=softmax_predict(te_feats,*softmax_regression(feats_tr,tr_label[clean],n_iter=2000))
        mlp=MLPRouter(64,seed=s); preds["mlp"],_,_=train_eval(mlp,tr_x,tr_y,torch.from_numpy(ctx).float(),100,1e-3,64,a.device,s)
        pat=PatchTSTRouter(64,16,8,seed=s); preds["patchtst"],_,_=train_eval(pat,tr_x,tr_y,torch.from_numpy(ctx).float(),100,1e-3,64,a.device,s)
        preds["constant_trend"]=np.zeros(len(ctx),int)
        # Chronos-T5-base
        from transformers import AutoConfig, AutoModelForSeq2SeqLM
        from chronos import ChronosConfig
        mpath=a.chronos_root/"amazon__chronos-t5-base"
        cfg=ChronosConfig(**{k:v for k,v in AutoConfig.from_pretrained(str(mpath)).chronos_config.items() if k in ChronosConfig.__dataclass_fields__})
        tokz=cfg.create_tokenizer(); m=AutoModelForSeq2SeqLM.from_pretrained(str(mpath),torch_dtype=torch.bfloat16).to(a.device).eval()
        def emb(X):
            out=[]
            with torch.no_grad():
                for st in range(0,len(X),64):
                    b=torch.from_numpy(X[st:st+64]).float()
                    tids,attn,_=tokz.context_input_transform(b.cpu()); tids=tids.to(a.device); attn=attn.to(a.device)
                    enc=m.encoder(input_ids=tids,attention_mask=attn); last=attn.sum(1)-1; idx=torch.arange(len(b),device=a.device)
                    out.append(enc.last_hidden_state[idx,last].float().cpu().numpy())
            return np.concatenate(out,axis=0)
        Hb=emb(ctx); Hc=emb(c); del m; torch.cuda.empty_cache()
        _,preds["chronos_base"]=softmax_predict(Hb,*softmax_regression(Hc[train_idx][clean],tr_label[clean],n_iter=2000))
        row={"oracle_counts":[int((oracle==k).sum()) for k in range(3)]}
        for tag,p in preds.items(): row[tag]=metrics(p,oracle,err)
        row["oracle_mse"]=float(np.mean(err.min(axis=1)))
        report["per_seed"][str(s)]=row
        print(f"    qwen balacc={row['qwen']['balanced_acc']:.3f} mse={row['qwen']['mse']:.3f} | chronos_base={row['chronos_base']['balanced_acc']:.3f} mlp={row['mlp']['balanced_acc']:.3f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
