"""Unified cross-model suite (task A recognition / B readout / C routing).

Runs for any model with a working frozen representation extractor:
  - task A: 5-class recognition, original + shuffled (pretrained + random)
  - task B: frozen-rep linear readout -> H=16 MSE
  - task C: 3-class balanced OOD routing (both AR_weak_sine and periodic+AR sets)
Outputs metrics.csv rows + per-window npz per (model,init,seed).
"""
from __future__ import annotations
import argparse, json, sys, csv, re
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from analysis.features import temporal_features
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]

def get_rep_extractor(name, init, device, seed=7):
    """Return embed(ctx_np)->(n,d). init in {pretrained,random}."""
    if name=="qwen3_8b_base":
        # use cached official hidden (fast)
        cache=REPO_ROOT/f"results/iclr/probe_official/qwen3_8b/seed{seed}/cache"
        p=init if init=="pretrained" else "random"
        def embed(ctx, seed=seed):
            # ctx is in ORIGINAL order (750). caches are train/test split; need all 750
            htr=np.load(cache/f"train_h_{p}_seed{seed}.npy"); hte=np.load(cache/f"test_h_{p}_seed{seed}.npy")
            return np.concatenate([np.concatenate([htr[k*90:(k+1)*90],hte[k*60:(k+1)*60]]) for k in range(5)])
        return embed, 4096
    if name.startswith("chronos"):
        sys.path.insert(0,str(REPO_ROOT/"third_party/chronos-forecasting/src"))
        tag=name.replace("chronos_","")
        mpath=f"/public/chenjiahui/SAFER-TS/code/model_cache/amazon__chronos-{tag}"
        from transformers import AutoConfig, AutoModelForSeq2SeqLM
        from chronos import ChronosConfig
        cfg=ChronosConfig(**{k:v for k,v in AutoConfig.from_pretrained(mpath).chronos_config.items() if k in ChronosConfig.__dataclass_fields__})
        tok=cfg.create_tokenizer()
        m=AutoModelForSeq2SeqLM.from_pretrained(mpath,torch_dtype=torch.bfloat16)
        if init=="random":
            torch.manual_seed(seed)
            for mod in m.modules():
                if isinstance(mod,(nn.Linear,nn.LayerNorm,nn.Embedding)): mod.reset_parameters()
        m=m.to(device).eval()
        def embed(ctx):
            out=[]
            with torch.no_grad():
                for st in range(0,len(ctx),64):
                    b=torch.from_numpy(ctx[st:st+64]).float()
                    tids,attn,_=tok.context_input_transform(b.cpu()); tids=tids.to(device); attn=attn.to(device)
                    e=m.encoder(input_ids=tids,attention_mask=attn); last=attn.sum(1)-1; idx=torch.arange(len(b),device=device)
                    out.append(e.last_hidden_state[idx,last].float().cpu().numpy())
            return np.concatenate(out)
        return embed, m.config.d_model
    if name.startswith("timesfm"):
        sys.path.insert(0,'/public/chenjiahui/SAFER-TS/code/baselines_external/timesfm/src')
        from scripts.tsfm_models import TimesFMRep
        r=TimesFMRep("/public/chenjiahui/SAFER-TS/code/model_cache/google__timesfm-2.5-200m-pytorch/model.safetensors", device, init=="random", seed)
        return r.embed, 1280
    raise ValueError(name)

def linear_readout(Xtr,Ytr,Xte,seed):
    torch.manual_seed(seed); net=nn.Linear(Xtr.shape[1],16)
    Xt=torch.from_numpy(Xtr).float(); Yt=torch.from_numpy(Ytr).float()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    for _ in range(300):
        opt.zero_grad(); l=nn.functional.mse_loss(net(Xt),Yt); l.backward(); opt.step()
    with torch.no_grad(): return net(torch.from_numpy(Xte).float()).numpy(), (Xtr.shape[1]*16+16)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models", default="timesfm_2.5-200m")
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("results/iclr/tsfm_suite"))
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in a.models.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            for init in ["pretrained","random"]:
                try:
                    embed,dim=get_rep_extractor(name,init,a.device,seed)
                except Exception as e:
                    print(f"[{name}/{init}/s{seed}] extractor FAIL {e}"); continue
                n=150; ntr=90
                ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,seed)
                train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
                test_idx=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
                tr_label,te_label=kk[train_idx],kk[test_idx]
                H=embed(ctx)
                # task A recognition (original + shuffled)
                rng=np.random.default_rng(0); ctx_sh=np.stack([rng.permutation(w) for w in ctx])
                H_sh=embed(ctx_sh)
                for cond,Hc in [("orig",H),("shuf",H_sh)]:
                    _,p=softmax_predict(Hc[test_idx],*softmax_regression(Hc[train_idx],tr_label,n_iter=2000))
                    acc=accuracy(p,te_label)
                    rows.append({"model":name,"init":init,"seed":seed,"task":f"A_recog_{cond}","test_set":"synth5",
                                 "accuracy":round(acc,4),"head_params":""})
                    print(f"[{name}/{init}/s{seed}] A_{cond} acc={acc:.3f}")
                # task B readout (forecast MSE)
                clean=np.ones(len(train_idx),bool)  # use all 5 kinds for readout
                Ytr=fut[train_idx]; Yte=fut[test_idx]
                pred,hp=linear_readout(H[train_idx],Ytr,H[test_idx],seed)
                mse=float(np.mean((pred-Yte)**2))
                rows.append({"model":name,"init":init,"seed":seed,"task":"B_readout","test_set":"synth5",
                             "mse":round(mse,4),"head_params":hp})
                print(f"[{name}/{init}/s{seed}] B_readout mse={mse:.3f} head_params={hp}")
    # write metrics.csv
    fields=["model","init","seed","task","test_set","n_test","accuracy","balanced_accuracy","macro_f1",
            "recall_trend","recall_periodic","recall_local","mse","head_params"]
    with open(a.out/"metrics.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})
    print(f"saved {a.out/'metrics.csv'} with {len(rows)} rows")

if __name__=="__main__":
    main()
