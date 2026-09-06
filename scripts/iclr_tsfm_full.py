"""Full cross-model suite: tasks A/B/C + native forecast -> metrics.csv, per-window npz, config.

Models: qwen3_8b_base, chronos_t5-{small,base}, chronos_bolt-{small,base}, timesfm_2.5-200m.
Each model x init{pretrained,random} x seed{7,17,27}:
  A: 5-class recognition (orig + shuffled)
  B: linear readout H=16 MSE (head params reported)
  C: 3-class balanced OOD routing on two test sets (e4_balanced3, e4_balanced3_natural)
Native forecast MSE where the model has a native interface (chronos, timesfm).
"""
from __future__ import annotations
import argparse, json, sys, csv, re
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]

def recog_metrics(pred, oracle):
    n=len(pred); cm=np.zeros((3,3),int)
    for t,pp in zip(oracle,pred): cm[t,pp]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    prec=np.array([cm[:,i][i]/(cm[:,i].sum()+1e-12) if cm[:,i].sum()>0 else 0 for i in range(3)])
    f1=float(np.mean([2*p*r/(p+r+1e-12) for p,r in zip(prec,rec)]))
    return float(rec.mean()), f1, rec

# ---------- model representation extractors ----------
def make_extractor(name, init, seed, device):
    if name=="qwen3_8b_base":
        cache=REPO_ROOT/f"results/iclr/probe_official/qwen3_8b/seed{seed}/cache"
        p=init
        def embed(ctx):
            # ctx in ORIGINAL 750 order -> rebuild from train/test caches
            htr=np.load(cache/f"train_h_{p}_seed{seed}.npy"); hte=np.load(cache/f"test_h_{p}_seed{seed}.npy")
            return np.concatenate([np.concatenate([htr[k*90:(k+1)*90],hte[k*60:(k+1)*60]]) for k in range(5)])
        return embed,4096,None
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
        def native(ctx):
            from chronos import ChronosPipeline
            pipe=ChronosPipeline.from_pretrained(mpath)
            pipe=pipe.to(device)
            outs=[]
            with torch.no_grad():
                for st in range(0,len(ctx),8):
                    b=torch.from_numpy(ctx[st:st+8]).float()
                    fc=pipe.predict(b, prediction_length=16, num_samples=1)  # (B, samples, H)
                    outs.append(fc[:,0,:].cpu().numpy())
            return np.concatenate(outs)
        return embed, m.config.d_model, native
    if name.startswith("timesfm"):
        sys.path.insert(0,'/public/chenjiahui/SAFER-TS/code/baselines_external/timesfm/src')
        from scripts.tsfm_models import TimesFMRep
        r=TimesFMRep("/public/chenjiahui/SAFER-TS/code/model_cache/google__timesfm-2.5-200m-pytorch/model.safetensors", device, init=="random", seed)
        return r.embed,1280,None
    if name.startswith("moment"):
        sys.path.insert(0,'/public/chenjiahui/波数据时序基座大模型.bak_public/moment-main/Inference')
        import json as _json
        from momentfm.models.moment import MOMENT
        wdir='/public/chenjiahui/波数据时序基座大模型.bak_public/WaveFormer/models/external/moment'
        cfg=_json.load(open(f"{wdir}/config.json")); cfg['d_model']=cfg['t5_config']['d_model']
        m=MOMENT(cfg, model_kwargs={})
        sd=torch.load(f"{wdir}/pytorch_model.bin", map_location='cpu')
        m.load_state_dict(sd, strict=False)
        if init=="random":
            torch.manual_seed(seed)
            for mod in m.modules():
                if isinstance(mod,(nn.Linear,nn.LayerNorm,nn.Embedding)): mod.reset_parameters()
        m=m.to(device).eval()
        def embed(ctx):
            x=torch.from_numpy(ctx).float().unsqueeze(1).to(device)
            out=[]
            with torch.no_grad():
                for st in range(0,len(ctx),32):
                    o=m.embed(x_enc=x[st:st+32]); out.append(o.embeddings.cpu().numpy())
            return np.concatenate(out)
        return embed, cfg['d_model'], None
    raise ValueError(name)
def lin_readout(Xtr,Ytr,Xte,seed):
    torch.manual_seed(seed); net=nn.Linear(Xtr.shape[1],16)
    Xt=torch.from_numpy(Xtr).float(); Yt=torch.from_numpy(Ytr).float()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    for _ in range(300):
        opt.zero_grad(); l=nn.functional.mse_loss(net(Xt),Yt); l.backward(); opt.step()
    with torch.no_grad(): return net(torch.from_numpy(Xte).float()).numpy(), (Xtr.shape[1]*16+16)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models", default="timesfm_2.5-200m,chronos_t5-base")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("results/iclr/tsfm_full"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=[]; windows={}
    # balanced OOD windows (shared, seed-specific)
    bal={}
    for s in [int(x) for x in a.seeds.split(",")]:
        for tag,root in [("bal3",f"results/iclr/e4_balanced3/windows/bal3_s{s}.npz"),
                         ("bal3n",f"results/iclr/e4_balanced3_natural/windows/bal3n_s{s}.npz")]:
            if Path(root).is_file():
                z=np.load(root); bal[f"{tag}_s{s}"]=(z["ctx"],z["fut"],z["oracle"])
    for name in a.models.split(","):
        for seed in [int(x) for x in a.seeds.split(",")]:
            for init in ["pretrained","random"]:
                try: embed,dim,native=make_extractor(name,init,seed,a.device)
                except Exception as e:
                    print(f"[{name}/{init}/s{seed}] extractor FAIL {e}"); continue
                n=150; ntr=90
                ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,seed)
                tr_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
                te_idx=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
                tr_lab,te_lab=kk[tr_idx],kk[te_idx]
                H=embed(ctx)
                # ---- A recognition (orig+shuf) ----
                for cond,Hc in [("orig",H),("shuf",embed(np.stack([np.random.default_rng(0).permutation(w) for w in ctx])))]:
                    _,p=softmax_predict(Hc[te_idx],*softmax_regression(Hc[tr_idx],tr_lab,n_iter=2000))
                    rows.append({"model":name,"init":init,"seed":seed,"task":f"A_recog_{cond}","test_set":"synth5",
                                 "accuracy":round(accuracy(p,te_lab),4)})
                # ---- B readout ----
                pred,hp=lin_readout(H[tr_idx],fut[tr_idx],H[te_idx],seed)
                mse_b=float(np.mean((pred-fut[te_idx])**2))
                rows.append({"model":name,"init":init,"seed":seed,"task":"B_readout","test_set":"synth5","mse":round(mse_b,4),"head_params":hp})
                # ---- C routing on balanced sets ----
                for tag in [k for k in bal if k.endswith(f"_s{seed}")]:
                    bctx,bfut,bor=bal[tag]
                    Hb=embed(bctx)
                    tr3=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
                    _,rp=softmax_predict(Hb,*softmax_regression(H[tr3],kk[tr3],n_iter=2000))
                    err=np.zeros((len(bctx),3))
                    for j,e in enumerate(EXPERTS):
                        for i in range(len(bctx)): err[i,j]=float(np.mean((e.predict(bctx[i],16)-bfut[i])**2))
                    ba,mf1,rec=recog_metrics(rp,bor)
                    mse_c=float(np.mean(err[np.arange(len(rp)),rp]))
                    rows.append({"model":name,"init":init,"seed":seed,"task":"C_routing","test_set":tag,
                                 "balanced_accuracy":round(ba,4),"macro_f1":round(mf1,4),
                                 "recall_trend":round(rec[0],4),"recall_periodic":round(rec[1],4),"recall_local":round(rec[2],4),
                                 "mse":round(mse_c,4)})
                    print(f"[{name}/{init}/s{seed}] C_{tag} ba={ba:.3f} mse={mse_c:.3f}")
                # ---- native forecast ----
                if native is not None:
                    te_idx5=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
                    try:
                        fc=native(ctx[te_idx5]); mse_n=float(np.mean((fc-fut[te_idx5])**2))
                        rows.append({"model":name,"init":init,"seed":seed,"task":"native_forecast","test_set":"synth5","mse":round(mse_n,4)})
                        print(f"[{name}/{init}/s{seed}] native mse={mse_n:.3f}")
                    except Exception as ne:
                        print(f"[{name}/{init}/s{seed}] native FAIL {type(ne).__name__}: {ne}")
                print(f"[{name}/{init}/s{seed}] A_orig=", [r['accuracy'] for r in rows if r['model']==name and r['seed']==seed and r['init']==init and r['task']=='A_recog_orig'], f" B={mse_b:.3f}")
    fields=["model","init","seed","task","test_set","n_test","accuracy","balanced_accuracy","macro_f1","recall_trend","recall_periodic","recall_local","mse","head_params"]
    with open(a.out/"metrics.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})
    print(f"saved {a.out/'metrics.csv'} rows={len(rows)}")

if __name__=="__main__":
    main()
