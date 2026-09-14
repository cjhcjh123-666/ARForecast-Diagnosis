"""Open-LLM cross-family attribution suite (protocol identical to the Qwen runs).

Per (model, init, seed):
  A  clean 5-family recognition (orig + per-window shuffle)
  B  frozen numerical readout: linear head (H=16) and MLP64 head
  C  family-label 3-expert balanced-OOD routing (clean270 -> bal3/bal3n)
  C2 oracle-label routing (argmax/argmin future-MSE expert labels on clean270)
  R  MLP64 router (same clean270 family labels)
  P  pooling sensitivity: last-nonpad vs mean-nonpad
Native numeric generation (greedy) with the historical prompt/parser.

Only frozen backbones; random = architecture-native from-config (seeded).
Results -> results/open_llm_suite/{raw,features,tables}
"""
from __future__ import annotations
import argparse, csv, json, math, re, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
# some envs ship a broken torchaudio; transformers imports it lazily -> stub if broken
try:
    import torchaudio  # noqa: F401
except Exception:
    import types as _types
    _ta=_types.ModuleType("torchaudio"); _ta.__file__="<torchaudio-stub>"
    import importlib.machinery as _im
    _ta.__spec__=_im.ModuleSpec("torchaudio", loader=None)
    _ta.__version__="0.0.0-stub"
    import sys as _sys; _sys.modules.setdefault("torchaudio", _ta)
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
SEEDS=[7,17,27]
EXTRA_SEEDS=[37,47,57,67,77,87,97]
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]
NEW_EXPERTS=[TrendExpert(),TrendSeasonal(),PeriodicExpert(),LocalExpert(),AR10()] if False else None

# ---------------- prompt/serialization (identical to run_frozen_probe) ----------------
def _format_values(values):
    clipped=np.clip(values,-9.99,9.99)
    return " ".join(f"{float(v):+.2f}" for v in clipped)
def prompts(contexts):
    return ["Forecast the next values of this normalized time series.\nHistory: "+_format_values(r)+"\nForecast:" for r in contexts]
NUM_RE=re.compile(r"(?<![0-9])[+-]?\d+\.\d{2}")

def parse_numbers(text, h=16):
    vals=[float(m) for m in NUM_RE.findall(text)]
    if len(vals)<h: return None, len(vals)
    return np.asarray(vals[:h],dtype=float), len(vals)

# ---------------- model loading ----------------
def resolve_path(model_id, cache="/9950backfile/chenjiahui/hf_cache/hub", local=None):
    if local and Path(local).is_dir(): return str(local)
    from huggingface_hub import snapshot_download
    import os
    os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
    for k in ["http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
        os.environ.pop(k,None)
    os.environ["no_proxy"]="*"; os.environ["HF_HUB_DISABLE_XET"]="1"
    return snapshot_download(model_id, cache_dir=cache, local_files_only=False)

def load_lm(path, init, seed, device, trust_remote_code=True):
    import transformers.modeling_utils as _mu
    if hasattr(_mu,"ALL_PARALLEL_STYLES") and not isinstance(_mu.ALL_PARALLEL_STYLES,(set,frozenset)):
        _mu.ALL_PARALLEL_STYLES=frozenset({"tp","block","sharded","pp","sequence","rowwise","colwise",
            "naive","serial","manual","flex","ddp","colwise_rep","rowwise_rep","colwise_sharded",
            "rowwise_sharded","local_colwise","local_rowwise","gather","local","replicate"})
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    # NOTE: the attention implementation MUST be identical for pretrained and random,
    # otherwise the two branches run different inference code. gemma2 under sdpa+static
    # cache triggers torch.compile (can_compile=True) and dies inside dynamo, so both
    # branches use eager for gemma.
    _attn="eager" if "gemma" in str(path).lower() else "sdpa"
    if init=="random":
        torch.manual_seed(seed)
        cfg=AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        def _build(attn):
            try:
                cfg._attn_implementation=attn
            except Exception:
                pass
            try:
                # build directly in bf16 to avoid a large fp32 CPU copy (OOM-prone for >=3B)
                return AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16)
            except TypeError:
                return AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code).to(torch.bfloat16)
        try:
            model=_build(_attn)
        except ValueError as e:
            # e.g. DeepSeek-V2-Lite's remote code rejects sdpa: mirror the pretrained-branch fallback
            if "attention" in str(e).lower():
                model=_build("eager")
            else:
                raise
    else:
        dev=int(device.split(":")[1]) if ":" in device else 0
        try:
            model=AutoModelForCausalLM.from_pretrained(path, trust_remote_code=trust_remote_code,
                  torch_dtype=torch.bfloat16, attn_implementation=_attn, device_map={"":dev})
        except ValueError as e:
            if "attention" in str(e).lower() or "sdpa" in str(e).lower() or "colwise" in str(e).lower():
                model=AutoModelForCausalLM.from_pretrained(path, trust_remote_code=trust_remote_code,
                      torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"":dev})
            else:
                raise
    return model.to(device).eval(), tok

@torch.inference_mode()
def embed(model, tok, ctx, device, pooling="last", bs=16):
    out=[]
    for st in range(0,len(ctx),bs):
        chunk=ctx[st:st+bs]
        enc=tok(prompts(chunk),return_tensors="pt",padding=True,add_special_tokens=False).to(device)
        h=model(**enc,output_hidden_states=True,use_cache=False).hidden_states[-1]
        mask=enc["attention_mask"].bool()
        if pooling=="last":
            idx=mask.sum(1)-1
            out.append(h[torch.arange(len(chunk),device=device),idx].float().cpu().numpy())
        else:
            m=mask.unsqueeze(-1).to(h.dtype)
            out.append(((h*m).sum(1)/m.sum(1)).float().cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def generate_native(model, tok, ctx, device, h=16, max_new_tokens=160, bs=8):
    # compat shim: DeepSeek-V2-Lite's official remote code (modeling_deepseek.py) calls
    # DynamicCache.get_max_length(), removed in transformers>=4.49. Semantics preserved
    # (old get_max_length returned the configured max cache length; None = dynamic).
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache,"get_max_length"):
            DynamicCache.get_max_length=lambda self: self.get_max_cache_shape()
    except Exception:
        pass
    model.eval(); preds=[]; parses=[]; raws=[]
    for st in range(0,len(ctx),bs):
        chunk=ctx[st:st+bs]
        enc=tok(prompts(chunk),return_tensors="pt",padding=True,add_special_tokens=False).to(device)
        gen=model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,
                           pad_token_id=tok.pad_token_id,use_cache=True)
        new=gen[:,enc["input_ids"].shape[1]:]
        for row in new:
            txt=tok.decode(row,skip_special_tokens=True)
            vals,n=parse_numbers(txt,h); raws.append(txt)
            if vals is None: preds.append(np.full(h,np.nan)); parses.append(0)
            else: preds.append(vals); parses.append(1)
    return np.asarray(preds), np.asarray(parses), raws

# ---------------- metrics / router helpers ----------------
def recog_metrics3(pred,oracle):
    cm=np.zeros((3,3),int)
    for t,p in zip(oracle,pred):
        if t<3 and p<3: cm[t,p]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(3)])
    prec=np.array([cm[:,i][i]/(cm[:,i].sum()+1e-12) if cm[:,i].sum() else 0 for i in range(3)])
    f1=float(np.mean([2*p*r/(p+r+1e-12) for p,r in zip(prec,rec)]))
    return float(rec.mean()),f1,rec,cm

def expert_errors(ctx,fut,experts=EXPERTS):
    E=np.zeros((len(ctx),len(experts)))
    for j,e in enumerate(experts):
        for i in range(len(ctx)):
            E[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
    return E

def mlp_router(Xtr,ytr,Xte,h=64,steps=600,lr=1e-3,wd=1e-2,seed=0):
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(Xtr.shape[1],h),nn.ReLU(),nn.Linear(h,int(ytr.max())+1)).double()
    Xt=torch.from_numpy(Xtr).double(); yt=torch.from_numpy(ytr).long()
    mu=Xt.mean(0); sd=Xt.std(0)+1e-9
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    Xn=(Xt-mu)/sd
    for _ in range(steps):
        opt.zero_grad(); loss=nn.functional.cross_entropy(net(Xn),yt); loss.backward(); opt.step()
    with torch.no_grad(): return net((torch.from_numpy(Xte).double()-mu)/sd).argmax(1).numpy()

def lin_head(Xtr,Ytr,Xte,seed,wd=0.01,epochs=300):
    torch.manual_seed(seed); net=nn.Linear(Xtr.shape[1],Ytr.shape[1])
    Xt=torch.from_numpy(Xtr).float(); Yt=torch.from_numpy(Ytr).float()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(); l=nn.functional.mse_loss(net(Xt),Yt); l.backward(); opt.step()
    with torch.no_grad(): return net(torch.from_numpy(Xte).float()).numpy(), Xtr.shape[1]*Ytr.shape[1]+Ytr.shape[1]

def mlp_head(Xtr,Ytr,Xte,seed,h=256,steps=400,lr=1e-3,wd=1e-4):
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(Xtr.shape[1],h),nn.ReLU(),nn.Linear(h,Ytr.shape[1]))
    Xt=torch.from_numpy(Xtr).float(); Yt=torch.from_numpy(Ytr).float()
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); l=nn.functional.mse_loss(net(Xt),Yt); l.backward(); opt.step()
    with torch.no_grad(): return net(torch.from_numpy(Xte).float()).numpy(), sum(p.numel() for p in net.parameters())

def shuffled(ctx,seed):
    out=np.empty_like(ctx)
    for i,w in enumerate(ctx):
        out[i]=np.random.default_rng(100000*seed+i).permutation(w)
    return out

def load_bal(seed):
    bal={}
    for tag,root in [("bal3",f"results/iclr/e4_balanced3/windows/bal3_s{seed}.npz"),
                     ("bal3n",f"results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz")]:
        if Path(root).is_file():
            z=np.load(root); bal[tag]=(z["ctx"],z["fut"],z["oracle"])
    return bal

# ---------------- per-model runner ----------------
def run_model(key, model_id, path, device, seeds, tasks, outdir, inits=("pretrained","random"), native=False):
    outdir=Path(outdir); (outdir/"features").mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        n=150; ntr=90
        ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,seed)
        tr_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
        te_idx=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
        tr_lab,te_lab=kk[tr_idx],kk[te_idx]
        clean270=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
        ctx_sh=shuffled(ctx,seed)
        bal=load_bal(seed)
        for init in inits:
            t0=time.time()
            model,tok=load_lm(path,init,seed,device)
            try:
                H=embed(model,tok,ctx,device,"last")
                Hs=embed(model,tok,ctx_sh,device,"last")
                Hmean=embed(model,tok,ctx,device,"mean") if "pool" in tasks else None
                if "A" in tasks:
                    _,p=softmax_predict(H[te_idx],*softmax_regression(H[tr_idx],tr_lab,n_iter=2000))
                    rows.append(dict(model=key,init=init,seed=seed,task="A_recog_orig",n_test=len(te_idx),accuracy=round(accuracy(p,te_lab),4)))
                    _,ps=softmax_predict(Hs[te_idx],*softmax_regression(Hs[tr_idx],tr_lab,n_iter=2000))
                    rows.append(dict(model=key,init=init,seed=seed,task="A_recog_shuf",n_test=len(te_idx),accuracy=round(accuracy(ps,te_lab),4)))
                if "B" in tasks:
                    pred_b,hp=lin_head(H[tr_idx],fut[tr_idx],H[te_idx],seed)
                    rows.append(dict(model=key,init=init,seed=seed,task="B_readout_linear",mse=round(float(np.mean((pred_b-fut[te_idx])**2)),4),head_params=hp))
                    pred_m,hp2=mlp_head(H[tr_idx],fut[tr_idx],H[te_idx],seed)
                    rows.append(dict(model=key,init=init,seed=seed,task="B_readout_mlp",mse=round(float(np.mean((pred_m-fut[te_idx])**2)),4),head_params=hp2))
                if "C" in tasks:
                    W,mu,sd=softmax_regression(H[clean270],kk[clean270],n_iter=2000)
                    for tag,(bctx,bfut,bor) in bal.items():
                        Hb=embed(model,tok,bctx,device,"last")
                        _,rp=softmax_predict(Hb,W,mu,sd)
                        E=expert_errors(bctx,bfut)
                        ba,f1,rec,_=recog_metrics3(rp,bor)
                        rows.append(dict(model=key,init=init,seed=seed,task=f"C_family_{tag}",balanced_accuracy=round(ba,4),
                                         macro_f1=round(f1,4),recall_trend=round(rec[0],4),recall_periodic=round(rec[1],4),
                                         recall_local=round(rec[2],4),routed_mse=round(float(np.mean(E[np.arange(len(rp)),rp])),4)))
                if "C2" in tasks:
                    Ec=expert_errors(ctx[clean270],fut[clean270]); oclean=Ec.argmin(1)
                    Wo,muo,sdo=softmax_regression(H[clean270],oclean,n_iter=2000)
                    for tag,(bctx,bfut,bor) in bal.items():
                        Hb=embed(model,tok,bctx,device,"last")
                        _,rp=softmax_predict(Hb,Wo,muo,sdo)
                        E=expert_errors(bctx,bfut); ba,f1,rec,_=recog_metrics3(rp,bor)
                        rows.append(dict(model=key,init=init,seed=seed,task=f"C2_oracle_{tag}",balanced_accuracy=round(ba,4),
                                         macro_f1=round(f1,4),routed_mse=round(float(np.mean(E[np.arange(len(rp)),rp])),4)))
                if "R" in tasks:
                    for tag,(bctx,bfut,bor) in bal.items():
                        Hb=embed(model,tok,bctx,device,"last")
                        pm=mlp_router(H[clean270],kk[clean270],Hb,seed=seed)
                        E=expert_errors(bctx,bfut); ba,f1,rec,_=recog_metrics3(pm,bor)
                        rows.append(dict(model=key,init=init,seed=seed,task=f"R_mlp_{tag}",balanced_accuracy=round(ba,4),
                                         routed_mse=round(float(np.mean(E[np.arange(len(pm)),pm])),4)))
                if "pool" in tasks:
                    for tag,(bctx,bfut,bor) in bal.items():
                        Hbm=embed(model,tok,bctx,device,"mean")
                        _,rp=softmax_predict(Hbm,*softmax_regression(Hmean[clean270],kk[clean270],n_iter=2000))
                        ba,_,_,_=recog_metrics3(rp,bor)
                        rows.append(dict(model=key,init=init,seed=seed,task=f"P_meanpool_{tag}",balanced_accuracy=round(ba,4)))
                if native:
                    te=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
                    preds,parses,raws=generate_native(model,tok,ctx[te],device)
                    ok=parses==1
                    mse=np.nanmean([np.mean((preds[i]-fut[te][i])**2) for i in range(len(te)) if ok[i]]) if ok.any() else float("nan")
                    rows.append(dict(model=key,init=init,seed=seed,task="N_native",parse_rate=round(float(ok.mean()),4),
                                     mse=round(float(mse),4) if mse==mse else ""))
                    np.savez(outdir/"features"/f"native_{key}_{init}_s{seed}.npz",parse=parses,pred=preds,fut=fut[te])
                np.savez(outdir/"features"/f"pw_{key}_{init}_s{seed}.npz",H=H.astype(np.float32))
                print(f"[{key}/{init}/s{seed}] done in {time.time()-t0:.0f}s rows={len(rows)}",flush=True)
            finally:
                del model; torch.cuda.empty_cache()
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-key",required=True)
    ap.add_argument("--model-id",required=True)
    ap.add_argument("--path",default=None)
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--seeds",default="7,17,27")
    ap.add_argument("--tasks",default="A,B,C")
    ap.add_argument("--native",action="store_true")
    ap.add_argument("--inits",default="pretrained,random")
    ap.add_argument("--out",type=Path,default=Path("results/open_llm_suite"))
    a=ap.parse_args()
    path=resolve_path(a.model_id,local=a.path)
    rows=run_model(a.model_key,a.model_id,path,a.device,[int(x) for x in a.seeds.split(",")],
                   set(a.tasks.split(",")),a.out,inits=tuple(x for x in a.inits.split(",") if x))
    a.out.mkdir(parents=True,exist_ok=True)
    tag="-".join(a.inits.split(","))
    f=a.out/"raw"/f"metrics_{a.model_key}_{tag}.csv"
    f.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    with open(f,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    print("wrote",len(rows),"rows ->",f)

if __name__=="__main__":
    main()
