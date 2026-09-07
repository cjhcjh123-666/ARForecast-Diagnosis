"""Dump frozen Qwen features (any size) for routing-level followups.

Sizes: qwen3_0.6b (1024), qwen3_1.7b (2048), qwen3_8b_base (4096).
For each (seed, init): embed the 270 clean windows + bal3/bal3n OOD contexts.
random = architecture-native from-config init (no released weights), seeded.
Output: qwen_feats/{tag}_{init}_s{seed}.npz  (tag per model)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from data.dynamics import build_labeled_windows
from scripts.iclr_tsfm_deliver import KINDS
PATHS={"qwen3_0.6b":"/9950backfile/chenjiahui/hf_cache/Qwen3-0.6B",
       "qwen3_1.7b":"/9950backfile/chenjiahui/hf_cache/Qwen3-1.7B",
       "qwen3_8b_base":str(REPO/"models/qwen3-8b-base")}

def load_qwen(path, random, seed, device):
    import transformers.modeling_utils as _mu
    if not isinstance(_mu.ALL_PARALLEL_STYLES,(set,frozenset)):
        _mu.ALL_PARALLEL_STYLES=frozenset({"tp","block","sharded","pp","sequence","rowwise","colwise","naive","serial","manual","flex","ddp"})
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    if random:
        torch.manual_seed(seed)
        cfg=AutoConfig.from_pretrained(path, trust_remote_code=True)
        model=AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(torch.bfloat16)
    else:
        devidx=int(device.split(":")[1]) if ":" in device else 0
        model=AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa", device_map={"":devidx})
    return model.to(device).eval(), tok

def embed(model, tok, ctx, device, bs=24):
    sys.path.insert(0,str(REPO))
    from scripts.run_frozen_probe import _prompts
    from transformers import AutoTokenizer as _T
    out=[]
    with torch.no_grad():
        for st in range(0,len(ctx),bs):
            chunk=ctx[st:st+bs]
            enc=tok(_prompts(chunk), return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            h=model(**enc, output_hidden_states=True).hidden_states[-1]
            ln=enc["attention_mask"].sum(1)-1; ridx=torch.arange(len(chunk),device=device)
            out.append(h[ridx,ln].float().cpu().numpy())
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_8b_base", help="one of qwen3_0.6b,qwen3_1.7b,qwen3_8b_base")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--inits", default="pretrained,random")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("qwen_feats"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    assert a.model in PATHS
    for seed in [int(x) for x in a.seeds.split(",")]:
        n=150; ntr=90
        ctx,_,kk=build_labeled_windows(KINDS,64,16,n,seed)
        clean=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
        bal={}
        for tag,root in [("bal3",f"results/iclr/e4_balanced3/windows/bal3_s{seed}.npz"),
                         ("bal3n",f"results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz")]:
            z=np.load(root); bal[tag]=(z["ctx"],z["fut"],z["oracle"])
        for ini in [x for x in a.inits.split(",") if x]:
            model,tok=load_qwen(PATHS[a.model], ini=="random", seed, a.device)
            Hc=embed(model,tok,ctx[clean],a.device)
            Hb3=embed(model,tok,bal["bal3"][0],a.device)
            Hb3n=embed(model,tok,bal["bal3n"][0],a.device)
            np.savez(a.out/f"{a.model}_{ini}_s{seed}.npz", h_clean=Hc,h_bal3=Hb3,h_bal3n=Hb3n,
                     clean_family=kk[clean], bal3_oracle=bal["bal3"][2], bal3n_oracle=bal["bal3n"][2])
            print(f"[{a.model}/{ini}/s{seed}] saved {Hc.shape}",flush=True)
            del model,tok; import gc; gc.collect(); torch.cuda.empty_cache()
if __name__=="__main__":
    main()
