"""ICLR P1: readout sanity (last-token vs mean-pooling) + exact marginal-matched
temporal control (sorted = same multiset, order destroyed).

Seed 7, official Qwen3-8B-Base, five-way recognition probe.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from scripts.run_frozen_probe import _load_model, _prompts
KINDS=["trend","periodic","local","mixture","regime"]

@torch.no_grad()
def extract(model,tok,contexts,device,pool="last",batch=32):
    out=[]
    for st in range(0,len(contexts),batch):
        chunk=_prompts(contexts[st:st+batch])
        enc=tok(chunk,return_tensors="pt",padding=True,add_special_tokens=False).to(device)
        h=model(**enc,output_hidden_states=True).hidden_states[-1]
        mask=enc["attention_mask"]
        if pool=="last":
            lens=mask.sum(1)-1; idx=torch.arange(len(chunk),device=device)
            v=h[idx,lens]
        else:  # mean over non-pad
            v=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
        out.append(v.float().cpu().numpy())
    return np.concatenate(out,axis=0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/9950backfile/chenjiahui/ARForecast-Diagnosis/models/qwen3-8b-base")
    ap.add_argument("--cache-root", type=Path, default=Path("results/iclr/probe_official/qwen3_8b"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/readout_sanity/qwen3_8b_official.json"))
    a=ap.parse_args()
    n=150; ntr=90; s=a.seed
    ctx,fut,kk=build_labeled_windows(KINDS,64,16,n,s)
    train_idx=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(5)])
    test_idx=np.concatenate([np.arange(k*n+ntr,(k+1)*n) for k in range(5)])
    tr_label,te_label=kk[train_idx],kk[test_idx]
    def probe(X):
        _,p=softmax_predict(X[test_idx],*softmax_regression(X[train_idx],tr_label,n_iter=2000))
        return accuracy(p,te_label)
    rep={"model":"official Qwen3-8B-Base","seed":s}
    for init in ["pretrained","random"]:
        model,tok=_load_model(a.model_path,a.device,init=="random")
        # use ORIGINAL windows for readout comparison
        X_last=extract(model,tok,ctx,a.device,"last")
        X_mean=extract(model,tok,ctx,a.device,"mean")
        # sorted (ascending) = exact marginal-matched, order destroyed
        X_sort=extract(model,tok,np.sort(ctx,axis=1),a.device,"last")
        del model; torch.cuda.empty_cache()
        rep[init]={"last_token":probe(X_last),"mean_pool":probe(X_mean),"sorted":probe(X_sort)}
        print(f"[{init}] last={rep[init]['last_token']:.3f} mean={rep[init]['mean_pool']:.3f} sorted={rep[init]['sorted']:.3f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(rep,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
