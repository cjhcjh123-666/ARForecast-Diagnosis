"""Smoke test: tokenization, hidden shape/finiteness, final-nonpadding index,
and pretrained-vs-random token id identity (same tokenizer/serialization)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts.openllm_suite import resolve_path, load_lm, embed, prompts
from data.dynamics import build_labeled_windows
OUT=REPO/"results/open_llm_suite/raw"; OUT.mkdir(parents=True,exist_ok=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-key",required=True); ap.add_argument("--model-id",required=True)
    ap.add_argument("--path",default=None); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--seed",type=int,default=7)
    a=ap.parse_args()
    path=resolve_path(a.model_id,local=a.path)
    ctx,_,_=build_labeled_windows(["trend","periodic","local","mixture","regime"],64,16,150,a.seed)
    x=ctx[:10]
    rep={"model_key":a.model_key,"model_id":a.model_id,"path":path,"tokenizer_class":None}
    m1,t1=load_lm(path,"pretrained",a.seed,a.device)
    rep["tokenizer_class"]=type(t1).__name__
    enc1=t1(prompts(x),return_tensors="pt",padding=True,add_special_tokens=False)
    with torch.inference_mode():
        h1=m1(**enc1.to(a.device),output_hidden_states=True,use_cache=False).hidden_states[-1]
    H1=embed(m1,t1,x,a.device,"last")
    last_idx=enc1["attention_mask"].sum(1)-1
    rep.update(prompt_len_pad=int(enc1["input_ids"].shape[1]),
               tokens_per_context_mean=float(enc1["attention_mask"].sum(1).float().mean()),
               tokens_per_value=float(enc1["attention_mask"].sum(1).float().mean()/64),
               hidden_shape=list(h1.shape), hidden_dim=int(h1.shape[-1]),
               has_nan=bool(torch.isnan(h1).any()), has_inf=bool(torch.isinf(h1).any()),
               last_idx_max=int(last_idx.max()), last_idx_min=int(last_idx.min()),
               feature_shape=list(H1.shape), feature_finite=bool(np.isfinite(H1).all()),
               feature_norm_mean=float(np.linalg.norm(H1,axis=1).mean()))
    del m1; torch.cuda.empty_cache()
    m2,t2=load_lm(path,"random",a.seed,a.device)
    enc2=t2(prompts(x),return_tensors="pt",padding=True,add_special_tokens=False)
    a1,a2=enc1["input_ids"].cpu(),enc2["input_ids"].cpu(); m1_,m2_=enc1["attention_mask"].cpu(),enc2["attention_mask"].cpu()
    same=bool(torch.equal(a1,a2)) and bool(torch.equal(m1_,m2_))
    H2=embed(m2,t2,x,a.device,"last")
    rep.update(pretrained_random_token_ids_identical=same,
               random_feature_finite=bool(np.isfinite(H2).all()),
               pretrained_random_features_identical=bool(np.allclose(H1,H2)),
               max_abs_diff=float(np.abs(H1-H2).max()))
    del m2; torch.cuda.empty_cache()
    rep["status"]="OK" if (rep["feature_finite"] and rep["random_feature_finite"] and same and not rep["has_nan"] and not rep["has_inf"]) else "CHECK"
    p=OUT/f"smoke_{a.model_key}.json"; p.write_text(json.dumps(rep,indent=2))
    print(json.dumps(rep,indent=2)); print("saved",p)
if __name__=="__main__": main()
