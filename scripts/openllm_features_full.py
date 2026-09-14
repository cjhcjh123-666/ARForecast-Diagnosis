"""Consistent feature dump: clean750 + bal3 + bal3n for one model/init/seed set.

Guarantees the SAME model instance produces clean and OOD features, so CPU
re-analysis is exactly the runner protocol.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts.openllm_suite import resolve_path, load_lm, embed
from data.dynamics import build_labeled_windows
KINDS=["trend","periodic","local","mixture","regime"]
OUT=REPO/"results/open_llm_suite/features_full"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-key",required=True); ap.add_argument("--model-id",required=True)
    ap.add_argument("--path",default=None); ap.add_argument("--device",default="cuda:1")
    ap.add_argument("--seeds",default="7,17,27"); ap.add_argument("--inits",default="pretrained,random")
    ap.add_argument("--extra-pool",default="",help="e.g. 'mean' -> also dump mean-pooled variants (pooling sensitivity)")
    a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    path=resolve_path(a.model_id,local=a.path)
    extra=[p for p in a.extra_pool.split(",") if p]
    for seed in [int(x) for x in a.seeds.split(",")]:
        ctx,fut,kk=build_labeled_windows(KINDS,64,16,150,seed)
        bal={}
        for tag,root in [("bal3",f"results/iclr/e4_balanced3/windows/bal3_s{seed}.npz"),
                         ("bal3n",f"results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz")]:
            z=np.load(root); bal[tag]=(z["ctx"],z["oracle"])
        for init in [x for x in a.inits.split(",") if x]:
            model,tok=load_lm(path,init,seed,a.device)
            try:
                H=embed(model,tok,ctx,a.device,"last")
                h3=embed(model,tok,bal["bal3"][0],a.device,"last")
                h3n=embed(model,tok,bal["bal3n"][0],a.device,"last")
                payload=dict(h_clean=H.astype(np.float32),clean_kind=kk,
                             h_bal3=h3.astype(np.float32),h_bal3n=h3n.astype(np.float32),
                             bal3_oracle=bal["bal3"][1],bal3n_oracle=bal["bal3n"][1],fut=fut)
                for p in extra:  # same forward, different read-out -> pooling sensitivity
                    payload[f"h_clean_{p}"]=embed(model,tok,ctx,a.device,p).astype(np.float32)
                    payload[f"h_bal3_{p}"]=embed(model,tok,bal["bal3"][0],a.device,p).astype(np.float32)
                    payload[f"h_bal3n_{p}"]=embed(model,tok,bal["bal3n"][0],a.device,p).astype(np.float32)
                np.savez(OUT/f"full_{a.model_key}_{init}_s{seed}.npz", **payload)
                print(f"[{a.model_key}/{init}/s{seed}] saved clean+ood",flush=True)
            finally:
                del model; import torch; torch.cuda.empty_cache()
if __name__=="__main__": main()
