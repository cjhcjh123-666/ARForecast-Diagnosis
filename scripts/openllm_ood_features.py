"""Dump OOD (bal3/bal3n) features for open LMs so Table-B analyses are CPU-only."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts.openllm_suite import resolve_path, load_lm, embed
OUT=REPO/"results/open_llm_suite/features"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-key",required=True); ap.add_argument("--model-id",required=True)
    ap.add_argument("--path",default=None); ap.add_argument("--device",default="cuda:1")
    ap.add_argument("--seeds",default="7,17,27"); ap.add_argument("--inits",default="pretrained,random")
    a=ap.parse_args()
    path=resolve_path(a.model_id,local=a.path)
    for seed in [int(x) for x in a.seeds.split(",")]:
        bal={}
        for tag,root in [("bal3",f"results/iclr/e4_balanced3/windows/bal3_s{seed}.npz"),
                         ("bal3n",f"results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz")]:
            z=np.load(root); bal[tag]=(z["ctx"],z["oracle"])
        for init in [x for x in a.inits.split(",") if x]:
            model,tok=load_lm(path,init,seed,a.device)
            try:
                h3=embed(model,tok,bal["bal3"][0],a.device,"last")
                h3n=embed(model,tok,bal["bal3n"][0],a.device,"last")
                np.savez(OUT/f"ood_{a.model_key}_{init}_s{seed}.npz",
                         h_bal3=h3.astype(np.float32),h_bal3n=h3n.astype(np.float32),
                         bal3_oracle=bal["bal3"][1],bal3n_oracle=bal["bal3n"][1])
                print(f"[{a.model_key}/{init}/s{seed}] saved {h3.shape}",flush=True)
            finally:
                del model; import torch; torch.cuda.empty_cache()
if __name__=="__main__": main()
