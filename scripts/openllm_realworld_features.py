"""Embed the 15 real-world dataset windows + clean750 for routing attribution."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts.openllm_suite import resolve_path, load_lm, embed
from data.dynamics import build_labeled_windows
KINDS=["trend","periodic","local","mixture","regime"]
WDIR=REPO/"results/iclr/multi_dataset_router/qwen3_8b_official/windows_qwen3_8b_official_s7"
OUT=REPO/"results/open_llm_suite/realworld_features"; OUT.mkdir(parents=True,exist_ok=True)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-key",required=True); ap.add_argument("--model-id",required=True)
    ap.add_argument("--path",default=None); ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--inits",default="pretrained,random")
    a=ap.parse_args(); path=resolve_path(a.model_id,local=a.path)
    ctx,_,kk=build_labeled_windows(KINDS,64,16,150,7)
    for init in [x for x in a.inits.split(",") if x]:
        model,tok=load_lm(path,init,7,a.device)
        try:
            H=embed(model,tok,ctx,a.device,"last")
            np.savez(OUT/f"rw_{a.model_key}_{init}.npz", h_clean=H.astype(np.float32), clean_kind=kk)
            for f in sorted(WDIR.glob("*_windows.npz")):
                ds=f.name.replace("_windows.npz","")
                z=np.load(f); h=embed(model,tok,z["ctx"],a.device,"last")
                np.savez(OUT/f"rw_{a.model_key}_{init}_{ds}.npz", h=h.astype(np.float32), fut=z["fut"])
                print(f"[{a.model_key}/{init}/{ds}] {h.shape}",flush=True)
        finally:
            del model; import torch; torch.cuda.empty_cache()
if __name__=="__main__": main()
