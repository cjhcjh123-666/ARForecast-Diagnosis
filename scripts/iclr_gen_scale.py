"""ICLR P0-2b: scale up the generation evaluation (200 windows/seed, official 8B).

The native-generation claim (parse ~100%, MSE/oracle ~4x) is currently based on
40 windows/seed.  This runs 200 stratified windows/seed (40/kind) on the
official Qwen3-8B-Base to confirm the gap is stable.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _format_values, _load_model
KINDS=["trend","periodic","local","mixture","regime"]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/9950backfile/chenjiahui/ARForecast-Diagnosis/models/qwen3-8b-base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--per-kind", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/gen_scale/qwen3_8b_official.json"))
    a=ap.parse_args()
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]
    report={"model":"official Qwen3-8B-Base","per_kind":a.per_kind,"seeds":[int(s) for s in a.seeds.split(",")]}
    model,tok=_load_model(a.model_path,a.device,False)
    for s in [int(x) for x in a.seeds.split(",")]:
        ctx,fut,_=build_labeled_windows(KINDS,64,16,150,s)
        n_test=60; per=a.per_kind
        test_idx=np.concatenate([np.arange(k*150+90,(k+1)*150) for k in range(5)])
        gi=np.concatenate([np.arange(k*n_test,k*n_test+min(per,n_test)) for k in range(5)])
        C=ctx[test_idx[gi]]; F=fut[test_idx[gi]]
        err=np.zeros((len(C),3))
        for j,e in enumerate(experts):
            for i in range(len(C)): err[i,j]=float(np.mean((e.predict(C[i],16)-F[i])**2))
        oracle=err.min(axis=1)
        prompts=["Forecast the next values of this normalized time series.\nHistory: "+_format_values(r)+"\nForecast:" for r in C]
        mses=[]; parsed=0
        for st in range(0,len(prompts),8):
            enc=tok(prompts[st:st+8],return_tensors="pt",padding=True,add_special_tokens=False).to(a.device)
            gen=model.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id)
            ilen=enc["input_ids"].size(1)
            for r in range(gen.size(0)):
                text=tok.decode(gen[r,ilen:],skip_special_tokens=True)
                nums=[float(x) for x in re.findall(r"(?<![0-9])[+-]?\d+\.\d{2}",text)][:16]
                i=st+r
                if len(nums)==16:
                    parsed+=1; mses.append(float(np.mean((np.array(nums)-F[i])**2)))
        report[str(s)]={"windows":len(C),"parsed":parsed,"parse_rate":parsed/len(C),
                        "mse":float(np.mean(mses)) if mses else None,
                        "oracle_mse":float(np.mean(oracle)),
                        "ratio":float(np.mean(mses)/np.mean(oracle)) if mses else None}
        print(f"[seed {s}] parse={parsed}/{len(C)}={parsed/len(C):.3f} ratio={report[str(s)]['ratio']:.2f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"saved={a.out}")
if __name__=="__main__":
    main()
