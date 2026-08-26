"""ICLR P2: serialization robustness - decimal precision (1/2/3) + rounding floor.

Frozen official Qwen3-8B-Base generation with 1/2/3-decimal numeric prompts
(parse + MSE/oracle).  Also computes the 2-decimal quantization floor of the
true future, i.e. how much error two-decimal rounding itself contributes.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
from scripts.run_frozen_probe import _load_model
KINDS=["trend","periodic","local","mixture","regime"]

def fmt(vals, ndec):
    clipped=np.clip(vals,-9.99,9.99)
    return " ".join(f"{float(v):+.{ndec}f}" for v in clipped)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/9950backfile/chenjiahui/ARForecast-Diagnosis/models/qwen3-8b-base")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-kind", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/serialization/qwen3_8b_official.json"))
    a=ap.parse_args()
    s=a.seed
    ctx,fut,_=build_labeled_windows(KINDS,64,16,150,s)
    n_test=60; per=a.per_kind
    test_idx=np.concatenate([np.arange(k*150+90,(k+1)*150) for k in range(5)])
    gi=np.concatenate([np.arange(k*n_test,k*n_test+per) for k in range(5)])
    C=ctx[test_idx[gi]]; F=fut[test_idx[gi]]
    experts=[TrendExpert(),PeriodicExpert(),LocalExpert()]
    err=np.zeros((len(C),3))
    for j,e in enumerate(experts):
        for i in range(len(C)): err[i,j]=float(np.mean((e.predict(C[i],16)-F[i])**2))
    oracle=err.min(axis=1)
    model,tok=_load_model(a.model_path,a.device,False)
    report={"model":"official Qwen3-8B-Base","seed":s,"per_kind":per}
    # 2-decimal rounding floor of the TRUE future
    floor=np.mean((np.round(F,2)-F)**2)
    report["two_decimal_floor_of_true_future"]=float(floor)
    for ndec in [1,2,3]:
        pat=re.compile(r"(?<![0-9])[+-]?\d+\.\d{"+str(ndec)+r"}")
        prompts=["Forecast the next values of this normalized time series.\nHistory: "+fmt(r,ndec)+"\nForecast:" for r in C]
        mses=[]; parsed=0
        for st in range(0,len(prompts),8):
            enc=tok(prompts[st:st+8],return_tensors="pt",padding=True,add_special_tokens=False).to(a.device)
            gen=model.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id)
            ilen=enc["input_ids"].size(1)
            for r in range(gen.size(0)):
                text=tok.decode(gen[r,ilen:],skip_special_tokens=True)
                nums=[float(x) for x in re.findall(pat,text)][:16]
                i=st+r
                if len(nums)==16: parsed+=1; mses.append(float(np.mean((np.array(nums)-F[i])**2)))
        report[f"decimals_{ndec}"]={"parse_rate":parsed/len(C),
            "mse":float(np.mean(mses)) if mses else None,
            "ratio":float(np.mean(mses)/np.mean(oracle)) if mses else None}
        print(f"[{ndec} dec] parse={parsed/len(C):.3f} ratio={report[f'decimals_{ndec}']['ratio']:.2f}")
    print(f"two-decimal floor of true future = {floor:.4f}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
