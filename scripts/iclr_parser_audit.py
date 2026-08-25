"""ICLR P0-4: audit raw generation outputs for the one-digit-integer regex.

The parser `[+-]?\\d\\.\\d\\d` only matches a single digit before the decimal.
Re-run frozen generation on the stratified 40-window set for official 8B and
0.6B, SAVE the raw completion text, and audit:
  - any raw numeric token with abs(value) >= 10 (which the one-digit regex
    would truncate to its last digit, e.g. 12.34 -> 2.34)
  - fraction of completions where the current parser differs from a
    multi-digit parser `[+-]?\\d+\\.\\d{2}`
Report counts; if zero large values, the one-digit parser is safe on this data.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from data.dynamics import build_labeled_windows
from scripts.run_frozen_probe import _format_values, _load_model
KINDS=["trend","periodic","local","mixture","regime"]
P1 = re.compile(r"[+-]?\d\.\d\d")           # current (one digit before dot)
P2 = re.compile(r"[+-]?\d+\.\d{2}")         # multi-digit candidate
P2b = re.compile(r"(?<![0-9])[+-]?\d+\.\d{2}(?![0-9])")  # with boundaries

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/9950backfile/chenjiahui/ARForecast-Diagnosis/models/qwen3-8b-base")
    ap.add_argument("--model-tag", default="qwen3_8b_official")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--out", type=Path, default=Path("results/iclr/parser_audit/qwen3_8b_official.json"))
    a=ap.parse_args()
    report={"model":a.model_tag,"regex_current":str(P1.pattern),"regex_multi":str(P2b.pattern)}
    for seed in [int(s) for s in a.seeds.split(",")]:
        contexts,futures,_=build_labeled_windows(KINDS,64,16,150,seed)
        n_test=60
        test_idx=np.concatenate([np.arange(k*150+90,(k+1)*150) for k in range(5)])
        gen_idx=np.concatenate([np.arange(k*n_test,k*n_test+8) for k in range(5)])
        ctx=contexts[test_idx[gen_idx]]
        model,tok=_load_model(a.model_path,a.device,False)
        prompts=["Forecast the next values of this normalized time series.\nHistory: "+_format_values(r)+"\nForecast:" for r in ctx]
        raw=[]
        for start in range(0,len(prompts),8):
            enc=tok(prompts[start:start+8],return_tensors="pt",padding=True,add_special_tokens=False).to(a.device)
            gen=model.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id)
            ilen=enc["input_ids"].size(1)
            for r in range(gen.size(0)):
                raw.append(tok.decode(gen[r,ilen:],skip_special_tokens=True))
        del model; torch.cuda.empty_cache()
        # audit: any number with >=2 digits before decimal, or abs>=10
        big=[]
        for i,txt in enumerate(raw):
            for m in re.finditer(r"[+-]?\d+\.\d\d", txt):
                num=float(m.group())
                if abs(num)>=10 or (m.group().lstrip('+-').split('.')[0]).__len__()>1:
                    big.append((i,m.group(),num))
        # parse-rate diff between P1 and P2b (first 16 numbers)
        def parse(txt,pat):
            return [float(x) for x in re.findall(pat,txt)][:16]
        diff_windows=0
        for i,txt in enumerate(raw):
            v1=parse(txt,P1); v2=parse(txt,P2b)
            if v1!=v2: diff_windows+=1
        report[str(seed)]={"n_raw":len(raw),
            "windows_where_parser_differs":diff_windows,
            "large_values_abs_ge10":len(big),
            "examples":big[:5]}
        print(f"[seed {seed}] raw={len(raw)} parser_diff_windows={diff_windows} abs_ge10={len(big)}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"saved={a.out}")

if __name__=="__main__":
    main()
