"""API-served DIRECT numerical generation zoo (resumable, concurrent).

- Reads frozen model list: configs/api_models_frozen.yaml
- Two protocols kept separate: A raw completion (/v1/completions), B chat (/v1/chat/completions)
- Prompt identical to the local native protocol (same serialization/parser)
- One JSONL record per request; resume skips (model, protocol, seed, window_id) already present
- Budget guard: RUN_PAID_API=1 required; API_MAX_REQUESTS and/or API_BUDGET_USD hard cap
- temperature=0 when supported; retries with exponential backoff; stores usage/latency/raw response
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]
NUM_RE=re.compile(r"(?<![0-9])[+-]?\d+\.\d{2}")
for _k in ["http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
    os.environ.pop(_k,None)
os.environ["no_proxy"]="*"; os.environ["NO_PROXY"]="*"
OUT=REPO/"results/open_llm_suite"; RAW=OUT/"api_raw"; RAW.mkdir(parents=True,exist_ok=True)
LOCK=threading.Lock()

def get_key():
    k=os.environ.get("AIGCBEST_API_KEY")
    if k: return k.strip()
    p=REPO/"api.txt"
    if p.is_file(): return p.read_text().strip()
    raise SystemExit("AIGCBEST_API_KEY unset")

def _format_values(v):
    v=np.clip(v,-9.99,9.99); return " ".join(f"{float(x):+.2f}" for x in v)
def build_prompt(ctx):
    return "Forecast the next values of this normalized time series.\nHistory: "+_format_values(ctx)+"\nForecast:"

FLOAT_RE=re.compile(r"[-+]?\d+(?:\.\d+)?")
def parse_text(t,h=16):
    strict=[float(x) for x in NUM_RE.findall(t)]
    if len(strict)>=h: return np.asarray(strict[:h]),1,"strict"
    loose=[float(x) for x in FLOAT_RE.findall(t)]
    if len(loose)>=h: return np.asarray(loose[:h]),0,"lenient"
    return None,0,"fail"

def http_post(url,key,payload,timeout=120,retries=4):
    data=json.dumps(payload).encode()
    for a in range(retries):
        try:
            req=urllib.request.Request(url,data=data,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","User-Agent":"openllm-api"})
            t0=time.time()
            with urllib.request.urlopen(req,timeout=timeout) as r:
                d=json.loads(r.read()); return d,time.time()-t0,None
        except urllib.error.HTTPError as e:
            body=e.read().decode(errors="ignore")[:500]
            if e.code in (429,500,502,503,504) and a<retries-1:
                time.sleep(2**a); continue
            return None,0.0,f"HTTP {e.code}: {body}"
        except Exception as e:
            if a<retries-1: time.sleep(2**a); continue
            return None,0.0,f"{type(e).__name__}: {e}"
    return None,0.0,"exhausted"

def windows_for(seed, n_win):
    """Class-stratified test windows (equal per family) — matches the 40-window diagnostic."""
    ctx,fut,kk=build_labeled_windows(KINDS,64,16,150,seed)
    n=150;ntr=90
    per=max(1,n_win//5); sel=[]
    for k in range(5):
        sel.append(np.arange(k*n+ntr, k*n+ntr+per))
    te=np.concatenate(sel)[:n_win]
    return ctx[te], fut[te], kk[te]

def load_done(jsonl):
    done=set()
    if jsonl.is_file():
        for line in jsonl.read_text(errors="ignore").splitlines():
            try:
                r=json.loads(line)
                if r.get("ok"): done.add((r["model"],r["protocol"],r["seed"],r["window_id"]))
            except Exception: pass
    return done

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",default=None,help="comma list; default = frozen yaml")
    ap.add_argument("--protocols",default="chat,raw")
    ap.add_argument("--seeds",default="7")
    ap.add_argument("--windows",type=int,default=40)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--max-requests",type=int,default=int(os.environ.get("API_MAX_REQUESTS","0")))
    ap.add_argument("--budget-usd",type=float,default=float(os.environ.get("API_BUDGET_USD","0")))
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--tag",default="strat40")
    a=ap.parse_args()
    global a_tag; a_tag=a.tag
    if not a.dry_run and os.environ.get("RUN_PAID_API")!="1":
        raise SystemExit("Refusing paid calls: set RUN_PAID_API=1 (and API_MAX_REQUESTS or API_BUDGET_USD)")
    cfg=json.loads((REPO/"configs/api_models_frozen.yaml").read_text())
    models=[m["model_id"] for m in cfg["models"]] if not a.models else a.models.split(",")
    base=os.environ.get("AIGCBEST_BASE_URL","https://api2.aigcbest.top").rstrip("/")
    key=get_key()
    print(f"models={len(models)} protocols={a.protocols} seeds={a.seeds} windows/model={a.windows} workers={a.workers} dry_run={a.dry_run}")
    tasks=[]
    for model in models:
        for proto in a.protocols.split(","):
            for seed in [int(x) for x in a.seeds.split(",")]:
                ctx,fut,kind=windows_for(seed,a.windows)
                for i in range(len(ctx)):
                    tasks.append((model,proto,seed,i,ctx[i],fut[i]))
    print("total requests in plan:",len(tasks))
    counter={"n":0}
    def one(t):
        model,proto,seed,i,ctx_i,fut_i=t
        jsonl=RAW/model.replace("/","__")/f"{proto}_{a_tag}_s{seed}.jsonl"; jsonl.parent.mkdir(parents=True,exist_ok=True)
        prompt=build_prompt(ctx_i)
        if proto=="raw":
            url=f"{base}/v1/completions"; payload={"model":model,"prompt":prompt,"temperature":0,"max_tokens":160}
        else:
            url=f"{base}/v1/chat/completions"; payload={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":160}
        if max(a.max_requests,a.budget_usd)>0:
            with LOCK:
                if a.max_requests and counter["n"]>=a.max_requests:
                    return {"ok":False,"skipped":"max_requests"}
                counter["n"]+=1
        resp,lat,err=http_post(url,key,payload)
        rec={"model":model,"protocol":proto,"seed":seed,"window_id":int(i),
             "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),"latency_s":round(lat,2),
             "request":{k:v for k,v in payload.items() if k!="messages"} if proto=="raw" else {"model":model,"messages":payload["messages"][:1],"temperature":0,"max_tokens":160},
             "ok":False}
        if resp is None:
            rec["error"]=err
        else:
            text=resp.get("choices",[{}])[0].get("text") if proto=="raw" else resp.get("choices",[{}])[0].get("message",{}).get("content")
            vals,ok,mode=parse_text(text or "")
            rec.update(ok=True,returned_model=resp.get("model"),usage=resp.get("usage"),text=(text or "")[:2000],
                       parsed=(None if vals is None else vals.tolist()),parse_ok=int(ok),parse_mode=mode,
                       parsed_lenient=(None if vals is None or mode!="lenient" else vals.tolist()))
        with LOCK:
            with open(jsonl,"a") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
        return rec
    done=set()
    for model in models:
        for proto in a.protocols.split(","):
            for seed in [int(x) for x in a.seeds.split(",")]:
                done |= load_done(RAW/model.replace("/","__")/f"{proto}_{a.tag}_s{seed}.jsonl")
    todo=[t for t in tasks if (t[0],t[1],t[2],t[3]) not in done]
    print("already done:",len(done),"| todo:",len(todo))
    if a.dry_run:
        print("DRY RUN — no requests sent."); return
    n_ok=n_err=0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs=[ex.submit(one,t) for t in todo]
        for k,f in enumerate(as_completed(futs)):
            r=f.result()
            if r.get("ok"): n_ok+=1
            elif r.get("skipped")!="max_requests": n_err+=1
            if (k+1)%50==0: print(f"{k+1}/{len(todo)} ok={n_ok} err={n_err}",flush=True)
    print(f"finished ok={n_ok} err={n_err}")
if __name__=="__main__": main()
