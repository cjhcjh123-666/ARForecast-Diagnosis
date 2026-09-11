"""Download + register open base LMs for the cross-family attribution suite.

Uses HF_ENDPOINT (hf-mirror) with proxies disabled; pins resolved revision SHA.
Records per-model status incl. gated/blocked. No paid API.
"""
from __future__ import annotations
import os, json, csv, time, traceback
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
for k in ["http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
    os.environ.pop(k, None)
os.environ["no_proxy"]="*"; os.environ["NO_PROXY"]="*"; os.environ["HF_HUB_DISABLE_XET"]="1"
CACHE="/9950backfile/chenjiahui/hf_cache/hub"
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/"results/open_llm_suite"; OUT.mkdir(parents=True,exist_ok=True)

MODELS=[
 # qwen3_8b_base is already local at models/qwen3-8b-base (skipped)

 ("qwen3_8b_base","Qwen/Qwen3-8B-Base",False),
 ("deepseek_llm_7b","deepseek-ai/deepseek-llm-7b-base",False),
 ("deepseek_v2_lite","deepseek-ai/DeepSeek-V2-Lite",False),
 ("mistral_7b_v03","mistralai/Mistral-7B-v0.3",False),
 ("olmo2_7b","allenai/OLMo-2-1124-7B",False),
 ("llama31_8b","meta-llama/Llama-3.1-8B",True),
 ("gemma2_9b","google/gemma-2-9b",True),
 ("llama32_3b","meta-llama/Llama-3.2-3B",True),
 ("olmo2_13b","allenai/OLMo-2-1124-13B",False),
]

def main():
    from huggingface_hub import HfApi, snapshot_download
    api=HfApi(endpoint=os.environ["HF_ENDPOINT"])
    rows=[]
    for key, mid, gated in MODELS:
        row={"key":key,"model_id":mid,"gated":gated,"resolved_revision":"","status":"","hidden_dim":"",
             "num_layers":"","num_params":"","local_path":"","timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            info=api.model_info(mid)
            row["resolved_revision"]=info.sha or ""
            cfg=None
            try:
                cfg=json.loads(api.hf_hub_download(mid,"config.json",cache_dir=CACHE)); 
            except Exception:
                pass
            if cfg:
                row["hidden_dim"]=cfg.get("hidden_size") or (cfg.get("text_config",{}) or {}).get("hidden_size","")
                row["num_layers"]=cfg.get("num_hidden_layers") or (cfg.get("text_config",{}) or {}).get("num_hidden_layers","")
            t=time.time()
            p=snapshot_download(mid, cache_dir=CACHE, revision=info.sha,
                                allow_patterns=["*.json","*.safetensors","*.model","tokenizer*","*.txt","*.py"],
                                max_workers=4)
            row["local_path"]=p; row["status"]="OK"; row["download_s"]=round(time.time()-t,1)
        except Exception as e:
            msg=f"{type(e).__name__}: {e}"
            row["status"]="BLOCKED_GATED_ACCESS" if gated else f"FAIL:{type(e).__name__}"
            row["error"]=msg[:300]
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        Path(OUT/"raw/model_registry_partial.json").write_text(json.dumps(rows,indent=2))
    with open(OUT/"raw/model_registry_resolved.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r})); w.writeheader()
        for r in rows: w.writerow(r)
    print("DONE ->",OUT/"raw/model_registry_resolved.csv")

if __name__=="__main__":
    main()
