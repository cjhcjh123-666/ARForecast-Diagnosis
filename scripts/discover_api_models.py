"""Discover available API models (OpenAI-compatible listing). Free; no generation.

Key is read from AIGCBEST_API_KEY (fallback: local api.txt, never committed).
Base URL from AIGCBEST_BASE_URL (default https://api2.aigcbest.top).
"""
from __future__ import annotations
import os, json, time, urllib.request
for _k in ["http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]: os.environ.pop(_k,None)
os.environ["no_proxy"]="*"; os.environ["NO_PROXY"]="*"
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/"results/open_llm_suite"; OUT.mkdir(parents=True,exist_ok=True)
CONF=REPO/"configs"; CONF.mkdir(parents=True,exist_ok=True)

def get_key():
    k=os.environ.get("AIGCBEST_API_KEY")
    if k: return k.strip()
    p=REPO/"api.txt"
    if p.is_file(): return p.read_text().strip()
    raise SystemExit("AIGCBEST_API_KEY unset and api.txt missing")

def main():
    base=os.environ.get("AIGCBEST_BASE_URL","https://api2.aigcbest.top").rstrip("/")
    key=get_key()
    url=f"{base}/v1/models"
    req=urllib.request.Request(url, headers={"Authorization":f"Bearer {key}","User-Agent":"openllm-suite"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data=json.loads(r.read())
    ids=sorted({m["id"] for m in data.get("data",[]) if "id" in m})
    ts=time.strftime("%Y%m%d_%H%M%S")
    (OUT/f"api_models_available_{ts}.json").write_text(json.dumps({"base_url":base,"n":len(ids),"ids":ids},indent=2))
    # family selection (exact ids, resolved from the listing)
    def pick(*pats):
        out=[]
        for pat in pats:
            hits=[i for i in ids if pat.lower() in i.lower()]
            if hits: out.append(hits[0])
        return out
    fams={
      "deepseek": pick("deepseek-chat","deepseek-reasoner","deepseek-v3"),
      "llama":    pick("llama-3.3-70b","llama-3.1-405b","llama-3.1-8b"),
      "qwen":     pick("qwen-max","qwen-plus","qwen2.5-72b"),
      "mistral":  pick("mistral-large","mistral-small","mixtral"),
      "gemma":    pick("gemma-2-27b","gemma-2-9b","gemma"),
      "other":    pick("gpt-4o","claude-3-5-sonnet","glm-4"),
    }
    sel=[]
    for f,ms in fams.items():
        for m in ms[:2]:
            if m and m not in sel: sel.append({"family":f,"model_id":m,"protocol_hint":"chat"})
    conf={"created":ts,"base_url":base,"selection_rule":"<=2 per family from the live listing; chat protocol",
          "models":sel[:15]}
    (CONF/"api_models_frozen.yaml").write_text(json.dumps(conf,indent=2))
    print("available:",len(ids))
    for f,ms in fams.items(): print(f"  {f}: {ms}")
    print("frozen:",[m["model_id"] for m in conf["models"]])
if __name__=="__main__": main()
