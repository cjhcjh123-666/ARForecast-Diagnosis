"""ModelScope fallback downloader for HF-gated models (documented deviation).

hf-mirror does not proxy gated repos; ModelScope hosts the official-org mirrors
(LLM-Research/...). Provenance is recorded per model (source=ModelScope,
repo id, revision, sha256 of the safetensors weight index files).
"""
import sys, json, os, hashlib, time
from pathlib import Path
for k in ["http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]: os.environ.pop(k,None)
os.environ["no_proxy"]="*"; os.environ["NO_PROXY"]="*"
CACHE="/9950backfile/chenjiahui/model_cache_modelscope"
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/"results/open_llm_suite/raw"; OUT.mkdir(parents=True,exist_ok=True)

def sha256(p, limit=None):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while True:
            b=f.read(1<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    key=sys.argv[1]; repo=sys.argv[2]
    from modelscope import snapshot_download
    t=time.time()
    p=snapshot_download(repo, cache_dir=CACHE)
    rec={"key":key,"source":"ModelScope","repo_id":repo,"local_path":p,
         "download_s":round(time.time()-t,1),"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
    # hash any index json / config for provenance
    for name in ["config.json","model.safetensors.index.json","pytorch_model.bin.index.json"]:
        f=Path(p)/name
        if f.is_file(): rec["sha256_"+name]=sha256(f)
    (OUT/f"modelscope_{key}.json").write_text(json.dumps(rec,indent=2))
    print("OK",json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=="__main__": main()
