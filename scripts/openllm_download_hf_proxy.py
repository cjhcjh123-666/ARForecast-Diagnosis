"""Download gated models from huggingface.co through the working local proxy.

Requires env: HF_TOKEN, http_proxy/https_proxy=http://127.0.0.1:7899.
Does NOT strip proxy vars (unlike the hf-mirror downloader).
"""
import os, sys, json, time, hashlib
from pathlib import Path
CACHE="/9950backfile/chenjiahui/hf_cache/hub"
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/"results/open_llm_suite/raw"; OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER",None)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main():
    key=sys.argv[1]; mid=sys.argv[2]
    from huggingface_hub import HfApi, snapshot_download
    api=HfApi()
    info=api.model_info(mid); sha=info.sha
    t=time.time()
    p=snapshot_download(mid, cache_dir=CACHE, revision=sha, max_workers=8)
    rec={"key":key,"source":"huggingface.co (via proxy 127.0.0.1:7899)","model_id":mid,
         "resolved_revision":sha,"local_path":p,"download_s":round(time.time()-t,1),
         "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT/f"hfproxy_{key}.json").write_text(json.dumps(rec,indent=2))
    print("OK",json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=="__main__": main()
