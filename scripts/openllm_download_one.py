import os,sys,json,time
from pathlib import Path
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
for k in ["http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]: os.environ.pop(k,None)
os.environ["no_proxy"]="*"; os.environ["NO_PROXY"]="*"; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi, snapshot_download
mid=sys.argv[1]; key=sys.argv[2]; cache="/9950backfile/chenjiahui/hf_cache/hub"
api=HfApi(endpoint=os.environ["HF_ENDPOINT"])
info=api.model_info(mid); print("sha",info.sha,flush=True)
p=snapshot_download(mid,cache_dir=cache,revision=info.sha,max_workers=8)
print("OK",p,flush=True)
