"""Check download completeness by comparing local files against the HF/hf-mirror file list."""
import json, os, sys, urllib.request
from pathlib import Path
for k in ["http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]: os.environ.pop(k,None)
os.environ["no_proxy"]="*"
CHECK=[
 ("hf","deepseek-ai/deepseek-llm-7b-base","/9950backfile/chenjiahui/hf_cache/hub/models--deepseek-ai--deepseek-llm-7b-base"),
 ("hf","allenai/OLMo-2-1124-7B","/9950backfile/chenjiahui/hf_cache/hub/models--allenai--OLMo-2-1124-7B"),
 ("hf","mistralai/Mistral-7B-v0.3","/9950backfile/chenjiahui/hf_cache/hub/models--mistralai--Mistral-7B-v0.3"),
 ("hf","allenai/OLMo-2-1124-13B","/9950backfile/chenjiahui/hf_cache/hub/models--allenai--OLMo-2-1124-13B"),
 ("hf","deepseek-ai/DeepSeek-V2-Lite","/9950backfile/chenjiahui/hf_cache/hub/models--deepseek-ai--DeepSeek-V2-Lite"),
 ("plain","allenai/OLMo-2-1124-7B","/9950backfile/chenjiahui/hf_plain/olmo7b"),
 ("plain","mistralai/Mistral-7B-v0.3","/9950backfile/chenjiahui/hf_plain/mistral7b"),
 ("plain","deepseek-ai/deepseek-llm-7b-base","/9950backfile/chenjiahui/hf_plain/ds7b"),
 ("plain","allenai/OLMo-2-1124-13B","/9950backfile/chenjiahui/hf_plain/olmo13b"),
 ("plain","deepseek-ai/DeepSeek-V2-Lite","/9950backfile/chenjiahui/hf_plain/dsv2lite"),
]
def api(repo):
    with urllib.request.urlopen(f"https://hf-mirror.com/api/models/{repo}",timeout=40) as r:
        d=json.loads(r.read())
    return {s["rfilename"]:s.get("size",0) for s in d["siblings"]}, d["sha"]
for kind,repo,path in CHECK:
    p=Path(path)
    if kind=="hf":
        snaps=sorted((p/"snapshots").glob("*")) if (p/"snapshots").is_dir() else []
        root=snaps[-1] if snaps else None
    else:
        root=p
    if root is None or not root.is_dir(): print(f"{repo:<34} [{kind}] MISSING DIR"); continue
    try: files,sha=api(repo)
    except Exception as e: print(f"{repo:<34} [{kind}] api err {e}"); continue
    need=[f for f in files if f.endswith((".safetensors",".bin","index.json"))]
    miss=[]; bad=[]; tot=needtot=0
    for f in need:
        fp=root/f
        if not fp.is_file(): miss.append(f); continue
        got=fp.stat().st_size; tot+=got; needtot+=files[f]
        if files[f] and got!=files[f]: bad.append((f,got,files[f]))
    print(f"{repo:<34} [{kind}] files {len(need)-len(miss)}/{len(need)} missing={len(miss)} size_mismatch={len(bad)} "
          f"({tot/1e9:.1f}/{needtot/1e9:.1f} GB) {'OK' if not miss and not bad else 'INCOMPLETE'}")
