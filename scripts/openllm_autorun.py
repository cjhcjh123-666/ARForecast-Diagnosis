"""Watchdog: as each model finishes downloading, launch its attribution suite
on the next free GPU (0-3). Runs smoke + pretrained pass + random pass.

Model -> (kind, identifier) where kind in {hf, ms}:
  hf : HF hub cache, path discovered via snapshot_download(local_files_only=True)
  ms : ModelScope dir /9950backfile/chenjiahui/model_cache_modelscope/<repo>
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
PY="/public/duyinglong/miniconda3/envs/wavellm/bin/python"
HFC="/9950backfile/chenjiahui/hf_cache/hub"
MSC="/9950backfile/chenjiahui/model_cache_modelscope"
GPUS=["cuda:0","cuda:1","cuda:2","cuda:3"]
LOGS=REPO/"logs"; LOGS.mkdir(exist_ok=True)

MODELS=[
 ("olmo2_7b","hf","allenai/OLMo-2-1124-7B"),
 ("mistral_7b_v03","hf","mistralai/Mistral-7B-v0.3"),
 ("deepseek_llm_7b","hf","deepseek-ai/deepseek-llm-7b-base"),
 ("deepseek_v2_lite","hf","deepseek-ai/DeepSeek-V2-Lite"),
 ("olmo2_13b","hf","allenai/OLMo-2-1124-13B"),
 ("gemma2_9b","ms","LLM-Research/gemma-2-9b"),
 ("gemma2_2b","ms","LLM-Research/gemma-2-2b"),
 ("llama32_3b","ms","LLM-Research/Llama-3.2-3B"),
 ("llama31_8b","ms","LLM-Research/Meta-Llama-3.1-8B"),
]

def hf_path(model_id):
    d=Path(HFC)/("models--"+model_id.replace("/","--"))
    snaps=sorted((d/"snapshots").glob("*")) if (d/"snapshots").is_dir() else []
    if not snaps: return None
    s=snaps[-1]
    idx=list(s.glob("*.index.json"))
    weights=list(s.glob("*.safetensors"))+list(s.glob("pytorch_model*.bin"))
    if not idx or not weights: return None
    if list(d.rglob("*.incomplete")): return None
    # expected shard count from index
    try:
        j=json.loads(idx[0].read_text()); need=len({v for v in j["weight_map"].values()})
    except Exception:
        need=0
    names={p.name for p in weights}
    if need and len([n for n in names if n.startswith("model-") or n.startswith("pytorch_model")]) < need and not (s/"model.safetensors").exists():
        return None
    return str(s)

def ms_path(repo):
    d=Path(MSC)/repo
    if not d.is_dir() or not (d/"config.json").is_file(): return None
    tmp=Path(MSC)/"._____temp"/repo
    if tmp.exists() and any(tmp.rglob("*")): return None
    if list(Path(MSC).rglob("*.incomplete")): return None
    return str(d)

def resolve(kind, ident):
    return hf_path(ident) if kind=="hf" else ms_path(ident)

def run(model_key, model_id, path, gpu, tag):
    log=LOGS/f"autorun_{model_key}_{tag}.log"
    cmd=[PY,"-u","scripts/openllm_suite.py","--model-key",model_key,"--model-id",model_id,
         "--path",path,"--device",gpu,"--seeds","7,17,27","--tasks","A,B,C,C2,R","--inits",tag]
    with open(log,"a") as f:
        f.write(f"\n=== {time.strftime('%F %T')} {model_key} {tag} {gpu} ===\n"); f.flush()
        return subprocess.Popen(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)

def main():
    done=set(); running={}  # gpu->(proc,model,tag,path)
    print("watchdog started",flush=True)
    while len(done)<len(MODELS)*2:
        # launch
        for key,kind,ident in MODELS:
            if (key,"pretrained") in done and (key,"random") in done: continue
            path=resolve(kind,ident)
            if not path: continue
            for tag in ["pretrained","random"]:
                if (key,tag) in done: continue
                if any(r[1]==key and r[2]==tag for r in running.values()): continue
                # do not run the other pass of the SAME model concurrently (shared outputs)
                if any(r[1]==key for r in running.values()): continue
                if len(running)>=len(GPUS): continue
                free=[g for g in GPUS if g not in running]
                if not free: continue
                gpu=free[0]
                try:
                    p=run(key,ident,path,gpu,tag)
                    running[gpu]=(p,key,tag,path)
                    print(f"LAUNCH {key} {tag} on {gpu}",flush=True)
                except Exception as e:
                    print("launch fail",key,tag,e,flush=True)
        # reap
        for gpu,(p,key,tag,path) in list(running.items()):
            if p.poll() is not None:
                rc=p.returncode
                print(f"DONE {key} {tag} rc={rc} {gpu}",flush=True)
                if rc==0: done.add((key,tag))
                else: done.add((key,tag))  # avoid hot loop; failures logged
                del running[gpu]
        time.sleep(60)
    print("ALL_MODELS_DONE",flush=True)

if __name__=="__main__": main()
