"""Per-model init audit: pretrained (released) vs random (from-config) backbones.

For each tensor (parameter or buffer) records name / shape / module class /
equal_to_pretrained / max_abs_diff, so reviewers can verify the "random"
branch carries no unexplained pretrained residual. Runs only model
construction (no window embedding).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

def audit_qwen(seed, device):
    import transformers.modeling_utils as _mu
    if not isinstance(_mu.ALL_PARALLEL_STYLES, (set, frozenset)):
        _mu.ALL_PARALLEL_STYLES = frozenset({"tp","block","sharded","pp","sequence","rowwise","colwise","naive","serial","manual","flex","ddp"})
    from transformers import AutoConfig, AutoModelForCausalLM
    mpath=str(REPO/"models/qwen3-8b-base")
    cfg=AutoConfig.from_pretrained(mpath, trust_remote_code=True)
    pre=AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.bfloat16)
    torch.manual_seed(seed); ran=AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(torch.bfloat16)
    return pre, ran, {"config": str(Path(mpath)/"config.json"), "revision":"local official Qwen3-8B-Base", "construct":"AutoModelForCausalLM.from_config(config)"}

def audit_chronos_t5(tag, seed):
    CH=str(REPO/"third_party/chronos-forecasting/src")
    if CH not in sys.path: sys.path.insert(0, CH)
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    from chronos import ChronosConfig, ChronosModel
    mpath=f"/public/chenjiahui/SAFER-TS/code/model_cache/amazon__chronos-{tag}"
    cfg=AutoConfig.from_pretrained(mpath)
    pre_inner=AutoModelForSeq2SeqLM.from_pretrained(mpath)
    torch.manual_seed(seed); ran_inner=AutoModelForSeq2SeqLM.from_config(cfg)
    cc=ChronosConfig(**cfg.chronos_config)
    pre=ChronosModel(config=cc, model=pre_inner); ran=ChronosModel(config=cc, model=ran_inner)
    return pre, ran, {"config": mpath+"/config.json", "construct":"AutoModelForSeq2SeqLM.from_config(config) wrapped in ChronosModel"}

def audit_chronos_bolt(tag, seed):
    CH=str(REPO/"third_party/chronos-forecasting/src")
    if CH not in sys.path: sys.path.insert(0, CH)
    from transformers import AutoConfig
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    mpath=f"/public/chenjiahui/SAFER-TS/code/model_cache/amazon__chronos-{tag}"
    cfg=AutoConfig.from_pretrained(mpath)
    pre=ChronosBoltModelForForecasting.from_pretrained(mpath)
    torch.manual_seed(seed); ran=ChronosBoltModelForForecasting(cfg)
    return pre, ran, {"config": mpath+"/config.json", "construct":"ChronosBoltModelForForecasting(config)"}

def audit_timesfm(seed):
    sys.path.insert(0,'/public/chenjiahui/SAFER-TS/code/baselines_external/timesfm/src')
    from scripts.tsfm_models import TimesFMRep
    ckpt="/public/chenjiahui/SAFER-TS/code/model_cache/google__timesfm-2.5-200m-pytorch/model.safetensors"
    pre=TimesFMRep(ckpt,"cpu",False,seed).m
    torch.manual_seed(seed); ran=TimesFMRep(ckpt,"cpu",True,seed).m
    return pre, ran, {"config":"TimesFM_2p5_200M_Definition","construct":"TimesFM_2p5_200M_torch_module() (no load_checkpoint)"}

def audit_moment(seed):
    sys.path.insert(0,'/public/chenjiahui/波数据时序基座大模型.bak_public/moment-main/Inference')
    import json as _json
    from momentfm.models.moment import MOMENT
    wdir='/public/chenjiahui/波数据时序基座大模型.bak_public/WaveFormer/models/external/moment'
    cfg=_json.load(open(f"{wdir}/config.json")); cfg['d_model']=cfg['t5_config']['d_model']
    pre=MOMENT(cfg, model_kwargs={})
    sd=torch.load(f"{wdir}/pytorch_model.bin", map_location="cpu", weights_only=False)
    pre.load_state_dict(sd, strict=False)
    torch.manual_seed(seed); ran=MOMENT(cfg, model_kwargs={})
    return pre, ran, {"config": wdir+"/config.json", "construct":"MOMENT(config) without load_state_dict"}

def audit_moirai(seed):
    import types
    # env shims
    try:
        import jax  # noqa
    except Exception:
        from typing import Any
        class _Fake:
            def __class_getitem__(cls, item): return Any
        class _Mod(types.ModuleType):
            def __getattr__(self, name):
                if name.startswith('__') and name.endswith('__'): raise AttributeError(name)
                return _Fake
        _m=_Mod('jaxtyping'); _m.__file__='<shim>'
        sys.modules['jaxtyping']=_m
    import torch as _t
    try:
        import einops
        if tuple(int(x) for x in einops.__version__.split('.')[:2]) >= (0,7):
            _t.__version__="2.8.0"
    except Exception: pass
    UNI2TS="/public/chenjiahui/SAFER-TS/code/baselines_external/uni2ts/src"
    if UNI2TS not in sys.path: sys.path.insert(0, UNI2TS)
    import os
    base=os.path.join(UNI2TS,"uni2ts")
    def reg(name,path):
        m=types.ModuleType(name); m.__path__=[path]; m.__package__=name; sys.modules[name]=m
    if "uni2ts" not in sys.modules or not hasattr(sys.modules["uni2ts"],"__path__"):
        reg("uni2ts",base); reg("uni2ts.model",os.path.join(base,"model")); reg("uni2ts.model.moirai",os.path.join(base,"model","moirai"))
        for s in ["finetune","forecast","pretrain"]: sys.modules[f"uni2ts.model.moirai.{s}"]=types.ModuleType(f"uni2ts.model.moirai.{s}")
    from uni2ts.model.moirai.module import MoiraiModule, decode_distr_output
    import json as _json
    ckpt_dir="/public/chenjiahui/SAFER-TS/code/model_cache/Salesforce__moirai-1.1-R-small"
    cfgd=_json.load(open(f"{ckpt_dir}/config.json"))
    pre=MoiraiModule.from_pretrained(ckpt_dir)
    torch.manual_seed(seed)
    distr=decode_distr_output(cfgd["distr_output"])
    ran=MoiraiModule(distr_output=distr,d_model=cfgd["d_model"],num_layers=cfgd["num_layers"],
                     patch_sizes=cfgd["patch_sizes"],max_seq_len=cfgd["max_seq_len"],
                     attn_dropout_p=cfgd["attn_dropout_p"],dropout_p=cfgd["dropout_p"],scaling=cfgd["scaling"])
    return pre, ran, {"config": ckpt_dir+"/config.json", "construct":"MoiraiModule(**config) without from_pretrained"}

def compare(pre, ran, meta, model):
    sd_pre={k:v.detach().float().cpu() for k,v in pre.state_dict().items()}
    sd_ran={k:v.detach().float().cpu() for k,v in ran.state_dict().items()}
    mod_class={}
    for mn,m in pre.named_modules():
        for pn,p in m.named_parameters(prefix=mn, recurse=False):
            mod_class[pn]=type(m).__name__
        for bn,b in m.named_buffers(prefix=mn, recurse=False):
            mod_class[bn]=type(m).__name__
    keys=sorted(set(sd_pre)|set(sd_ran))
    rows=[]
    for k in keys:
        if k not in sd_pre or k not in sd_ran:
            rows.append({"name":k,"present":"pre_only" if k in sd_pre else "rand_only"}); continue
        a,b=sd_pre[k],sd_ran[k]
        if a.shape!=b.shape:
            rows.append({"name":k,"shape":list(a.shape),"shape_mismatch":True}); continue
        eq=bool(torch.allclose(a,b,atol=1e-6,rtol=1e-5))
        mad=float((a-b).abs().max().item()) if a.numel() else 0.0
        is_param=k in dict(ran.named_parameters())
        rows.append({"name":k,"shape":list(a.shape),"module":mod_class.get(k,""),"kind":"param" if is_param else "buffer",
                     "equal_to_pretrained":eq,"max_abs_diff":round(mad,6)})
    neq=[r for r in rows if r.get("equal_to_pretrained")==True]
    return {"model":model,"meta":meta,"n_tensors":len(rows),
            "n_equal_to_pretrained":len(neq),
            "equal_tensors":[(r["name"],r["module"],r["kind"]) for r in neq],
            "tensors":rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3_8b_base,chronos_t5-small,chronos_t5-base,chronos_bolt-small,timesfm_2.5-200m,moment-1-large,moirai-1.1-R-small")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results/iclr/tsfm_deliver_v2/init_audit"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    import importlib, json
    for m in a.models.split(","):
        if m=="qwen3_8b_base": res=audit_qwen(a.seed,"cpu")
        elif m.startswith("chronos_t5"): res=audit_chronos_t5(m.replace("chronos_",""),a.seed)
        elif m.startswith("chronos_bolt"): res=audit_chronos_bolt(m.replace("chronos_",""),a.seed)
        elif m.startswith("timesfm"): res=audit_timesfm(a.seed)
        elif m.startswith("moment"): res=audit_moment(a.seed)
        elif m.startswith("moirai"): res=audit_moirai(a.seed)
        else: continue
        report=compare(*res, m)
        (a.out/f"{m}.json").write_text(json.dumps(report,indent=2))
        print(f"[{m}] tensors={report['n_tensors']} equal_to_pretrained={report['n_equal_to_pretrained']}")
        for nm,md,kind in report["equal_tensors"][:12]:
            print("   equal:",nm,"|",md,"|",kind)
    # aggregate (skip the summary file itself)
    agg={}
    for jf in sorted(a.out.glob("*.json")):
        if jf.name=="init_audit_summary.json": continue
        d=json.loads(jf.read_text())
        if "n_tensors" not in d: continue
        agg[d["model"]]={"n_tensors":d["n_tensors"],"n_equal_to_pretrained":d["n_equal_to_pretrained"],"equal_tensors":d["equal_tensors"],"meta":d.get("meta",{})}
    (a.out/"init_audit_summary.json").write_text(json.dumps(agg,indent=2))
    print("summary ->", a.out/"init_audit_summary.json")

if __name__=="__main__":
    main()
