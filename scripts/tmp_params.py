import json, os, glob
from pathlib import Path

def safetensors_count(path):
    import struct
    def header_bytes(path):
        with open(path,'rb') as f:
            n=struct.unpack('<Q', f.read(8))[0]
            return json.loads(f.read(n))
    total=0
    if os.path.isdir(path):
        files=sorted(glob.glob(os.path.join(path,'*.safetensors')))
    else:
        files=[path]
    for fp in files:
        h=header_bytes(fp)
        for k,v in h.items():
            if k=='__metadata__': continue
            d=v.get('dtype'); sh=v.get('shape',[])
            import numpy as np
            total+=int(np.prod(sh))
    return total

paths = {
 "qwen3_8b_base": "models/qwen3-8b-base",
 "chronos_t5-small": "/public/chenjiahui/SAFER-TS/code/model_cache/amazon__chronos-t5-small",
 "chronos_t5-base": "/public/chenjiahui/SAFER-TS/code/model_cache/amazon__chronos-t5-base",
 "chronos_bolt-small": "/public/chenjiahui/SAFER-TS/code/model_cache/amazon__chronos-bolt-small",
 "timesfm_2.5-200m": "/public/chenjiahui/SAFER-TS/code/model_cache/google__timesfm-2.5-200m-pytorch",
 "moment-1-large": "/public/chenjiahui/波数据时序基座大模型.bak_public/WaveFormer/models/external/moment",
 "moirai-1.1-R-small": "/public/chenjiahui/SAFER-TS/code/model_cache/Salesforce__moirai-1.1-R-small",
}
for m,p in paths.items():
    try:
        n=safetensors_count(p)
        print(f"{m:<20} {n/1e6:>10.2f} M params")
    except Exception as e:
        print(m, "ERR", e)
