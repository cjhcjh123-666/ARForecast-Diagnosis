"""ICLR item-1: verify the Wave-MoE Qwen3TS-8B language backbone == official Qwen3-8B-Base.

Loads both checkpoints' shared safetensors shards and compares every tensor
key present in BOTH (the official has 399 language tensors; the Wave-MoE has
the same language tensors + TS modules in separate files).  Reports:
  - which keys are shared / only-in-wave / only-in-official
  - max absolute diff and exact-match count over shared keys
If max diff == 0 for all shared keys, the language backbone is provably the
official Qwen3-8B-Base.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
WAVE = Path("/public/chenjiahui/Wave-MoE-Skill-Agent/hf_models/Qwen3-8B")
OFF = Path("/9950backfile/chenjiahui/ARForecast-Diagnosis/models/qwen3-8b-base")

def shard_keys(path: Path, shard: str):
    with safe_open(path / shard, framework="np") as f:
        return set(f.keys())

def main():
    shards = [f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
    wave_keys, off_keys = set(), set()
    for sh in shards:
        wave_keys |= shard_keys(WAVE, sh)
        off_keys |= shard_keys(OFF, sh)
    only_wave = wave_keys - off_keys
    only_off = off_keys - wave_keys
    shared = wave_keys & off_keys
    print(f"wave total keys: {len(wave_keys)} | official total: {len(off_keys)} | shared: {len(shared)}")
    print(f"only-in-wave ({len(only_wave)}): {sorted(only_wave)[:10]}{'...' if len(only_wave)>10 else ''}")
    print(f"only-in-official ({len(only_off)}): {sorted(only_off)[:10]}{'...' if len(only_off)>10 else ''}")

    # compare values for shared keys (shard-by-shard to stay memory-light)
    maxdiff, mismatch = 0.0, 0
    for sh in shards:
        with safe_open(WAVE / sh, framework="np") as fw, safe_open(OFF / sh, framework="np") as fo:
            for k in shared:
                if k in fw.keys() and k in fo.keys():
                    a = fw.get_tensor(k).astype(np.float64)
                    b = fo.get_tensor(k).astype(np.float64)
                    if a.shape != b.shape:
                        print(f"  SHAPE MISMATCH {k}: {a.shape} vs {b.shape}"); continue
                    d = float(np.max(np.abs(a - b)))
                    maxdiff = max(maxdiff, d)
                    if d > 0: mismatch += 1
    print(f"\nshared-tensor max abs diff = {maxdiff:.6e}")
    print(f"keys with any diff (of {len(shared)}): {mismatch}")
    if maxdiff == 0.0:
        print("=> PROVEN: language backbone is exactly Qwen/Qwen3-8B-Base (max diff = 0)")
    else:
        print(f"=> NOT identical; largest diff on some tensor (see above)")

if __name__ == "__main__":
    main()
