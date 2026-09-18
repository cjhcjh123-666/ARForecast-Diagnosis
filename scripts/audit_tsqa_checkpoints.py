#!/usr/bin/env python3
"""Write the locked model/checkpoint inventory without loading model weights."""
from __future__ import annotations

import csv
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from tsqa_evidence.representations import checkpoint_identity

HF = Path("/9950backfile/chenjiahui/hf_cache/hub")
MS = Path("/9950backfile/chenjiahui/model_cache_modelscope/LLM-Research")
MODELS = [
    ("qwen3_8b_base", "Qwen/Qwen3-8B-Base", REPO / "models/qwen3-8b-base", "official-local-copy", True),
    ("llama31_8b", "LLM-Research/Meta-Llama-3.1-8B", MS / "Meta-Llama-3.1-8B", "24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea", True),
    ("llama32_3b", "LLM-Research/Llama-3.2-3B", MS / "Llama-3.2-3B", "48d84dd14886f01964f72e7f742a404dd86f9fb6", False),
    ("gemma2_9b", "LLM-Research/gemma-2-9b", MS / "gemma-2-9b", "dfd734b14754b9f17baa1ffb6bb7aa47f19b7810", True),
    ("gemma2_2b", "LLM-Research/gemma-2-2b", MS / "gemma-2-2b", "d445d76b68a9839195aebd927b7f882186bd78d4", False),
    ("mistral_7b_v03", "mistralai/Mistral-7B-v0.3", HF / "models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1", "caa1feb0e54d415e2df31207e5f4e273e33509b1", False),
    ("deepseek_llm_7b", "deepseek-ai/deepseek-llm-7b-base", HF / "models--deepseek-ai--deepseek-llm-7b-base/snapshots/7683fea62db869066ddaff6a41d032262c490d4f", "7683fea62db869066ddaff6a41d032262c490d4f", False),
    ("olmo2_7b", "allenai/OLMo-2-1124-7B", HF / "models--allenai--OLMo-2-1124-7B/snapshots/7df9a82518afdecae4e8c026b27adccc8c1f0032", "7df9a82518afdecae4e8c026b27adccc8c1f0032", True),
    ("olmo2_13b", "allenai/OLMo-2-1124-13B", HF / "models--allenai--OLMo-2-1124-13B/snapshots/3fefddc1bf18a30e1d9b91000271630718f2aa8b", "3fefddc1bf18a30e1d9b91000271630718f2aa8b", False),
    ("deepseek_v2_lite", "deepseek-ai/DeepSeek-V2-Lite", HF / "models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0", "604d5664dddd88a0433dbae533b7fe9472482de0", True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "results/tsqa_evidence/v1/validation/checkpoint_cache_audit.csv")
    args = parser.parse_args()
    rows = []
    for model_key, model_id, path, source_revision, native_core in MODELS:
        try:
            identity = checkpoint_identity(model_key, path)
            rows.append({
                "model_key": model_key, "model_id": model_id, "path": identity["model_path"],
                "source_revision": source_revision, "model_type": identity["model_type"],
                "architectures": ";".join(identity["architectures"]),
                "config_sha256": identity["config_sha256"],
                "tokenizer_config_sha256": identity["tokenizer_config_sha256"],
                "weight_manifest_sha256": identity["weight_manifest_sha256"],
                "weight_fingerprint_kind": identity["weight_fingerprint_kind"],
                "weight_files": identity["weight_files"], "weight_bytes": identity["weight_bytes"],
                "diagnostic_matrix": "YES", "native_core": "YES" if native_core else "NO",
                "status": "AVAILABLE",
            })
        except Exception as exc:
            rows.append({"model_key": model_key, "model_id": model_id, "path": str(path),
                         "source_revision": source_revision, "model_type": "", "architectures": "",
                         "config_sha256": "", "tokenizer_config_sha256": "",
                         "weight_manifest_sha256": "", "weight_fingerprint_kind": "",
                         "weight_files": 0, "weight_bytes": 0, "diagnostic_matrix": "YES",
                         "native_core": "YES" if native_core else "NO", "status": f"UNAVAILABLE:{exc}"})
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out} available={sum(r['status']=='AVAILABLE' for r in rows)}/{len(rows)}")


if __name__ == "__main__": main()
