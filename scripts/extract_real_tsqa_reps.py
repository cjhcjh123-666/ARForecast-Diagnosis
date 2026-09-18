#!/usr/bin/env python3
"""Extract locked, isolated question-option and temporal caches for interface B."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

from scripts.transformers_runtime_compat import guard_optional_sklearn

guard_optional_sklearn()

from scripts.audit_tsqa_checkpoints import MODELS
from scripts.extract_qa_reps import validate_model_identity
from scripts.openllm_suite import load_lm
from tsqa_evidence.real_protocol import render_question_option, render_temporal


ROOT = REPO / "results/tsqa_evidence/real_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def registry() -> dict[str, dict]:
    return {key: {"model_id": model_id, "path": Path(path), "revision": revision}
            for key, model_id, path, revision, _ in MODELS}


def ensure_lock() -> str:
    lock = json.loads((ROOT / "protocol_lock.json").read_text())
    expected = "9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c"
    if lock["protocol_sha256"] != expected:
        raise RuntimeError("protocol SHA mismatch")
    return expected


@torch.inference_mode()
def embed_mean(model, tok, texts: list[str], device: str, batch_size: int) -> tuple[np.ndarray, list[int]]:
    output = []
    lengths = []
    context = int(getattr(model.config, "max_position_embeddings", 1 << 30))
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        encoded = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False, truncation=False)
        batch_lengths = encoded["attention_mask"].sum(1).tolist()
        if max(batch_lengths) > context:
            raise ValueError(f"input length {max(batch_lengths)} exceeds context {context}; truncation forbidden")
        encoded = encoded.to(device)
        hidden = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        output.append(pooled.float().cpu().numpy())
        lengths.extend(int(x) for x in batch_lengths)
    return np.concatenate(output), lengths


def tensor_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--branch", choices=("question", "temporal"), required=True)
    ap.add_argument("--init", choices=("pretrained", "random"), default="pretrained")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "diagnostic/reps")
    args = ap.parse_args()
    protocol_sha = ensure_lock()
    spec = registry().get(args.model_key)
    if spec is None:
        raise ValueError(args.model_key)
    if args.branch == "question" and args.init != "pretrained":
        raise ValueError("the locked question branch is official pretrained only")
    identity = validate_model_identity(args.model_key, spec["path"])
    suffix = "question_pretrained" if args.branch == "question" else f"temporal_{args.init}_s{args.seed}"
    out = args.output_dir / f"{args.model_key}_{suffix}.npz"
    if out.exists():
        print(json.dumps({"status": "SKIP_EXISTS", "output": str(out)}))
        return
    families = {f["family_id"]: f for f in read_jsonl(ROOT / "manifests/families_candidate.jsonl")}
    items = read_jsonl(ROOT / "manifests/item_manifest.jsonl")
    if args.branch == "question":
        keys, texts = [], []
        for item in items:
            for label in sorted(item["option_order"]):
                keys.append(f"{item['family_id']}|{item['cell']}|{item['question_id']}|{label}")
                texts.append(render_question_option(item, label))
        model_init, model_seed = "pretrained", 7
    else:
        keys, texts = [], []
        seen = set()
        for item in items:
            key = f"{item['family_id']}|{item['variant_id']}"
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
            texts.append(render_temporal(families[item["family_id"]], item["variant_id"], "full"))
        model_init, model_seed = args.init, args.seed
    model, tok = load_lm(str(spec["path"]), model_init, model_seed, args.device)
    try:
        features, lengths = embed_mean(model, tok, texts, args.device, args.batch_size)
    finally:
        del model
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, keys=np.asarray(keys), features=features, input_lengths=np.asarray(lengths),
        feature_sha256=np.asarray(tensor_hash(features)), protocol_sha256=np.asarray(protocol_sha),
        model_key=np.asarray(args.model_key), model_path=np.asarray(str(spec["path"])),
        checkpoint_revision=np.asarray(spec["revision"]), init=np.asarray(model_init), seed=np.asarray(model_seed),
        model_type=np.asarray(identity["model_type"]), config_sha256=np.asarray(identity["config_sha256"]),
        pooling=np.asarray("final_layer_mean_nonpadding"),
    )
    print(json.dumps({"status": "WROTE", "output": str(out), "rows": len(keys),
                      "shape": list(features.shape), "feature_sha256": tensor_hash(features)}))


if __name__ == "__main__":
    main()
