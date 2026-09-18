"""Frozen-representation extraction and checkpoint provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def checkpoint_identity(model_key: str, path: str | Path) -> dict:
    path = Path(path).resolve()
    config_path = path / "config.json"
    tokenizer_path = path / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint metadata: {path}")
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    architectures = [str(x) for x in config.get("architectures", [])]
    model_type = str(config.get("model_type", "unknown"))
    if model_key == "qwen3_8b_base" and (model_type != "qwen3" or "Qwen3ForCausalLM" not in architectures):
        raise ValueError(f"official Qwen3 base required, got {model_type}/{architectures} at {path}")
    if "qwen3ts" in str(path).lower() or model_type == "qwen3ts":
        raise ValueError(f"quarantined Qwen3TS path refused: {path}")
    weights = sorted(list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin")))
    if not weights:
        raise FileNotFoundError(f"no checkpoint weights at {path}")
    manifest = []
    for weight in weights:
        target = str(weight.readlink()) if weight.is_symlink() else ""
        manifest.append({"name": weight.name, "size": weight.stat().st_size, "symlink_target": target})
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "model_key": model_key, "model_path": str(path), "revision": path.name,
        "model_type": model_type, "architectures": architectures,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "weight_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "weight_fingerprint_kind": "ordered filename+byte-size+symlink-content-id manifest",
        "weight_files": len(weights), "weight_bytes": sum(x["size"] for x in manifest),
    }


@torch.inference_mode()
def embed_texts(model, tokenizer, texts: list[str], device: str, batch_size: int = 2) -> tuple[np.ndarray, dict]:
    tokenizer.padding_side = "right"
    outputs, token_counts = [], []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=False,
                            add_special_tokens=False).to(device)
        mask = encoded["attention_mask"].bool()
        token_counts.extend(mask.sum(1).cpu().tolist())
        hidden = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[-1]
        # Correct for either padding side: take the largest true mask position in each row.
        positions = torch.arange(mask.shape[1], device=mask.device)[None, :].expand_as(mask)
        last = positions.masked_fill(~mask, -1).max(1).values
        outputs.append(hidden[torch.arange(len(chunk), device=hidden.device), last].float().cpu().numpy())
    return np.concatenate(outputs), {"input_tokens_total": int(sum(token_counts)),
                                     "input_tokens_max": int(max(token_counts)),
                                     "n_sequences": len(texts)}


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
