#!/usr/bin/env python3
"""Train paired question-conditioned heads on locked real TSQA representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

from tsqa_evidence.candidate_scorer import QuestionEvidenceScorer


ROOT = REPO / "results/tsqa_evidence/real_v1"
PROTOCOL_SHA = "9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    cache = np.load(path, allow_pickle=False)
    protocol = str(cache["protocol_sha256"].item())
    if protocol != PROTOCOL_SHA:
        raise ValueError(f"cache protocol mismatch: {path}")
    features = np.asarray(cache["features"], dtype=np.float32)
    mapping = {str(k): features[i] for i, k in enumerate(cache["keys"])}
    metadata = {k: cache[k].item() for k in cache.files if np.asarray(cache[k]).ndim == 0}
    return mapping, metadata


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_arrays(items: list[dict], qmap: dict[str, np.ndarray], xmap: dict[str, np.ndarray]):
    n = len(items)
    max_options = max(len(r["option_order"]) for r in items)
    qdim = len(next(iter(qmap.values())))
    xdim = len(next(iter(xmap.values())))
    q = np.zeros((n, max_options, qdim), dtype=np.float32)
    x = np.zeros((n, xdim), dtype=np.float32)
    mask = np.zeros((n, max_options), dtype=bool)
    y = np.zeros(n, dtype=np.int64)
    labels_by_item = []
    for i, row in enumerate(items):
        labels = sorted(row["option_order"])
        labels_by_item.append(labels)
        for j, label in enumerate(labels):
            key = f"{row['family_id']}|{row['cell']}|{row['question_id']}|{label}"
            q[i, j] = qmap[key]
            mask[i, j] = True
        y[i] = labels.index(row["gold_position"])
        x[i] = xmap[f"{row['family_id']}|{row['variant_id']}"]
    return x, q, mask, y, labels_by_item


def train_head(x: np.ndarray, q: np.ndarray, mask: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
               *, seed: int, device: str, epochs: int = 100, batch_size: int = 256):
    model = QuestionEvidenceScorer(x.shape[1], q.shape[2], projection_dim=128, hidden_dim=64, seed=seed).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    xt, qt = torch.from_numpy(x).to(device), torch.from_numpy(q).to(device)
    mt, yt = torch.from_numpy(mask).to(device), torch.from_numpy(y).to(device)
    permutations = [np.random.default_rng(seed + epoch).permutation(train_idx) for epoch in range(epochs)]
    model.train()
    for order in permutations:
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=device)
            logits = model(xt[idx], qt[idx]).masked_fill(~mt[idx], -1e9)
            loss = torch.nn.functional.cross_entropy(logits, yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model, initial_hash


@torch.inference_mode()
def predict(model, x: np.ndarray, q: np.ndarray, mask: np.ndarray, indices: np.ndarray, device: str,
            question_only: bool = False):
    model.eval()
    xb = np.zeros_like(x[indices]) if question_only else x[indices]
    began = time.perf_counter()
    logits = model(torch.from_numpy(xb).to(device), torch.from_numpy(q[indices]).to(device))
    logits = logits.masked_fill(~torch.from_numpy(mask[indices]).to(device), -1e9)
    elapsed = (time.perf_counter() - began) / len(indices)
    return logits.float().cpu().numpy(), elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--seed", type=int, choices=(7, 17, 27), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps-dir", type=Path, default=ROOT / "diagnostic/reps")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "diagnostic/predictions")
    args = ap.parse_args()
    lock = json.loads((ROOT / "protocol_lock.json").read_text())
    if lock["protocol_sha256"] != PROTOCOL_SHA:
        raise RuntimeError("protocol SHA mismatch")
    qpath = args.reps_dir / f"{args.model_key}_question_pretrained.npz"
    ppath = args.reps_dir / f"{args.model_key}_temporal_pretrained_s7.npz"
    rpath = args.reps_dir / f"{args.model_key}_temporal_random_s{args.seed}.npz"
    for path in (qpath, ppath, rpath):
        if not path.is_file():
            raise FileNotFoundError(path)
    qmap, qmeta = load_cache(qpath)
    pmap, pmeta = load_cache(ppath)
    rmap, rmeta = load_cache(rpath)
    items = read_jsonl(ROOT / "manifests/item_manifest.jsonl")
    split_idx = {split: np.asarray([i for i, row in enumerate(items) if row["split"] == split], dtype=int)
                 for split in ("train", "validation", "test")}
    branch_data = {}
    for branch, xmap in (("pretrained_temporal", pmap), ("random_temporal", rmap)):
        branch_data[branch] = make_arrays(items, qmap, xmap)
    rows = []
    initial_hashes = {}
    projection_hashes = {}
    for branch in ("pretrained_temporal", "random_temporal"):
        x, q, mask, y, labels = branch_data[branch]
        model, initial_hash = train_head(x, q, mask, y, split_idx["train"], seed=args.seed, device=args.device)
        initial_hashes[branch] = initial_hash
        projection_hashes[branch] = model.projection_hashes()
        for condition in ("full", "question_only"):
            logits, latency = predict(model, x, q, mask, split_idx["test"], args.device, condition == "question_only")
            for local_i, global_i in enumerate(split_idx["test"]):
                item = items[int(global_i)]
                predicted_index = int(np.argmax(logits[local_i]))
                position = labels[int(global_i)][predicted_index]
                semantic = item["option_order"][position]
                rows.append({
                    "benchmark_version": "real_tsqa_v1.0", "protocol_sha256": PROTOCOL_SHA,
                    "model": args.model_key, "checkpoint_revision": pmeta["checkpoint_revision"],
                    "interface": "question_conditioned_supervised_QA", "pretrained_or_random": branch,
                    "seed": args.seed, "condition": condition,
                    **{k: item[k] for k in ("split", "track", "cell", "domain", "dataset", "group_id", "family_id", "series_id", "variant_id", "question_id", "task_type", "question", "semantic_options", "option_order", "gold_semantic", "gold_position")},
                    "raw_response": json.dumps({label: float(logits[local_i, j]) for j, label in enumerate(labels[int(global_i)])}, sort_keys=True),
                    "parsed_semantic": semantic, "parsed_position": position, "valid_output": True,
                    "correct": position == item["gold_position"], "latency": latency, "error_type": "",
                    "question_cache_feature_sha256": qmeta["feature_sha256"],
                })
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if initial_hashes["pretrained_temporal"] != initial_hashes["random_temporal"]:
        raise AssertionError("P/R heads did not start from identical state")
    if projection_hashes["pretrained_temporal"] != projection_hashes["random_temporal"]:
        raise AssertionError("P/R fixed projections differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.model_key}_s{args.seed}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "model": args.model_key, "seed": args.seed, "rows": len(rows), "question_cache": str(qpath),
        "question_cache_feature_sha256": qmeta["feature_sha256"], "pretrained_temporal_cache": str(ppath),
        "random_temporal_cache": str(rpath), "paired_initial_state_sha256": initial_hashes["pretrained_temporal"],
        "projection_hashes": projection_hashes["pretrained_temporal"], "protocol_sha256": PROTOCOL_SHA,
    }
    (args.output_dir / f"{args.model_key}_s{args.seed}.metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "rows": len(rows), "correct": sum(r["correct"] for r in rows)}))


if __name__ == "__main__":
    main()
