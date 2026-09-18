#!/usr/bin/env python3
"""Run locked Interface A task-specific linear readouts on temporal caches."""

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

from scripts.run_real_tsqa_supervised import load_cache, read_jsonl, state_hash  # noqa: E402


ROOT = REPO / "results/tsqa_evidence/real_v1"
PROTOCOL_SHA = "9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c"
ADDENDUM_SHA = "63af96cbd7bf167882f8fe9b39ea22a1abcc2e6e35c85f832ba7ef8b5cad6a57"


def make_temporal(items: list[dict], xmap: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([xmap[f"{row['family_id']}|{row['variant_id']}"] for row in items]).astype(np.float32)


def train_task(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, classes: list[str],
               seed: int, device: str, epochs: int = 100, batch_size: int = 256):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Linear(x.shape[1], len(classes), bias=True).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    xt = torch.nn.functional.layer_norm(xt, (xt.shape[-1],))
    model.train()
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(train_idx)
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=device)
            loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model, xt, initial_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--seed", type=int, choices=(7, 17, 27), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reps-dir", type=Path, default=ROOT / "diagnostic/reps")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagnostic/readout")
    args = parser.parse_args()
    lock = json.loads((ROOT / "protocol_lock.json").read_text())
    if lock["protocol_sha256"] != PROTOCOL_SHA:
        raise RuntimeError("parent protocol SHA mismatch")
    addendum = (ROOT / "interface_a_addendum_sha256.txt").read_text().strip().split()[0]
    if addendum != ADDENDUM_SHA:
        raise RuntimeError("Interface A addendum SHA mismatch")

    ppath = args.reps_dir / f"{args.model_key}_temporal_pretrained_s7.npz"
    rpath = args.reps_dir / f"{args.model_key}_temporal_random_s{args.seed}.npz"
    for path in (ppath, rpath):
        if not path.is_file():
            raise FileNotFoundError(path)
    pmap, pmeta = load_cache(ppath)
    rmap, rmeta = load_cache(rpath)
    items = read_jsonl(ROOT / "manifests/item_manifest.jsonl")
    branch_x = {"pretrained_temporal": make_temporal(items, pmap),
                "random_temporal": make_temporal(items, rmap)}
    split = {name: np.asarray([i for i, row in enumerate(items) if row["split"] == name], dtype=int)
             for name in ("train", "test")}

    rows = []
    hashes: dict[str, dict[str, str]] = {branch: {} for branch in branch_x}
    tasks = sorted({row["task_type"] for row in items})
    for branch, x in branch_x.items():
        for task in tasks:
            task_idx = np.asarray([i for i, row in enumerate(items) if row["task_type"] == task], dtype=int)
            train_idx = np.intersect1d(task_idx, split["train"], assume_unique=True)
            test_idx = np.intersect1d(task_idx, split["test"], assume_unique=True)
            classes = sorted({items[int(i)]["gold_semantic"] for i in train_idx})
            unseen = sorted({items[int(i)]["gold_semantic"] for i in test_idx} - set(classes))
            if unseen:
                raise ValueError(f"unseen test labels for {task}: {unseen}")
            class_to_id = {label: i for i, label in enumerate(classes)}
            y = np.asarray([class_to_id.get(row["gold_semantic"], -1) for row in items], dtype=np.int64)
            model, xt, initial_hash = train_task(x, y, train_idx, classes=classes,
                                                  seed=args.seed, device=args.device)
            hashes[branch][task] = initial_hash
            model.eval()
            began = time.perf_counter()
            with torch.inference_mode():
                logits = model(xt[torch.as_tensor(test_idx, device=args.device)]).float().cpu().numpy()
            latency = (time.perf_counter() - began) / len(test_idx)
            for local_i, global_i in enumerate(test_idx):
                item = items[int(global_i)]
                pred_id = int(np.argmax(logits[local_i]))
                semantic = classes[pred_id]
                position = next((label for label, value in item["option_order"].items()
                                 if value == semantic), "")
                rows.append({
                    "benchmark_version": "real_tsqa_v1.0", "protocol_sha256": PROTOCOL_SHA,
                    "interface_a_addendum_sha256": ADDENDUM_SHA, "model": args.model_key,
                    "checkpoint_revision": pmeta["checkpoint_revision"],
                    "interface": "controlled_representation_readout",
                    "pretrained_or_random": branch, "seed": args.seed, "condition": "full",
                    **{k: item[k] for k in ("split", "track", "cell", "domain", "dataset", "group_id", "family_id", "series_id", "variant_id", "question_id", "task_type", "question", "semantic_options", "option_order", "gold_semantic", "gold_position")},
                    "raw_response": json.dumps({label: float(logits[local_i, j]) for j, label in enumerate(classes)}, sort_keys=True),
                    "parsed_semantic": semantic, "parsed_position": position, "valid_output": True,
                    "correct": semantic == item["gold_semantic"], "latency": latency, "error_type": "",
                    "temporal_cache_feature_sha256": (pmeta if branch == "pretrained_temporal" else rmeta)["feature_sha256"],
                })
            del model, xt
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if hashes["pretrained_temporal"] != hashes["random_temporal"]:
        raise AssertionError("P/R readout heads did not share initial states")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.model_key}_s{args.seed}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "model": args.model_key, "seed": args.seed, "rows": len(rows),
        "protocol_sha256": PROTOCOL_SHA, "interface_a_addendum_sha256": ADDENDUM_SHA,
        "pretrained_temporal_cache": str(ppath), "random_temporal_cache": str(rpath),
        "paired_initial_state_sha256_by_task": hashes["pretrained_temporal"],
    }
    (args.output_dir / f"{args.model_key}_s{args.seed}.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(out), "rows": len(rows),
                      "correct": sum(row["correct"] for row in rows)}))


if __name__ == "__main__":
    main()
