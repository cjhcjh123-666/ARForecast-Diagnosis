#!/usr/bin/env python3
"""Run locked native zero-shot QA on the real TSQA v1 test split."""

from __future__ import annotations

import argparse
import csv
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

from scripts.transformers_runtime_compat import guard_legacy_dynamic_cache_api, guard_optional_sklearn

guard_optional_sklearn()
guard_legacy_dynamic_cache_api()

from scripts.audit_tsqa_checkpoints import MODELS
from scripts.extract_qa_reps import validate_model_identity
from scripts.openllm_suite import load_lm
from tsqa_evidence.real_protocol import parse_strict_choice, render_native


ROOT = REPO / "results/tsqa_evidence/real_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def registry() -> dict[str, dict]:
    return {key: {"model_id": model_id, "path": Path(path), "revision": revision, "native_core": native}
            for key, model_id, path, revision, native in MODELS}


def ensure_lock() -> dict:
    lock = json.loads((ROOT / "protocol_lock.json").read_text())
    if lock["protocol_sha256"] != "9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c":
        raise RuntimeError("protocol SHA mismatch; do not run target models")
    return lock


def allowed_positions(item: dict) -> tuple[str, ...]:
    return tuple(sorted(item["option_order"]))


def preflight_lengths(tok, prompts: list[str], model) -> tuple[list[int], int]:
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    context = int(getattr(model.config, "max_position_embeddings", 1 << 30))
    too_long = [n for n in lengths if n + 8 > context]
    if too_long:
        raise ValueError(f"{len(too_long)} prompts exceed context={context}; truncation is forbidden")
    return lengths, context


@torch.inference_mode()
def free_generate(model, tok, prompts: list[str], device: str, batch_size: int) -> list[tuple[str, float]]:
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False, truncation=False).to(device)
        began = time.perf_counter()
        generated = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                   pad_token_id=tok.pad_token_id, use_cache=True)
        elapsed = (time.perf_counter() - began) / len(chunk)
        new = generated[:, enc["input_ids"].shape[1]:]
        out.extend((tok.decode(row, skip_special_tokens=True), elapsed) for row in new)
    return out


@torch.inference_mode()
def score_choices(model, tok, prompts: list[str], choices: list[tuple[str, ...]], device: str,
                  batch_size: int) -> list[tuple[str, dict[str, float], float]]:
    expanded = []
    for item_index, (prompt, labels) in enumerate(zip(prompts, choices)):
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        for label in labels:
            continuation = tok(" " + label, add_special_tokens=False)["input_ids"]
            expanded.append((item_index, label, prompt_ids, continuation))
    scores: dict[int, dict[str, float]] = {i: {} for i in range(len(prompts))}
    latency = np.zeros(len(prompts), dtype=float)
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start:start + batch_size]
        sequences = [p + c for _, _, p, c in batch]
        max_len = max(map(len, sequences))
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids = torch.full((len(batch), max_len), pad, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        starts = []
        for i, (_, _, prompt_ids, continuation) in enumerate(batch):
            seq = prompt_ids + continuation
            input_ids[i, -len(seq):] = torch.tensor(seq, device=device)
            attention[i, -len(seq):] = 1
            starts.append(max_len - len(continuation))
        began = time.perf_counter()
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
        elapsed = time.perf_counter() - began
        logp = torch.log_softmax(logits, dim=-1)
        for i, (item_index, label, _, continuation) in enumerate(batch):
            first = starts[i]
            value = 0.0
            for j, token in enumerate(continuation):
                value += float(logp[i, first + j - 1, token].item())
            scores[item_index][label] = value
            latency[item_index] += elapsed / len(batch)
    return [(max(score, key=score.get), score, float(latency[i])) for i, score in scores.items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", choices=("free", "constrained"), default="free")
    ap.add_argument("--conditions", default="full,question_only,shuffled")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "native")
    args = ap.parse_args()
    lock = ensure_lock()
    spec = registry().get(args.model_key)
    if spec is None or not spec["native_core"]:
        raise ValueError(f"{args.model_key} is not in the locked native-core matrix")
    identity = validate_model_identity(args.model_key, spec["path"])
    families = {f["family_id"]: f for f in read_jsonl(ROOT / "manifests/families_candidate.jsonl")}
    items = [r for r in read_jsonl(ROOT / "manifests/item_manifest.jsonl") if r["split"] == "test"]
    conditions = [x for x in args.conditions.split(",") if x]
    jobs = [(condition, item, render_native(families[item["family_id"]], item, condition))
            for condition in conditions for item in items]
    model, tok = load_lm(str(spec["path"]), "pretrained", 20260918, args.device)
    try:
        lengths, context = preflight_lengths(tok, [x[2] for x in jobs], model)
        if args.mode == "free":
            outputs = free_generate(model, tok, [x[2] for x in jobs], args.device, args.batch_size)
        else:
            scored = score_choices(model, tok, [x[2] for x in jobs], [allowed_positions(x[1]) for x in jobs], args.device, args.batch_size)
            outputs = [(json.dumps(scores, sort_keys=True), latency) for _, scores, latency in scored]
        rows = []
        for (condition, item, prompt), output, length in zip(jobs, outputs, lengths):
            raw, latency = output
            allowed = allowed_positions(item)
            parsed_position = (parse_strict_choice(raw, allowed) if args.mode == "free"
                               else max(json.loads(raw), key=json.loads(raw).get))
            parsed_semantic = item["option_order"].get(parsed_position, "") if parsed_position else ""
            rows.append({
                "benchmark_version": "real_tsqa_v1.0", "protocol_sha256": lock["protocol_sha256"],
                "model": args.model_key, "checkpoint_revision": spec["revision"], "interface": f"native_{args.mode}",
                "pretrained_or_random": "pretrained", "seed": 20260918, "condition": condition,
                **{k: item[k] for k in ("split", "track", "cell", "domain", "dataset", "group_id", "family_id", "series_id", "variant_id", "question_id", "task_type", "question", "semantic_options", "option_order", "gold_semantic", "gold_position")},
                "raw_prompt": prompt, "raw_response": raw, "parsed_semantic": parsed_semantic,
                "parsed_position": parsed_position or "", "valid_output": bool(parsed_position),
                "correct": bool(parsed_position == item["gold_position"]), "latency": latency,
                "input_tokens": length, "context_limit": context,
                "error_type": "" if parsed_position else "UNPARSED_OR_OUT_OF_RANGE",
                "model_type": identity["model_type"], "config_sha256": identity["config_sha256"],
            })
    finally:
        del model
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.model_key}_{args.mode}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "rows": len(rows), "valid": sum(r["valid_output"] for r in rows),
                      "correct": sum(r["correct"] for r in rows)}))


if __name__ == "__main__":
    main()
