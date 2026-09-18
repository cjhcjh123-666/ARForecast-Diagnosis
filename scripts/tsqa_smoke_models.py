#!/usr/bin/env python3
"""Run checkpoint/prompt/cache/batch smoke checks on independent development families."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from scripts.openllm_suite import load_lm
from tsqa_evidence.benchmark import (TASK_SPECS, gold, make_base, make_relevant,
                                     permute_options, semantic_options)
from tsqa_evidence.candidate_scorer import QuestionEvidenceScorer, tensor_sha256
from tsqa_evidence.rendering import render_question_option, render_temporal
from tsqa_evidence.representations import array_hash, checkpoint_identity, embed_texts


def dev_prompts() -> tuple[list[str], list[str], list[dict]]:
    temporal, question_options, rows = [], [], []
    tasks = ("A1_interval_trend", "B1_interval_mean", "D2_channel_lead_lag", "F2_event_order")
    for family_index in range(2):
        family_id = f"SMOKE_DEV_{family_index:02d}"
        base, _ = make_base(family_id, "validation", 99173)
        for task_id in tasks:
            relevant = make_relevant(base, task_id)
            a, b = gold(task_id, base), gold(task_id, relevant)
            options = permute_options(semantic_options(task_id, a, b), family_id, task_id)
            row = {"question_text": TASK_SPECS[task_id].question,
                   "options_json": json.dumps(options, sort_keys=True)}
            temporal.append(render_temporal(base)); rows.append(row)
            question_options.extend(render_question_option(row, label) for label in "ABCD")
    return temporal, question_options, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True); parser.add_argument("--model-id", required=True)
    parser.add_argument("--path", required=True); parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(); args.out.parent.mkdir(parents=True, exist_ok=True)
    temporal_texts, question_texts, _ = dev_prompts()
    identity = checkpoint_identity(args.model_key, args.path)
    t0 = time.time(); payload = {}; report = {"identity": identity, "dev_families": 2,
                                               "target_test_scores_computed": False, "seed": args.seed}
    pretrained, tok = load_lm(args.path, "pretrained", args.seed, args.device)
    try:
        token_ids_p = tok(temporal_texts, add_special_tokens=False)["input_ids"]
        hp, p_stats = embed_texts(pretrained, tok, temporal_texts, args.device, batch_size=2)
        hq_flat, q_stats = embed_texts(pretrained, tok, question_texts, args.device, batch_size=4)
        hq = hq_flat.reshape(len(temporal_texts), 4, -1)
        singleton, _ = embed_texts(pretrained, tok, [temporal_texts[0]], args.device, batch_size=1)
        batch_error = float(np.max(np.abs(singleton[0] - hp[0])))
        payload.update(h_temporal_pretrained=hp.astype(np.float16), h_question_option=hq.astype(np.float16))
        report.update(pretrained_stats=p_stats, question_stats=q_stats, mixed_vs_singleton_max_abs=batch_error,
                      pretrained_temporal_sha256=array_hash(payload["h_temporal_pretrained"]),
                      question_feature_sha256=tensor_sha256(payload["h_question_option"]))
    finally:
        del pretrained; torch.cuda.empty_cache()
    random_model, tok_r = load_lm(args.path, "random", args.seed, args.device)
    try:
        token_ids_r = tok_r(temporal_texts, add_special_tokens=False)["input_ids"]
        if token_ids_p != token_ids_r: raise AssertionError("P/R tokenizer serialization mismatch")
        hr, r_stats = embed_texts(random_model, tok_r, temporal_texts, args.device, batch_size=2)
        payload["h_temporal_random"] = hr.astype(np.float16)
        report.update(random_stats=r_stats, random_temporal_sha256=array_hash(payload["h_temporal_random"]),
                      tokenizer_ids_identical_pretrained_random=True)
    finally:
        del random_model; torch.cuda.empty_cache()
    scorer_p = QuestionEvidenceScorer(hp.shape[1], hq.shape[2], projection_dim=32, hidden_dim=16, seed=args.seed)
    scorer_r = QuestionEvidenceScorer(hr.shape[1], hq.shape[2], projection_dim=32, hidden_dim=16, seed=args.seed)
    report["fixed_projection_hashes_pretrained"] = scorer_p.projection_hashes()
    report["fixed_projection_hashes_random"] = scorer_r.projection_hashes()
    report["fixed_projection_hashes_match"] = scorer_p.projection_hashes() == scorer_r.projection_hashes()
    with torch.no_grad():
        s0 = scorer_p(torch.from_numpy(hp[:2]), torch.from_numpy(hq[:2]))
        s1 = scorer_p(torch.from_numpy(hp[2:4]), torch.from_numpy(hq[2:4]))
    report["candidate_score_difference_depends_on_evidence"] = bool(
        not torch.allclose(s0[:, 0] - s0[:, 1], s1[:, 0] - s1[:, 1]))
    report["question_cache_shared_by_branches"] = ["pretrained_temporal", "random_temporal"]
    report["runtime_seconds"] = round(time.time() - t0, 3)
    report["status"] = "PASS" if (report["fixed_projection_hashes_match"] and
                                      report["candidate_score_difference_depends_on_evidence"] and
                                      report["tokenizer_ids_identical_pretrained_random"] and
                                      report["mixed_vs_singleton_max_abs"] < 0.05) else "FAIL"
    np.savez_compressed(args.out.with_suffix(".npz"), **payload)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise SystemExit(2)


if __name__ == "__main__": main()
