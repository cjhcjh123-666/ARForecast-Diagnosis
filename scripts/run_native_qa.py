"""Native QA behaviour: (time series, question) -> answer, under four input conditions.

Conditions (locked): full / question_only / shuffled_ts / reversed_ts.
Runs on pretrained models only — a matched-random LM cannot read the natural-language question,
so native QA can never be used as pretrained-vs-random attribution (see the factorized protocol).
Per item we save the raw prompt, raw response, parsed answer, correctness, task type and condition.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from scripts.openllm_suite import resolve_path, load_lm  # noqa: E402
from scripts.extract_qa_reps import adapter, pick_items, validate_model_identity  # noqa: E402

OUT = REPO / "results/temporal_abstraction/corrected_v2/qa/native"


def prepare_decoder_tokenizer(tok):
    """Left-pad batched decoder-only prompts so generation starts after real input."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="irts")
    ap.add_argument("--task-types", default="temporal_relationship")
    ap.add_argument("--per-task", type=int, default=150)
    ap.add_argument("--conditions", default="full,question_only,shuffled_ts")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--path", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    mod = adapter(a.benchmark)
    items = pick_items(mod.load_items(), [t for t in a.task_types.split(",") if t], a.per_task, a.seed)
    path = resolve_path(a.model_id, local=a.path)
    identity = validate_model_identity(a.model_key, path)
    model, tok = load_lm(path, "pretrained", a.seed, a.device)
    tok = prepare_decoder_tokenizer(tok)
    rows = []
    try:
        for cond in [c for c in a.conditions.split(",") if c]:
            texts = [mod.render(i, cond, a.seed) for i in items]
            for st in range(0, len(items), a.batch_size):
                chunk = texts[st:st + a.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False,
                          truncation=True, max_length=1024).to(a.device)
                with torch.inference_mode():
                    gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                         pad_token_id=tok.pad_token_id, use_cache=True)
                new = gen[:, enc["input_ids"].shape[1]:]
                for k, row in enumerate(new):
                    txt = tok.decode(row, skip_special_tokens=True)
                    item = items[st + k]
                    pred = mod.parse(txt, item.get("answer_format", "multiple_choice_abcd")) \
                        if hasattr(mod, "parse") else None
                    rows.append(dict(benchmark=a.benchmark, model=a.model_key, condition=cond, item_id=item["id"],
                                     task_type=item["task_type"], gold=item["gold"], parsed=pred if pred else "",
                                     correct=int(bool(pred) and mod.is_correct(pred, item["gold"])),
                                     protocol_version="irts_corrected_v2_left_pad_exact_lexemes",
                                     model_type=identity["model_type"],
                                     config_sha256=identity["config_sha256"],
                                     raw_prompt=chunk[k][:2000], raw_response=txt[:2000]))
            print(f"[{a.model_key}/{cond}] done {len(items)} items", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()
    out = a.out_dir / f"native_{a.benchmark}_{a.model_key}_s{a.seed}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    acc = {}
    for cond in {r["condition"] for r in rows}:
        sub = [r for r in rows if r["condition"] == cond]
        acc[cond] = sum(r["correct"] for r in sub) / len(sub)
    print(f"wrote {out.name}: " + ", ".join(f"{k}={v:.3f}" for k, v in acc.items()), flush=True)


if __name__ == "__main__":
    main()
