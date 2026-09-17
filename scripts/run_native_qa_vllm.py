"""Native QA via local vLLM endpoints (same protocol as run_native_qa.py, ~10-20x faster).

One request per item/condition sent to an OpenAI-compatible endpoint with greedy decoding;
raw prompt, raw response, parsed answer, correctness, task type and condition are saved.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.extract_qa_reps import adapter, pick_items  # noqa: E402

OUT = REPO / "results/temporal_abstraction/qa"


def call(base, model, prompt, max_tokens=64, timeout=180):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
                       "top_p": 1.0}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="irts")
    ap.add_argument("--task-types", default="temporal_relationship")
    ap.add_argument("--per-task", type=int, default=150)
    ap.add_argument("--conditions", default="full,question_only,shuffled_ts,reversed_ts")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mod = adapter(a.benchmark)
    items = pick_items(mod.load_items(), [t for t in a.task_types.split(",") if t], a.per_task, a.seed)
    base = f"http://127.0.0.1:{a.port}"
    rows = []
    def one(job):
        item, cond = job
        prompt = mod.render(item, cond, a.seed)
        try:
            txt, dt = call(base, a.served_name, prompt, a.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            txt, dt = f"<error: {e}>", -1.0
        pred = mod.parse(txt, item.get("answer_format", "multiple_choice_abcd")) if hasattr(mod, "parse") else None
        return dict(benchmark=a.benchmark, model=a.model_key, condition=cond, item_id=item["id"],
                    task_type=item["task_type"], gold=item["gold"], parsed=pred or "",
                    correct=int(bool(pred) and mod.is_correct(pred, item["gold"])),
                    latency_s=round(dt, 2), raw_prompt=prompt[:2000], raw_response=txt[:2000])
    jobs = [(i, c) for c in [x for x in a.conditions.split(",") if x] for i in items]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
    out = OUT / f"native_{a.benchmark}_{a.model_key}_s{a.seed}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    acc = {}
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        acc[cond] = sum(r["correct"] for r in sub) / len(sub)
    print(f"wrote {out.name}: " + ", ".join(f"{k}={v:.3f}" for k, v in acc.items()), flush=True)


if __name__ == "__main__":
    main()
