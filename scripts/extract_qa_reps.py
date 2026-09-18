"""Extract frozen representations of QA items (question branch and temporal branch).

For each item and condition we store two isolated representations:
  h_q : question and choices with every labelled series/timestamp array removed (pretrained only)
  h_x : labelled temporal arrays only; it never contains the question or answer choices

The factorized protocol (docs/temporal_abstraction/02_protocol_lock.md) compares
`h_x(pretrained)` vs `h_x(random)` with everything else held fixed.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from scripts.openllm_suite import resolve_path, load_lm  # noqa: E402

OUT = REPO / "results/temporal_abstraction/corrected_v2/qa/reps"


def validate_model_identity(model_key: str, path: str | Path) -> dict:
    """Reject a known checkpoint alias mismatch and return cache provenance."""
    path = Path(path)
    config_path = path / "config.json"
    if not config_path.is_file():
        return {"model_path": str(path), "model_type": "unknown", "config_sha256": ""}
    raw = config_path.read_bytes()
    config = json.loads(raw)
    model_type = str(config.get("model_type", "unknown"))
    architectures = [str(x) for x in config.get("architectures", [])]
    if model_key == "qwen3_8b_base" and (
        model_type != "qwen3" or "Qwen3ForCausalLM" not in architectures
    ):
        raise ValueError(
            f"{model_key} requires official Qwen3ForCausalLM/qwen3 config; "
            f"found architectures={architectures}, model_type={model_type} at {path}"
        )
    return {
        "model_path": str(path.resolve()),
        "model_type": model_type,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
    }


def adapter(name: str):
    if name == "irts":
        from temporal_abstraction.qa_adapters import irts as m
    elif name == "arfbench":
        from temporal_abstraction.qa_adapters import arfbench as m
    elif name == "tsqa":
        from temporal_abstraction.qa_adapters import tsqa as m
    elif name == "scits":
        from temporal_abstraction.qa_adapters import scits as m
    else:
        raise ValueError(name)
    return m


def pick_items(items, task_types, per_task, seed):
    rng = np.random.default_rng(seed)
    out = []
    for t in task_types:
        sub = [i for i in items if i["task_type"] == t]
        idx = rng.permutation(len(sub))[:per_task]
        out += [sub[i] for i in sorted(idx)]
    return out


@torch.inference_mode()
def embed_texts(model, tok, texts, device, bs=16):
    acc = []
    for st in range(0, len(texts), bs):
        chunk = texts[st:st + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False,
                  truncation=True, max_length=1024).to(device)
        h = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[-1]
        mask = enc["attention_mask"].bool()
        idx = mask.sum(1) - 1
        acc.append(h[torch.arange(len(chunk), device=device), idx].float().cpu().numpy())
    return np.concatenate(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--task-types", default="")
    ap.add_argument("--per-task", type=int, default=150)
    ap.add_argument("--conditions", default="full,question_only,shuffled_ts,reversed_ts")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--path", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--inits", default="pretrained,random")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    mod = adapter(a.benchmark)
    items = mod.load_items()
    tts = [t for t in a.task_types.split(",") if t] or sorted({i["task_type"] for i in items})
    items = pick_items(items, tts, a.per_task, a.seed)
    conds = [c for c in a.conditions.split(",") if c]
    path = resolve_path(a.model_id, local=a.path)
    identity = validate_model_identity(a.model_key, path)
    for init in [x for x in a.inits.split(",") if x]:
        out = a.out_dir / f"{a.benchmark}_{a.model_key}_{init}_s{a.seed}.npz"
        if out.is_file():
            print("skip", out.name, flush=True); continue
        model, tok = load_lm(path, init, a.seed, a.device)
        try:
            payload = {"ids": np.asarray([i["id"] for i in items])}
            payload.update({k: np.asarray(v) for k, v in identity.items()})
            payload["gold"] = np.asarray([i["gold"] for i in items])
            payload["task_type"] = np.asarray([i["task_type"] for i in items])
            if init == "pretrained":
                payload["h_q"] = embed_texts(
                    model, tok, [mod.render_question(i) for i in items], a.device
                )
            for c in conds:
                payload[f"h_x_{c}"] = embed_texts(
                    model, tok, [mod.render_temporal(i, c, a.seed) for i in items], a.device
                )
            np.savez(out, **payload)
            print(f"[{a.benchmark}/{a.model_key}/{init}] items={len(items)} conds={conds} -> {out.name}", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
