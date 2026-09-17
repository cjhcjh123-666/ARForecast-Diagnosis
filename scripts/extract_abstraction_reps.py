"""Extract frozen representations of the abstraction probe pool (multi-layer).

For one (model, init, seed): loads the model with the *frozen* loader from the forecasting suite
(same attention implementation for pretrained and random), embeds the 4380-window probe pool with
`output_hidden_states=True`, and stores the final non-pad token state at 5 depth positions
(early, 25%, 50%, 75%, final) as float16.

Outputs
  results/temporal_abstraction/primitives/pool_s<seed>.npz     (ctx/fut, written once)
  results/temporal_abstraction/primitives/pool_s<seed>.jsonl   (latents per window, written once)
  results/temporal_abstraction/reps/<key>_<init>_s<seed>.npz   (h_<depth> arrays)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from scripts.openllm_suite import prompts, resolve_path, load_lm  # noqa: E402
from temporal_abstraction.primitive_targets import build_pool  # noqa: E402

OUT_POOL = Path(os.environ.get("TA_POOL_DIR", REPO / "results/temporal_abstraction/primitives"))
OUT_REPS = Path(os.environ.get("TA_REPS_DIR", REPO / "results/temporal_abstraction/reps"))


def pool_paths(seed: int):
    return OUT_POOL / f"pool_s{seed}.npz", OUT_POOL / f"pool_s{seed}.jsonl"


def ensure_pool(seed: int):
    npz, js = pool_paths(seed)
    if npz.is_file() and js.is_file():
        return npz, js
    OUT_POOL.mkdir(parents=True, exist_ok=True)
    pool = build_pool(seed)
    np.savez(npz, ctx=np.stack([w.ctx for w in pool]).astype(np.float32),
             fut=np.stack([w.fut for w in pool]).astype(np.float32))
    with open(js, "w") as f:
        for i, w in enumerate(pool):
            rec = {"index": i}
            rec.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in w.latents.items()})
            f.write(json.dumps(rec) + "\n")
    return npz, js


@torch.inference_mode()
def embed_layers(model, tok, ctx, device, layer_ids, bs=48):
    """Final non-padding token state at the requested hidden-state indices."""
    acc = {k: [] for k in layer_ids}
    n_hidden = None
    for st in range(0, len(ctx), bs):
        chunk = ctx[st:st + bs]
        enc = tok(prompts(chunk), return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        n_hidden = len(hs)
        mask = enc["attention_mask"].bool()
        idx = mask.sum(1) - 1
        for k in layer_ids:
            h = hs[k]
            acc[k].append(h[torch.arange(len(chunk), device=device), idx].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}, n_hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--path", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--inits", default="pretrained,random")
    ap.add_argument("--depth-fracs", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--include-early", action="store_true", default=True)
    a = ap.parse_args()
    OUT_REPS.mkdir(parents=True, exist_ok=True)
    path = resolve_path(a.model_id, local=a.path)
    fracs = [float(x) for x in a.depth_fracs.split(",") if x]
    for seed in [int(x) for x in a.seeds.split(",") if x]:
        npz, _ = ensure_pool(seed)
        ctx = np.load(npz)["ctx"]
        for init in [x for x in a.inits.split(",") if x]:
            out = OUT_REPS / f"{a.model_key}_{init}_s{seed}.npz"
            if out.is_file():
                print(f"[skip] {out.name}", flush=True); continue
            model, tok = load_lm(path, init, seed, a.device)
            try:
                n_layers = model.config.num_hidden_layers
                sel = sorted({max(1, int(round(fr * n_layers))) for fr in fracs} | ({1} if a.include_early else set()))
                sel = [min(s, n_layers) for s in sel]
                H, n_hidden = embed_layers(model, tok, ctx, a.device, sel)
                payload = {f"h_L{k}": H[k].astype(np.float16) for k in sel}
                payload["layer_ids"] = np.asarray(sel, dtype=np.int32)
                payload["n_hidden_states"] = np.asarray([n_hidden], dtype=np.int32)
                payload["hidden_size"] = np.asarray([H[sel[-1]].shape[1]], dtype=np.int32)
                np.savez(out, **payload)
                print(f"[{a.model_key}/{init}/s{seed}] layers={sel} n_hidden_states={n_hidden} "
                      f"dim={H[sel[-1]].shape[1]} -> {out.name}", flush=True)
            finally:
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
