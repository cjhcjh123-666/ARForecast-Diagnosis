"""Primitive + layer-wise probes over frozen LM representations (Level 1 / 1b).

One shard = one (model, seed, init). Writes a long-format CSV:
  model, family, pretrained_or_random, task, ability_level, dataset, split, seed, layer,
  condition, metric, score, delta

`condition` = representation source: the LM (last non-pad token at a given depth) or
`handcrafted_features` (analysis.features.temporal_features) as a learnability reference.
`delta` is filled in by the aggregator (paired pretrained − random on the same windows).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.features import temporal_features
from temporal_abstraction.primitive_targets import build_pool, target_labels
from temporal_abstraction.probes import run_probe

POOL = REPO / "results/temporal_abstraction/primitives"
REPS = REPO / "results/temporal_abstraction/reps"
OUT = REPO / "results/temporal_abstraction/primitives"
FAMILY = {"qwen3_8b_base": "Qwen", "llama31_8b": "Llama", "llama32_3b": "Llama", "gemma2_9b": "Gemma",
          "gemma2_2b": "Gemma", "mistral_7b_v03": "Mistral", "deepseek_llm_7b": "DeepSeek",
          "deepseek_v2_lite": "DeepSeek", "olmo2_7b": "OLMo", "olmo2_13b": "OLMo"}
LAYER_LEVEL = {1: "early", 0.25: "p25", 0.5: "p50", 0.75: "p75", 1.0: "final"}


def ensure_pool_npz(seed):
    p = POOL / f"pool_s{seed}.npz"
    if not p.is_file():
        build_pool(seed)  # writes via extract script's helper contract
        raise SystemExit(f"pool missing: {p} (run scripts/extract_abstraction_reps.py first)")
    return np.load(p)


def handcrafted(H_ctx):
    return np.stack([temporal_features(x) for x in H_ctx]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--targets", default="")
    ap.add_argument("--no-handcrafted", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pool_npz = ensure_pool_npz(a.seed)
    ctx = pool_npz["ctx"]
    pool = build_pool(a.seed)
    T = target_labels(pool)
    want = [t for t in a.targets.split(",") if t] or sorted(T.keys())

    rows = []
    def emit(layer_name, depth_frac, cond):
        for ti, name in enumerate(sorted(T.keys())):
            if name not in want:
                continue
            y, idx, kind = T[name]
            ti_real = sorted(T.keys()).index(name)
            if cond is None:
                continue
            H = cond
            res = run_probe(H[idx], np.asarray(y), ti_real, kind, a.seed)
            for metric, value in res.items():
                if metric.startswith("n_"):
                    continue
                rows.append(dict(model=a.model_key, family=FAMILY.get(a.model_key, ""),
                                 pretrained_or_random=a.init, task=name,
                                 ability_level="L1_primitive" if depth_frac == 1.0 else "L1b_layerwise",
                                 dataset="abstraction_pool", split="train60_test40", seed=a.seed,
                                 layer=layer_name, depth_frac=depth_frac, condition="lm_representation",
                                 probe_kind=kind, metric=metric, score=round(float(value), 5),
                                 n_train=res.get("n_train", ""), n_test=res.get("n_test", "")))

    # handcrafted reference (one row set, layer="n/a")
    if not a.no_handcrafted:
        Hc = handcrafted(ctx)
        for ti, name in enumerate(sorted(T.keys())):
            if name not in want:
                continue
            y, idx, kind = T[name]
            ti_real = sorted(T.keys()).index(name)
            res = run_probe(Hc[idx], np.asarray(y), ti_real, kind, a.seed)
            for metric, value in res.items():
                if metric.startswith("n_"):
                    continue
                rows.append(dict(model=a.model_key, family=FAMILY.get(a.model_key, ""),
                                 pretrained_or_random="handcrafted", task=name, ability_level="L1_reference",
                                 dataset="abstraction_pool", split="train60_test40", seed=a.seed,
                                 layer="n/a", depth_frac=-1.0, condition="handcrafted_features",
                                 probe_kind=kind, metric=metric, score=round(float(value), 5),
                                 n_train=res.get("n_train", ""), n_test=res.get("n_test", "")))

    rep_file = REPS / f"{a.model_key}_{a.init}_s{a.seed}.npz"
    if rep_file.is_file():
        z = np.load(rep_file)
        layer_ids = z["layer_ids"].tolist()
        n_layers = int(z["n_hidden_states"][0]) - 1 if "n_hidden_states" in z else max(layer_ids)
        for lid in layer_ids:
            frac = round(lid / max(1, n_layers), 4)
            lname = LAYER_LEVEL.get(frac, f"L{lid}")
            H = z[f"h_L{lid}"].astype(np.float32)
            emit(lname, frac, H)
    else:
        print(f"[warn] no representation file for {a.model_key}/{a.init}/s{a.seed}", flush=True)

    out = OUT / f"probes_{a.model_key}_{a.init}_s{a.seed}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out.name} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
