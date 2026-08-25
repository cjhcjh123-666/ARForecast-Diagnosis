"""ICLR item-3: context-only generation baselines (no oracle in the ratio).

For each model x seed, on the SAME stratified 40-window generation set, compute
per-window MSE for:
  - LM frozen generation (parsed windows, from iclr_gen_save_windows JSON)
  - Persistence (repeat last context value)
  - Best fixed expert (target-aware: single lowest-mean-error expert on the set)
  - Context-only router (feature router trained on clean synthetic, deployable)
  - Oracle (best expert per window, target-aware upper reference)
Report ratios LM/baseline and the parse rate.  Pure numpy, no LLM re-run.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from analysis.features import temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS = ["trend", "periodic", "local", "mixture", "regime"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="qwen3_8b")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--gen-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    experts = [TrendExpert(), PeriodicExpert(), LocalExpert()]
    g = json.loads(a.gen_json.read_text())
    report = {"model": a.model_tag, "seeds": seeds}

    for seed in seeds:
        contexts, futures, kind_ids = build_labeled_windows(KINDS, 64, 16, 150, seed)
        n_test = 60
        test_idx = np.concatenate([np.arange(k*150+90, (k+1)*150) for k in range(5)])
        gen_idx = np.concatenate([np.arange(k*n_test, k*n_test+8) for k in range(5)])
        ctx = contexts[test_idx[gen_idx]]; fut = futures[test_idx[gen_idx]]

        # expert errors + oracle per window
        err = np.zeros((len(ctx), 3))
        for j, e in enumerate(experts):
            for i in range(len(ctx)):
                err[i, j] = float(np.mean((e.predict(ctx[i], 16) - fut[i]) ** 2))
        oracle = err.min(axis=1)

        # persistence: repeat last context value
        persistence = np.array([np.mean((np.full(16, ctx[i, -1]) - fut[i]) ** 2) for i in range(len(ctx))])
        # best fixed (target-aware)
        best_fixed_j = int(err.mean(axis=0).argmin())
        best_fixed = err[:, best_fixed_j]
        # context-only router: feature router trained on clean synthetic kinds
        c, _, kk = build_labeled_windows(KINDS, 64, 16, 150, seed)
        tr = np.concatenate([np.arange(k*150, k*150+90) for k in range(3)])
        clean = kk[tr]
        feats_tr = np.stack([temporal_features(x) for x in c[tr]])
        feats_te = np.stack([temporal_features(x) for x in ctx])
        _, rpred = softmax_predict(feats_te, *softmax_regression(feats_tr, clean, n_iter=2000))
        ctx_router = err[np.arange(len(ctx)), rpred]

        # LM per-window MSE (parsed)
        pairs = np.asarray(g[str(seed)]["pairs"], dtype=float)  # [mse, oracle, window_idx]
        pidx = pairs[:, 2].astype(int)
        lm_mse = pairs[:, 0]
        o_mse = pairs[:, 1]

        # align baselines to the parsed windows
        row = {
            "lm_mse": lm_mse.tolist(),
            "oracle_mse": o_mse.tolist(),
            "persistence_mse": persistence[pidx].tolist(),
            "best_fixed_mse": best_fixed[pidx].tolist(),
            "best_fixed_expert": experts[best_fixed_j].name,
            "context_router_mse": ctx_router[pidx].tolist(),
            "parse_rate": len(pairs) / len(ctx),
            "window_indices": pidx.tolist(),
        }
        report[str(seed)] = row
        # summary on parsed windows (align by index: pairs are first n_parsed in order)
        def ratio(baseline): return float(np.mean(lm_mse) / (np.mean(baseline) + 1e-12))
        print(f"[seed {seed}] parse={row['parse_rate']:.2f} "
              f"LM/persist={ratio(persistence[pidx]):.2f} LM/bestfixed={ratio(best_fixed[pidx]):.2f} "
              f"LM/ctxrouter={ratio(ctx_router[pidx]):.2f} LM/oracle={ratio(o_mse):.2f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={a.out}")

if __name__ == "__main__":
    main()
