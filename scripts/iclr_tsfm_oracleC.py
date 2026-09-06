"""Oracle-expert-label C variant (literal task-sheet reading).

Router is trained on the 270 clean trend/periodic/local windows with labels =
argmin future-MSE expert on each clean window (NOT family id), then evaluated
on the same two balanced OOD sets (bal3, bal3n) used by the family-label C.

Writes per-window npz as pw_oracle_<model>_<init>_s<seed>.npz and appends rows
with task='C_routing_oracle' (merge via rebuild_metrics_from_pw.py).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from scripts.iclr_tsfm_deliver import (KINDS, EXPERTS, expert_errors, recog_metrics3,
                                       make_extractor)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3_8b_base,chronos_t5-small,chronos_t5-base,chronos_bolt-small,timesfm_2.5-200m,moment-1-large,moirai-1.1-R-small")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", type=Path, default=Path("results/iclr/tsfm_deliver"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "perwindow").mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    for name in a.models.split(","):
        for seed in seeds:
            n, ntr = 150, 90
            ctx, fut, kk = build_labeled_windows(KINDS, 64, 16, n, seed)
            clean270 = np.concatenate([np.arange(k * n, k * n + ntr) for k in range(3)])
            err_clean = expert_errors(ctx[clean270], fut[clean270])
            oracle_clean = err_clean.argmin(1)
            bal = {}
            for tag, root in [("bal3", f"results/iclr/e4_balanced3/windows/bal3_s{seed}.npz"),
                              ("bal3n", f"results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz")]:
                if Path(root).is_file():
                    z = np.load(root); bal[tag] = (z["ctx"], z["fut"], z["oracle"])
            for init in ["pretrained", "random"]:
                try:
                    embed, dim, _ = make_extractor(name, init, seed, a.device)
                except Exception as e:
                    print(f"[{name}/{init}/s{seed}] extractor FAIL {type(e).__name__}: {e}", flush=True)
                    continue
                Hc = embed(ctx[clean270])
                w, mu, sd = softmax_regression(Hc, oracle_clean, n_iter=2000)
                pw = {"model": name, "init": init, "seed": seed}
                for tag in sorted(bal):
                    bctx, bfut, bor = bal[tag]
                    Hb = embed(bctx)
                    _, rp = softmax_predict(Hb, w, mu, sd)
                    err = expert_errors(bctx, bfut)
                    ba, mf1, rec = recog_metrics3(rp, bor)
                    mse_c = float(np.mean(err[np.arange(len(rp)), rp]))
                    print(f"[{name}/{init}/s{seed}] C_oracle_{tag} ba={ba:.4f} f1={mf1:.4f} "
                          f"recT={rec[0]:.3f} recP={rec[1]:.3f} recL={rec[2]:.3f} mse={mse_c:.4f}", flush=True)
                    pw[f"c_{tag}_oracle"] = bor; pw[f"c_{tag}_pred"] = rp; pw[f"c_{tag}_err"] = err
                np.savez(a.out / "perwindow" / f"pw_oracle_{name}_{init}_s{seed}.npz", **pw)
        print(f"[{name}] oracle-C finished", flush=True)

if __name__ == "__main__":
    main()
