"""Rebuild metrics.csv deterministically from per-window npz files.

Handles both pw_<model>_<init>_s<seed>.npz (tasks A/B/C_routing/native) and
pw_oracle_<model>_<init>_s<seed>.npz (task C_routing_oracle).
"""
import csv, sys
from pathlib import Path
import numpy as np

DIM = {  # hidden dim used for B head_params = dim*16+16
    "qwen3_8b_base": 4096, "chronos_t5-small": 512, "chronos_t5-base": 768,
    "chronos_bolt-small": 512, "timesfm_2.5-200m": 1280, "moment-1-large": 1024,
    "moirai-1.1-R-small": 384,
}
OUT = Path(sys.argv[1]) if len(sys.argv)>1 else Path("results/iclr/tsfm_deliver")

def recog_metrics3(pred, oracle):
    n = len(pred); cm = np.zeros((3, 3), int)
    for t, pp in zip(oracle, pred):
        if t < 3 and pp < 3: cm[t, pp] += 1
    rec = np.array([cm[i, i] / (cm[i].sum() + 1e-12) for i in range(3)])
    prec = np.array([cm[:, i][i] / (cm[:, i].sum() + 1e-12) if cm[:, i].sum() > 0 else 0 for i in range(3)])
    f1 = float(np.mean([2 * p * r / (p + r + 1e-12) for p, r in zip(prec, rec)]))
    return float(rec.mean()), f1, rec

fields = ["model","init","seed","task","test_set","n_test","accuracy","balanced_accuracy",
          "macro_f1","recall_trend","recall_periodic","recall_local","mse","head_params"]
rows = []
files = sorted((OUT/"perwindow").glob("pw_*.npz"))  # matches pw_* and pw_oracle_*; parse_stem disambiguates

def parse_stem(stem):
    # stem like 'pw_qwen3_8b_base_pretrained_s7' or 'pw_oracle_qwen3_8b_base_pretrained_s7'
    oracle = stem.startswith("pw_oracle_")
    core = stem[len("pw_oracle_"):] if oracle else stem[len("pw_"):]
    parts = core.rsplit("_s", 1)
    seed = int(parts[1]); mid = parts[0]
    assert mid.endswith("_pretrained") or mid.endswith("_random"), mid
    init = "pretrained" if mid.endswith("_pretrained") else "random"
    model = mid[: -len("_" + init)]
    return oracle, model, init, seed

for pw in files:
    oracle, model, init, seed = parse_stem(pw.stem)
    z = np.load(pw)
    rec = {"model": model, "init": init, "seed": seed}
    ctask = "C_routing_oracle" if oracle else "C_routing"
    if not oracle:
        if "a_pred" in z:
            rows.append({**rec, "task": "A_recog_orig", "test_set": "synth5", "n_test": len(z["a_pred"]),
                         "accuracy": round(float(np.mean(z["a_pred"] == z["a_true"])), 4)})
        if "as_pred" in z:
            rows.append({**rec, "task": "A_recog_shuf", "test_set": "synth5", "n_test": len(z["as_pred"]),
                         "accuracy": round(float(np.mean(z["as_pred"] == z["as_true"])), 4)})
        if "b_pred" in z:
            mse = float(np.mean((z["b_pred"] - z["b_future"]) ** 2))
            dim = DIM.get(model, 0)
            rows.append({**rec, "task": "B_readout", "test_set": "synth5", "n_test": len(z["b_pred"]),
                         "mse": round(mse, 4), "head_params": dim * 16 + 16 if dim else ""})
        if "n_pred" in z:
            mse = float(np.mean((z["n_pred"] - z["n_future"]) ** 2))
            rows.append({**rec, "task": "native_forecast", "test_set": "synth5", "n_test": len(z["n_pred"]),
                         "mse": round(mse, 4)})
    for tag in ["bal3", "bal3n"]:
        o, pr, er = f"c_{tag}_oracle", f"c_{tag}_pred", f"c_{tag}_err"
        if o in z and pr in z and er in z:
            ba, mf1, r3 = recog_metrics3(z[pr], z[o])
            mse = float(np.mean(z[er][np.arange(len(z[pr])), z[pr]]))
            rows.append({**rec, "task": ctask, "test_set": f"{tag}_s{seed}", "n_test": len(z[pr]),
                         "balanced_accuracy": round(ba, 4), "macro_f1": round(mf1, 4),
                         "recall_trend": round(r3[0], 4), "recall_periodic": round(r3[1], 4),
                         "recall_local": round(r3[2], 4), "mse": round(mse, 4)})
rows.sort(key=lambda r: (r["model"], r["seed"], r["init"], r["task"], r["test_set"]))
with open(OUT / "metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, "") for k in fields})
print(f"wrote {len(rows)} rows -> {OUT/'metrics.csv'}")
