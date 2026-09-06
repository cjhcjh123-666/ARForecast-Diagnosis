"""Merge suite + moirai deliverables into results/iclr/tsfm_deliver, run QA."""
import csv, json, sys
from pathlib import Path
import numpy as np

OUT = Path("results/iclr/tsfm_deliver")
MOI = Path("results/iclr/tsfm_deliver_moirai")
fields = ["model","init","seed","task","test_set","n_test","accuracy","balanced_accuracy",
          "macro_f1","recall_trend","recall_periodic","recall_local","mse","head_params"]

def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))

rows = []
if (OUT/"metrics.csv").is_file(): rows += load(OUT/"metrics.csv")
if (MOI/"metrics.csv").is_file(): rows += load(MOI/"metrics.csv")

# dedupe (same model/init/seed/task/test_set): keep last
seen = {}
for r in rows:
    key = (r["model"], r["init"], r["seed"], r["task"], r["test_set"])
    seen[key] = r
rows = list(seen.values())

# copy moirai per-window files
for f in (MOI/"perwindow").glob("*.npz"):
    dst = OUT/"perwindow"/f.name
    if not dst.exists():
        import shutil; shutil.copy2(f, dst)

with open(OUT/"metrics.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in sorted(rows, key=lambda r:(r["model"],r["seed"],r["init"],r["task"],r["test_set"])):
        w.writerow({k: r.get(k,"") for k in fields})

# QA: table of counts
from collections import Counter
c = Counter((r["model"], r["init"], r["task"]) for r in rows)
print("=== row counts (model/init/task -> n) ===")
for k in sorted(c): print(f"{k[0]:<20} {k[1]:<10} {k[2]:<15} {c[k]}")
print(f"total rows = {len(rows)}")

# print aggregate summary: mean over seeds of key metrics per model/init
print("\n=== A_orig acc (mean over seeds) ===")
for m in sorted(set(r["model"] for r in rows)):
    for ini in ["pretrained","random"]:
        vals=[float(r["accuracy"]) for r in rows if r["model"]==m and r["init"]==ini and r["task"]=="A_recog_orig" and r["accuracy"]]
        if vals: print(f"  {m:<20} {ini:<10} {np.mean(vals):.4f}")
print("\n=== B_readout MSE (mean over seeds) ===")
for m in sorted(set(r["model"] for r in rows)):
    for ini in ["pretrained","random"]:
        vals=[float(r["mse"]) for r in rows if r["model"]==m and r["init"]==ini and r["task"]=="B_readout" and r["mse"]]
        if vals: print(f"  {m:<20} {ini:<10} {np.mean(vals):.4f}")
print("\n=== C bal3 balanced_acc (mean over seeds) ===")
for m in sorted(set(r["model"] for r in rows)):
    for ini in ["pretrained","random"]:
        vals=[float(r["balanced_accuracy"]) for r in rows if r["model"]==m and r["init"]==ini and r["task"]=="C_routing" and r["test_set"].startswith("bal3_s") and r["balanced_accuracy"]]
        if vals: print(f"  {m:<20} {ini:<10} {np.mean(vals):.4f}")
print("\n=== native MSE (mean over seeds) ===")
for m in sorted(set(r["model"] for r in rows)):
    for ini in ["pretrained","random"]:
        vals=[float(r["mse"]) for r in rows if r["model"]==m and r["init"]==ini and r["task"]=="native_forecast" and r["mse"]]
        if vals: print(f"  {m:<20} {ini:<10} {np.mean(vals):.4f}")
