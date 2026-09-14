"""Export the open-LLM suite as machine-readable long/wide tables and LaTeX paper tables.

Inputs (never modified): prior-round `results/iclr/tsfm_deliver_v2/metrics.csv`,
`results/open_llm_suite/raw/metrics_*.csv`, `results/open_llm_suite/native/native_local.csv`,
`results/open_llm_suite/raw/tableD_realworld.csv`.
Outputs: results/open_llm_suite/{all_metrics_long.csv,all_metrics_wide.csv,paper_tables.tex}
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/open_llm_suite"
RAW = OUT / "raw"

LMS = [("qwen3_8b_base", "Qwen3-8B"), ("llama31_8b", "Llama-3.1-8B"), ("llama32_3b", "Llama-3.2-3B"),
       ("gemma2_9b", "Gemma-2-9B"), ("gemma2_2b", "Gemma-2-2B"), ("mistral_7b_v03", "Mistral-7B-v0.3"),
       ("deepseek_llm_7b", "DeepSeek-LLM-7B"), ("olmo2_7b", "OLMo-2-7B"), ("olmo2_13b", "OLMo-2-13B"),
       ("deepseek_v2_lite", "DeepSeek-V2-Lite (MoE)")]
PARAMS = {"qwen3_8b_base": "8.2B", "llama31_8b": "8.0B", "llama32_3b": "3.2B", "gemma2_9b": "9.2B",
          "gemma2_2b": "2.6B", "mistral_7b_v03": "7.2B", "deepseek_llm_7b": "6.9B", "olmo2_7b": "6.9B",
          "olmo2_13b": "13B", "deepseek_v2_lite": "15.7B (2.4B act.)"}

long_rows = []
for r in csv.DictReader(open(REPO / "results/iclr/tsfm_deliver_v2/metrics.csv")):
    r = dict(r); r["suite"] = "prior_round"
    long_rows.append(r)
for f in sorted(RAW.glob("metrics_*.csv")):
    name = f.stem[len("metrics_"):]
    for init in ("pretrained", "random"):
        if name.endswith("_" + init):
            model = name[: -len(init) - 1]; break
    else:
        continue
    for r in csv.DictReader(open(f)):
        r = dict(r); r["suite"] = "open_llm_suite"; r["test_set"] = r.get("test_set") or ""
        long_rows.append(r)

fields = ["suite", "model", "init", "seed", "task", "test_set", "n_test", "accuracy", "balanced_accuracy",
          "macro_f1", "recall_trend", "recall_periodic", "recall_local", "mse", "routed_mse", "head_params"]
with open(OUT / "all_metrics_long.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
    for r in long_rows:
        w.writerow({k: r.get(k, "") for k in fields})
print("wrote all_metrics_long.csv rows:", len(long_rows))

# ---- wide: one row per (model, init, seed) ----
def num(x):
    try:
        v = float(x); return v if v == v else np.nan
    except (TypeError, ValueError):
        return np.nan

W = defaultdict(dict)
for r in long_rows:
    key = (r["model"], r["init"], str(r["seed"]))
    ts = r.get("test_set", "") or ""
    t = r["task"]
    if t == "A_recog_orig": W[key]["clean_acc"] = num(r["accuracy"])
    elif t == "A_recog_shuf": W[key]["shuffle_acc"] = num(r["accuracy"])
    elif t == "B_readout": W[key]["readout_mse"] = num(r["mse"])
    elif t == "B_readout_linear": W[key]["readout_mse"] = num(r["mse"])
    elif t == "B_readout_mlp": W[key]["readout_mse_mlp"] = num(r["mse"])
    elif t == "C_routing" and ts.startswith("bal3_s"): W[key]["family_ba_bal3"] = num(r["balanced_accuracy"])
    elif t == "C_routing" and ts.startswith("bal3n_s"): W[key]["family_ba_bal3n"] = num(r["balanced_accuracy"])
    elif t == "C_routing_oracle" and ts.startswith("bal3_s"): W[key]["oracle_ba_bal3"] = num(r["balanced_accuracy"])
    elif t == "C_family_bal3": W[key]["family_ba_bal3"] = num(r["balanced_accuracy"])
    elif t == "C_family_bal3n": W[key]["family_ba_bal3n"] = num(r["balanced_accuracy"])
    elif t == "C2_oracle_bal3": W[key]["oracle_ba_bal3"] = num(r["balanced_accuracy"])
    elif t == "C2_oracle_bal3n": W[key]["oracle_ba_bal3n"] = num(r["balanced_accuracy"])
    elif t == "R_mlp_bal3": W[key]["mlp_router_ba_bal3"] = num(r["balanced_accuracy"])
    elif t == "R_mlp_bal3n": W[key]["mlp_router_ba_bal3n"] = num(r["balanced_accuracy"])
    elif t == "native_forecast": W[key]["native_mse"] = num(r["mse"])

native = OUT / "native/native_local.csv"
if native.is_file():
    for r in csv.DictReader(open(native)):
        key = (r["model"], r["init"], str(r["seed"]))
        W[key]["native_parse_rate"] = num(r["parse_rate"]); W[key]["native_mse"] = num(r["native_mse"])
cols = ["model", "init", "seed", "clean_acc", "shuffle_acc", "readout_mse", "readout_mse_mlp",
        "family_ba_bal3", "family_ba_bal3n", "oracle_ba_bal3", "oracle_ba_bal3n",
        "mlp_router_ba_bal3", "mlp_router_ba_bal3n", "native_parse_rate", "native_mse"]
with open(OUT / "all_metrics_wide.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for key in sorted(W, key=lambda k: (k[0], k[1], int(k[2]))):
        d = W[key]
        w.writerow({**{ "model": key[0], "init": key[1], "seed": key[2]},
                    **{c: ("" if d.get(c, np.nan) != d.get(c, np.nan) else round(d[c], 4)) for c in cols[3:]}})
print("wrote all_metrics_wide.csv rows:", len(W))

# ---- mean helpers ----
def mean(model, init, col, seeds=("7", "17", "27")):
    v = [W.get((model, init, s), {}).get(col, np.nan) for s in seeds]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else np.nan

def per_seed_delta(model, col):
    out = []
    for s in ("7", "17", "27"):
        a = W.get((model, "pretrained", s), {}).get(col, np.nan)
        b = W.get((model, "random", s), {}).get(col, np.nan)
        if a == a and b == b:
            out.append(a - b)
    return out

def f(x, nd=3):
    return "--" if x != x else f"{x:.{nd}f}"

# ---- LaTeX tables ----
lines = ["% auto-generated by scripts/openllm_export.py (do not edit by hand)",
         "% Table A: matched-scale open language models, mean over seeds 7/17/27",
         "\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "Model & Params & Clean P/R & Shuffle P/R & Readout P/R & OOD P/R & $\\Delta$BA & Oracle $\\Delta$ \\\\",
         "\\midrule"]
for model, label in LMS:
    cP, cR = mean(model, "pretrained", "clean_acc"), mean(model, "random", "clean_acc")
    sP, sR = mean(model, "pretrained", "shuffle_acc"), mean(model, "random", "shuffle_acc")
    rP, rR = mean(model, "pretrained", "readout_mse"), mean(model, "random", "readout_mse")
    fP, fR = mean(model, "pretrained", "family_ba_bal3"), mean(model, "random", "family_ba_bal3")
    d = np.nanmean(per_seed_delta(model, "family_ba_bal3")) if per_seed_delta(model, "family_ba_bal3") else np.nan
    od = np.nanmean(per_seed_delta(model, "oracle_ba_bal3")) if per_seed_delta(model, "oracle_ba_bal3") else np.nan
    lines.append(f"{label} & {PARAMS[model]} & {f(cP)}/{f(cR)} & {f(sP)}/{f(sR)} & {f(rP)}/{f(rR)} & "
                 f"{f(fP)}/{f(fR)} & {f(d)} & {f(od)} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "", "% Table D: real-world zero-shot routed MSE (pretrained / random)",
          "\\begin{tabular}{lrrrr}", "\\toprule",
          "Model & Wins vs random & Median $\\Delta$\\% & Wins vs features & FDR-sig. (vs random) \\\\", "\\midrule"]
TDcsv = RAW / "tableD_realworld.csv"; TDS = RAW / "tableD_realworld_stats.csv"
if TDcsv.is_file():
    D = list(csv.DictReader(open(TDcsv)))
    ST = list(csv.DictReader(open(TDS))) if TDS.is_file() else []
    for model, label in LMS:
        sub = [r for r in D if r["model"] == model and r.get("random") not in (None, "")]
        if not sub:
            continue
        w = sum(1 for r in sub if float(r["pretrained"]) < float(r["random"]))
        wf = sum(1 for r in sub if r.get("feature_router") not in (None, "") and float(r["pretrained"]) < float(r["feature_router"]))
        rel = [100 * (float(r["pretrained"]) - float(r["random"])) / float(r["random"]) for r in sub if float(r["random"])]
        sig = sum(1 for r in ST if r["model"] == model and r["contrast"] == "P-R" and r.get("q_value") not in (None, "")
                  and float(r["q_value"]) < 0.05 and float(r["delta"]) < 0)
        lines.append(f"{label} & {w}/{len(sub)} & {np.median(rel):+.1f} & {wf}/{len(sub)} & {sig}/{len(sub)} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "", "% Table E: native numerical generation (pretrained vs matched random)",
          "\\begin{tabular}{lrrrr}", "\\toprule",
          "Model & Parse rate (P) & Native MSE & MSE / oracle & MSE / best-fixed \\\\", "\\midrule"]
if native.is_file():
    NR = list(csv.DictReader(open(native)))
    for model, label in LMS:
        r = next((x for x in NR if x["model"] == model and x["init"] == "pretrained"), None)
        if not r:
            continue
        nm = num(r.get("native_mse")); orc = num(r.get("oracle_mse")); bf = num(r.get("best_fixed_mse"))
        lines.append(f"{label} & {f(num(r['parse_rate']), 3)} & {f(nm, 3)} & {f(nm/orc, 2) if nm == nm else '--'} & "
                     f"{f(nm/bf, 2) if nm == nm else '--'} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", ""]
(OUT / "paper_tables.tex").write_text("\n".join(lines) + "\n")
print("wrote paper_tables.tex")
