"""Figures A/C/D + robustness table for the open-LLM cross-family attribution suite.

All numbers are read from result CSVs; nothing is hard-coded.
  Figure A  forest plot of family-label OOD routing delta (pretrained - random)
  Figure C  structure vs numerics (routing delta vs frozen-readout MSE delta)
  Figure D  real-world win/loss heatmap (relative routed-MSE change vs random)
Plus results/open_llm_suite/tables/robustness_matrix.csv (supervision/expert/MLP/real-world/native).
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results/open_llm_suite/raw"
TAB = REPO / "results/open_llm_suite/tables"; TAB.mkdir(parents=True, exist_ok=True)
FIG = REPO / "figures/open_llm_suite"; FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

LMS = [("qwen3_8b_base", "Qwen3-8B"), ("llama31_8b", "Llama-3.1-8B"), ("llama32_3b", "Llama-3.2-3B"),
       ("gemma2_9b", "Gemma-2-9B"), ("gemma2_2b", "Gemma-2-2B"), ("mistral_7b_v03", "Mistral-7B-v0.3"),
       ("deepseek_llm_7b", "DeepSeek-LLM-7B"), ("olmo2_7b", "OLMo-2-7B"), ("olmo2_13b", "OLMo-2-13B"),
       ("deepseek_v2_lite", "DeepSeek-V2-Lite (MoE)")]
TSFMS = [("chronos_t5-base", "Chronos-T5-base"), ("chronos_t5-small", "Chronos-T5-small"),
         ("chronos_bolt-small", "Chronos-Bolt-small"), ("timesfm_2.5-200m", "TimesFM-2.5-200m"),
         ("moment-1-large", "MOMENT-1-large"), ("moirai-1.1-R-small", "Moirai-1.1-R-small")]

# ---------------- loaders ----------------
def load_openllm():
    """(model, init) -> (task, group) -> seed -> row  for the 9 new open LMs."""
    D = defaultdict(lambda: defaultdict(dict))
    for f in RAW.glob("metrics_*.csv"):
        name = f.stem[len("metrics_"):]
        for init in ("pretrained", "random"):
            if name.endswith("_" + init):
                model = name[: -len(init) - 1]; break
        else:
            continue
        for r in csv.DictReader(open(f)):
            if r.get("seed") not in (None, ""):
                D[(model, init)][(r["task"], "")][str(r["seed"])] = r
    return D

def load_tsfm():
    """(model, init) -> (task, test-group) -> seed -> row  for prior-round TSFMs + Qwen3-8B.

    NOTE: the prior-round CSV stores bal3 and bal3n rows under the same `task` name, so the
    test set must be part of the key (otherwise one overwrites the other).
    """
    D = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(REPO / "results/iclr/tsfm_deliver_v2/metrics.csv")):
        ts = r.get("test_set", "") or ""
        g = "bal3n" if ts.startswith("bal3n") else ("bal3" if ts.startswith("bal3") else ts)
        D[(r["model"], r["init"])][(r["task"], g)][str(r["seed"])] = r
    return D

OPEN_D = load_openllm(); TSFM_D = load_tsfm()
print(f"loaded {len(OPEN_D)} open-LLM (model,init) runs and {len(TSFM_D)} prior-round runs")

def num(x):
    try:
        v = float(x)
        return v if v == v else np.nan
    except (TypeError, ValueError):
        return np.nan

def per_seed_delta(D, model, task, key, test=None):
    """Per-seed pretrained - random for a metric (seeds present in both branches)."""
    out = {}
    for seed in ("7", "17", "27"):
        pr = D.get((model, "pretrained"), {}).get((task, test or ""), {}).get(seed)
        rr = D.get((model, "random"), {}).get((task, test or ""), {}).get(seed)
        if not pr or not rr:
            continue
        a, b = num(pr.get(key)), num(rr.get(key))
        if a == a and b == b:
            out[seed] = a - b
    return out

def boot_ci(seed_deltas, n=10000):
    d = np.asarray(list(seed_deltas.values()), float)
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# ---------------- Figure A: forest plot ----------------
items = []
for model, label in LMS:
    if model == "qwen3_8b_base":
        ds = per_seed_delta(TSFM_D, model, "C_routing", "balanced_accuracy", test="bal3")
    else:
        ds = per_seed_delta(OPEN_D, model, "C_family_bal3", "balanced_accuracy")
    if ds:
        m, lo, hi = boot_ci(ds)
        items.append((label, m, lo, hi, "LM", sorted(ds.values())))
for model, label in TSFMS:
    ds = per_seed_delta(TSFM_D, model, "C_routing", "balanced_accuracy", test="bal3")
    if ds:
        m, lo, hi = boot_ci(ds)
        items.append((label, m, lo, hi, "TSFM", sorted(ds.values())))

figA_rows = []
for name, m, lo, hi, fam, seeds in items:
    figA_rows.append(dict(model=name, family=fam, delta=round(m, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                          per_seed=";".join(f"{s:+.3f}" for s in seeds), n_seeds=len(seeds)))
with open(TAB / "figA_forest_data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(figA_rows[0].keys())); w.writeheader(); w.writerows(figA_rows)

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srt = sorted(items, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for i, (name, m, lo, hi, fam, _) in enumerate(srt):
        col = "#1f77b4" if fam == "LM" else "#d62728"
        ax.plot([lo, hi], [i, i], color=col, lw=1.8, solid_capstyle="butt")
        ax.plot([m], [i], marker="o", color=col, ms=5)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(range(len(srt)))
    ax.set_yticklabels([f"{n}  ({f})" for n, _, _, _, f, _ in srt], fontsize=8)
    ax.set_xlabel("Family-label OOD routing gain (pretrained − random, balanced accuracy)")
    ax.set_xlim(-0.30, 0.45)
    ax.set_title("Compositional-transfer gain: open LMs vs time-series foundation models", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "figA_forest.png", dpi=200); fig.savefig(FIG / "figA_forest.pdf")
    print("wrote figA_forest.png/pdf with", len(srt), "entries")

    # ---------------- Figure C: structure vs numerics ----------------
    pts = []
    for model, label in LMS:
        if model == "qwen3_8b_base":
            dr = per_seed_delta(TSFM_D, model, "C_routing", "balanced_accuracy", test="bal3")
            dm = per_seed_delta(TSFM_D, model, "B_readout", "mse")
        else:
            dr = per_seed_delta(OPEN_D, model, "C_family_bal3", "balanced_accuracy")
            dm = per_seed_delta(OPEN_D, model, "B_readout_linear", "mse")
        if dr and dm:
            pts.append((label, float(np.mean(list(dr.values()))), float(np.mean(list(dm.values()))), "LM"))
    for model, label in TSFMS:
        dr = per_seed_delta(TSFM_D, model, "C_routing", "balanced_accuracy", test="bal3")
        dm = per_seed_delta(TSFM_D, model, "B_readout", "mse")
        if dr and dm:
            pts.append((label, float(np.mean(list(dr.values()))), float(np.mean(list(dm.values()))), "TSFM"))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for name, x, y, fam in pts:
        col = "#1f77b4" if fam == "LM" else "#d62728"
        ax.scatter([x], [y], color=col, s=34)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axvline(0, color="gray", ls=":", lw=1); ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("OOD structural-transfer gain ΔBA (pretrained − random)")
    ax.set_ylabel("Frozen numerical-readout ΔMSE (pretrained − random, <0 better)")
    ax.set_title("Structural transfer is decoupled from numerical readout", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "figC_structure_vs_numerics.png", dpi=200)
    fig.savefig(FIG / "figC_structure_vs_numerics.pdf")
    print("wrote figC_structure_vs_numerics.png/pdf")

    # ---------------- Figure D: real-world heatmap ----------------
    d = list(csv.DictReader(open(RAW / "tableD_realworld.csv")))
    models = sorted({r["model"] for r in d}); datasets = sorted({r["dataset"] for r in d})
    M = np.full((len(models), len(datasets)), np.nan)
    for r in d:
        if r.get("random") not in (None, "") and float(r["random"]) != 0:
            M[models.index(r["model"]), datasets.index(r["dataset"])] = \
                100 * (float(r["pretrained"]) - float(r["random"])) / float(r["random"])
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    vmax = float(np.nanpercentile(np.abs(M), 95)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=7)
    for i in range(len(models)):
        for j in range(len(datasets)):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i,j]:+.0f}", ha="center", va="center", fontsize=5.5)
    ax.set_title("Zero-shot routed-MSE change vs matched random (%): blue = pretrained better", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIG / "figD_realworld_heatmap.png", dpi=200)
    fig.savefig(FIG / "figD_realworld_heatmap.pdf")
    print("wrote figD_realworld_heatmap.png/pdf")
except Exception as e:  # noqa: BLE001
    print("figure generation failed:", e)

# ---------------- robustness matrix ----------------
nat = list(csv.DictReader(open(REPO / "results/open_llm_suite/native/native_local.csv")))
def natval(model, init, key):
    for r in nat:
        if r["model"] == model and r["init"] == init and r.get(key) not in (None, ""):
            return float(r[key])
    return np.nan
def d_rel_wins(model):
    sub = [r for r in d if r["model"] == model and r.get("random") not in (None, "")]
    w = sum(1 for r in sub if float(r["pretrained"]) < float(r["random"]))
    rel = [100 * (float(r["pretrained"]) - float(r["random"])) / float(r["random"]) for r in sub if float(r["random"])]
    return w, len(sub), float(np.median(rel)) if rel else np.nan

def five_expert_delta(model, test="bal3"):
    """mean over seeds of (pretrained - random) balanced accuracy of the 5-expert router.

    Prefers the full 10-model table (tableB_all.csv); the older 3-model tableB_openllm.csv
    stores ABSOLUTE balanced accuracies (chance = 1/5), not deltas.
    """
    full = RAW / "tableB_all.csv"
    if full.is_file():
        R = [r for r in csv.DictReader(open(full)) if r["model"] == model and r["test"] == test]
        out = {}
        for seed in ("7", "17", "27"):
            p5 = [num(r["family5"]) for r in R if r["seed"] == seed and r["init"] == "pretrained"]
            q5 = [num(r["family5"]) for r in R if r["seed"] == seed and r["init"] == "random"]
            if p5 and q5 and p5[0] == p5[0] and q5[0] == q5[0]:
                out[seed] = p5[0] - q5[0]
        if out:
            return out
    tb5f = RAW / "tableB_openllm.csv"
    if not tb5f.is_file():
        return {}
    R = [r for r in csv.DictReader(open(tb5f)) if r["model"] == model and r["test"] == test]
    out = {}
    for seed in ("7", "17", "27"):
        p = [num(r["family5"]) for r in R if r["seed"] == seed and r["init"] == "pretrained"]
        q = [num(r["family5"]) for r in R if r["seed"] == seed and r["init"] == "random"]
        if p and q and p[0] == p[0] and q[0] == q[0]:
            out[seed] = p[0] - q[0]
    return out

rows = []
for model, label in LMS:
    if model == "qwen3_8b_base":
        f3 = per_seed_delta(TSFM_D, model, "C_routing", "balanced_accuracy", test="bal3")
        orc = per_seed_delta(TSFM_D, model, "C_routing_oracle", "balanced_accuracy", test="bal3")
        f3n = per_seed_delta(TSFM_D, model, "C_routing", "balanced_accuracy", test="bal3n")
        mlp = {}
    else:
        f3 = per_seed_delta(OPEN_D, model, "C_family_bal3", "balanced_accuracy")
        orc = per_seed_delta(OPEN_D, model, "C2_oracle_bal3", "balanced_accuracy")
        f3n = per_seed_delta(OPEN_D, model, "C_family_bal3n", "balanced_accuracy")
        mlp = per_seed_delta(OPEN_D, model, "R_mlp_bal3", "balanced_accuracy")
    f5 = five_expert_delta(model)
    mlp_delta = {}
    rcf = RAW / "router_capacity_all.csv"
    if rcf.is_file():
        R = [r for r in csv.DictReader(open(rcf)) if r["model"] == model]
        for seed in ("7", "17", "27"):
            p0 = [num(r["mlp64"]) for r in R if r["seed"] == seed and r["init"] == "pretrained"]
            q0 = [num(r["mlp64"]) for r in R if r["seed"] == seed and r["init"] == "random"]
            if p0 and q0 and p0[0] == p0[0] and q0[0] == q0[0]:
                mlp_delta[seed] = p0[0] - q0[0]
    if mlp_delta:
        mlp = mlp_delta
    w, n, med = d_rel_wins(model) if model in {r["model"] for r in d} else (np.nan, 0, np.nan)
    nm = natval(model, "pretrained", "native_mse"); orc_mse = natval(model, "pretrained", "oracle_mse")
    rows.append(dict(
        model=label,
        family_label_delta=round(float(np.mean(list(f3.values()))), 4) if f3 else "",
        oracle_label_delta=round(float(np.mean(list(orc.values()))), 4) if orc else "",
        balanced_n_delta=round(float(np.mean(list(f3n.values()))), 4) if f3n else "",
        mlp_router_delta=round(float(np.mean(list(mlp.values()))), 4) if mlp else "",
        family5_delta=round(float(np.mean(list(f5.values()))), 4) if f5 else "",
        family5_per_seed=";".join(f"{v:+.3f}" for v in f5.values()),
        realworld_wins=f"{w}/{n}" if n else "",
        realworld_median_pct=round(med, 2) if med == med else "",
        native_over_oracle=round(nm / orc_mse, 2) if nm == nm and orc_mse == orc_mse else ""))
with open(TAB / "robustness_matrix.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote robustness_matrix.csv")
for r in rows:
    print(r)


# ---------------- Figure E: model scale + 10-seed stability, Figure F: dimension matching ----------------
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ss = REPO / "results/open_llm_suite/raw/scale_sweep.csv"
    ts = REPO / "results/open_llm_suite/raw/ten_seed_headline.csv"
    dmp = REPO / "results/open_llm_suite/raw/dimension_matching.csv"
    if ss.is_file() and ts.is_file():
        S = list(csv.DictReader(open(ss))); T = list(csv.DictReader(open(ts)))
        order = {"qwen3_0.6b": 0, "qwen3_1.7b": 1, "qwen3_8b_base": 2}
        S = sorted(S, key=lambda r: order.get(r["model"], 9))
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
        ax = axes[0]
        x = np.arange(len(S))
        ax.bar(x, [float(r["delta"]) for r in S],
               yerr=[[float(r["delta"]) - float(r["ci_lo"]) for r in S],
                     [float(r["ci_hi"]) - float(r["delta"]) for r in S]],
               color="#1f77b4", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels([f"{r['params']}\n({r['hidden']}-d)" for r in S], fontsize=7)
        ax.set_ylabel("$\\Delta$BA (pretrained − random)"); ax.set_xlabel("Qwen3 checkpoint size")
        ax.set_title("Scale sweep (seeds 7/17/27)", fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax2 = axes[1]
        for i, r in enumerate(T):
            per = [float(v.split(":")[1]) for v in r["per_seed"].split(";")]
            ax2.plot(range(1, len(per) + 1), per, marker="o", ms=4,
                     label=f"{r['test']} (mean {float(r['delta']):+.3f})")
        ax2.axhline(0, color="gray", ls="--", lw=1)
        ax2.set_xticks(range(1, 11)); ax2.set_xlabel("seed index (7,17,27,37,…,97)")
        ax2.set_ylabel("$\\Delta$BA"); ax2.legend(fontsize=6)
        ax2.set_title("Qwen3-8B headline: 10 seeds", fontsize=8); ax2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "figE_scale_and_seeds.png", dpi=200)
        fig.savefig(FIG / "figE_scale_and_seeds.pdf")
        print("wrote figE_scale_and_seeds.png/pdf")
    if dmp.is_file():
        D = list(csv.DictReader(open(dmp)))
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        for model in sorted({r["model"] for r in D}):
            dims = sorted({int(r["dim"]) for r in D if r["model"] == model and r["proj"] == "randproj"})
            ys = []
            for d0 in dims:
                p0 = [float(r["ba"]) for r in D if r["model"] == model and r["proj"] == "randproj" and int(r["dim"]) == d0 and r["init"] == "pretrained"]
                r0 = [float(r["ba"]) for r in D if r["model"] == model and r["proj"] == "randproj" and int(r["dim"]) == d0 and r["init"] == "random"]
                ys.append(np.mean(p0) - np.mean(r0))
            ax.plot(dims, ys, marker="o", ms=3, lw=1.2, label=model)
        ax.set_xscale("log", base=2)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel("random-projection width (dims)"); ax.set_ylabel("$\\Delta$BA")
        ax.set_title("Dimension matching: advantage survives projection to 64–2048 dims", fontsize=8)
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "figF_dimension_matching.png", dpi=200)
        fig.savefig(FIG / "figF_dimension_matching.pdf")
        print("wrote figF_dimension_matching.png/pdf")
except Exception as e:  # noqa: BLE001
    print("figures E/F failed:", e)


# ---------------- Figure G: oracle-target stability ----------------
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    osf = REPO / "results/open_llm_suite/raw/oracle_stability.csv"
    osr = REPO / "results/open_llm_suite/raw/oracle_stability_routing.csv"
    if osf.is_file():
        R = list(csv.DictReader(open(osf)))
        prim = [r for r in R if int(r["kind"]) < 3]
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
        ax = axes[0]
        st = np.array([float(r["stability"]) for r in R])
        ax.hist(st, bins=np.linspace(0.3, 1.0, 15), color="#1f77b4")
        ax.set_xlabel("oracle-label stability $s(x)$"); ax.set_ylabel("# windows")
        ax.set_title(f"K=20 future resamples (n={len(R)})", fontsize=8)
        ax = axes[1]
        buckets = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.01)]
        vals, labs = [], []
        for lo, hi in buckets:
            sub = [float(r["single_is_majority"]) for r in prim if lo <= float(r["stability"]) < hi]
            vals.append(np.mean(sub) if sub else np.nan); labs.append(f"{lo:.2f}-{hi:.2f}\n(n={len(sub)})")
        ax.bar(range(len(vals)), vals, color="#2ca02c"); ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=6)
        ax.set_ylabel("P(single-future winner = majority winner)")
        ax.set_title("label flips concentrate in near-ties", fontsize=8)
        ax = axes[2]
        if osr.is_file():
            RR = list(csv.DictReader(open(osr)))
            names = ["family", "single_oracle", "expected_risk"]
            xs = np.arange(len(names))
            for m in sorted({r["model"] for r in RR}):
                ys = []
                for nm in names:
                    p0 = [float(r[nm]) for r in RR if r["model"] == m and r["test"] == "bal3" and r["init"] == "pretrained"]
                    r0 = [float(r[nm]) for r in RR if r["model"] == m and r["test"] == "bal3" and r["init"] == "random"]
                    ys.append(np.mean(p0) - np.mean(r0))
                ax.plot(xs, ys, marker="o", ms=4, label=m)
            ax.set_xticks(xs); ax.set_xticklabels(["family", "single-future\noracle", "expected-risk\noracle"], fontsize=7)
            ax.axhline(0, color="gray", ls="--", lw=1); ax.legend(fontsize=5)
            ax.set_ylabel("$\\Delta$BA (bal3)")
            ax.set_title("supervision interface decides the sign", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "figG_oracle_stability.png", dpi=200)
        fig.savefig(FIG / "figG_oracle_stability.pdf")
        print("wrote figG_oracle_stability.png/pdf")
except Exception as e:  # noqa: BLE001
    print("figure G failed:", e)

# ---------------- Figure B: decision-interface robustness heatmap ----------------
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("family_label_delta", "family-label\n(3 experts)"), ("balanced_n_delta", "family-label\n(bal3n OOD)"),
            ("mlp_router_delta", "MLP64\nrouter"), ("oracle_label_delta", "oracle-label\n(3 experts)")]
    names = [r["model"] for r in rows]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, (k, _) in enumerate(cols):
            v = r.get(k, "")
            M[i, j] = float(v) if v not in ("", None) else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw=dict(width_ratios=[2.0, 1.0]))
    ax = axes[0]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "--" if M[i, j] != M[i, j] else f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Decision-interface sensitivity (ΔBA, pretrained − random)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax2 = axes[1]
    med = [float(r["realworld_median_pct"]) if r.get("realworld_median_pct") not in ("", None) else np.nan for r in rows]
    y = np.arange(len(names))
    ax2.barh(y, med, color="#1f77b4")
    ax2.axvline(0, color="gray", lw=1)
    ax2.set_yticks(y); ax2.set_yticklabels([]); ax2.invert_yaxis()
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax2.set_xlabel("median routed-MSE change vs random (%)", fontsize=7)
    ax2.set_title("Real-world zero-shot (15 datasets)", fontsize=8)
    ax2.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / "figB_robustness.png", dpi=200)
    fig.savefig(FIG / "figB_robustness.pdf")
    print("wrote figB_robustness.png/pdf")
except Exception as e:  # noqa: BLE001
    print("figure B failed:", e)
