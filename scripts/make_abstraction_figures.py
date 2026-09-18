"""Analysis figures for the temporal-abstraction study (all numbers read from CSVs)."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "results/temporal_abstraction/tables"
FIG = REPO / "results/temporal_abstraction/figures"; FIG.mkdir(parents=True, exist_ok=True)
QA = REPO / "results/temporal_abstraction/qa"

PRETTY = {"qwen3_8b_base": "Qwen3-8B", "llama31_8b": "Llama-3.1-8B", "llama32_3b": "Llama-3.2-3B",
          "gemma2_9b": "Gemma-2-9B", "gemma2_2b": "Gemma-2-2B", "mistral_7b_v03": "Mistral-7B",
          "deepseek_llm_7b": "DeepSeek-LLM-7B", "olmo2_7b": "OLMo-2-7B", "olmo2_13b": "OLMo-2-13B",
          "deepseek_v2_lite": "DeepSeek-V2-Lite"}
ORDER = ["P1_trend_direction", "P3_periodicity_present", "P9_changepoint_present", "P2_trend_strength",
         "P6_noise_level", "P8_anomaly_location", "P10_changepoint_location", "P5_local_dependence",
         "P7_anomaly_present", "P4_dominant_period"]
LABEL = {"P1_trend_direction": "trend dir", "P3_periodicity_present": "periodicity", "P9_changepoint_present": "changepoint",
         "P2_trend_strength": "trend strength", "P6_noise_level": "noise level", "P8_anomaly_location": "anomaly loc",
         "P10_changepoint_location": "changepoint loc", "P5_local_dependence": "local dependence",
         "P7_anomaly_present": "anomaly presence", "P4_dominant_period": "dominant period"}


def load(name):
    f = TAB / name
    return list(csv.DictReader(open(f))) if f.is_file() else []


def fig_a():
    rows = [r for r in load("level1_probes.csv") if r["init"] == "pretrained_minus_random"
            and r["metric"] == "balanced_accuracy" and r["layer"] == "final"]
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    M = np.full((len(models), len(ORDER)), np.nan)
    for r in rows:
        if r["task"] in ORDER:
            M[models.index(r["model"]), ORDER.index(r["task"])] = float(r["delta"])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.1, vmax=0.45, aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LABEL[t] for t in ORDER], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([PRETTY.get(m, m) for m in models], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Figure A — Temporal abstraction map: pretrained − random probe gain (final layer)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(FIG / "figA_abstraction_map.png", dpi=200); fig.savefig(FIG / "figA_abstraction_map.pdf")
    print("wrote figA_abstraction_map")


def fig_b():
    rows = [r for r in load("level1_probes.csv") if r["init"] == "pretrained_minus_random"
            and r["metric"] == "balanced_accuracy"]
    if not rows:
        return
    order = ["L1", "p25", "p50", "p75", "final"]
    tasks = ["P4_dominant_period", "P7_anomaly_present", "P5_local_dependence", "P6_noise_level", "P1_trend_direction"]
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for t in tasks:
        ys = []
        for L in order:
            v = [float(r["delta"]) for r in rows if r["task"] == t and r["layer"] == L]
            ys.append(np.mean(v) if v else np.nan)
        ax.plot(range(len(order)), ys, marker="o", ms=4, label=LABEL[t])
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(["early\n(L1)", "25%", "50%", "75%", "final"], fontsize=7)
    ax.set_ylabel("$\\Delta$ balanced accuracy"); ax.set_xlabel("relative layer depth")
    ax.set_title("Figure B — Where the pretrained advantage emerges (10-model mean)", fontsize=8)
    ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "figB_layerwise.png", dpi=200); fig.savefig(FIG / "figB_layerwise.pdf")
    print("wrote figB_layerwise")


def fig_d():
    rows = []
    for f in sorted(QA.glob("native_irts_*_s7.csv")):
        for r in csv.DictReader(open(f)):
            rows.append(r)
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    conds = ["full", "question_only", "shuffled_ts", "reversed_ts"]
    M = np.zeros((len(models), len(conds)))
    for i, m in enumerate(models):
        for j, c in enumerate(conds):
            sub = [r for r in rows if r["model"] == m and r["condition"] == c]
            M[i, j] = np.mean([int(r["correct"]) for r in sub]) if sub else np.nan
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=0.45, aspect="auto")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(["Full", "Question only", "Shuffled TS", "Reversed TS"], fontsize=7)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([PRETTY.get(m, m) for m in models], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    ax.axhline(-0.5, color="k"); ax.set_title("Figure D — IRTS temporal_relationship: does the series matter?   (chance 0.25)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(FIG / "figD_temporal_evidence.png", dpi=200); fig.savefig(FIG / "figD_temporal_evidence.pdf")
    print("wrote figD_temporal_evidence")


if __name__ == "__main__":
    fig_a(); fig_b(); fig_d()
