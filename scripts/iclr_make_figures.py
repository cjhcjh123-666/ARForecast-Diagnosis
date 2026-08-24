"""Generate the ICLR paper figures from the collected results."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from iclr_plot_style import apply_style, save

ROOT = Path("/9950backfile/chenjiahui/ARForecast-Diagnosis")
ICLR = ROOT / "results/iclr"
ICASSP = ROOT / "results/icassp"
FIG = ROOT / "paper/iclr2027/figs"
FIG.mkdir(parents=True, exist_ok=True)

apply_style()

def load(p):
    try: return json.loads(p.read_text())
    except Exception: return None

# ---------------- Fig 1: recognize vs generate across scale ----------------
def fig1():
    models = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-8B"]
    recog = [0.980, 0.979, 0.963]
    parse = [0.983, 0.575, 0.267]
    ratio = [9.8, 8.1, 5.5]
    x = np.arange(len(models)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    ax1.bar(x - w/2, recog, w, label="Recognition (probe)", color="#1f77b4")
    ax1.bar(x + w/2, parse, w, label="Generation (parse rate)", color="#d62728")
    ax1.set_xticks(x); ax1.set_xticklabels(models); ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Rate"); ax1.legend(); ax1.set_title("Frozen: recognize vs. generate")
    ax2.bar(x, ratio, color="#2ca02c")
    ax2.set_xticks(x); ax2.set_xticklabels(models)
    ax2.set_ylabel("Generation MSE / oracle expert")
    ax2.set_title("Generation error (same windows)")
    for i, v in enumerate(ratio): ax2.text(i, v + 0.2, f"{v:.1f}x", ha="center")
    plt.tight_layout(); save(fig, str(FIG / "fig1_recognize_vs_generate.pdf"))
    print("fig1 done")

# ---------------- Fig 2: E4 zero-shot routing ----------------
def fig2():
    methods = ["Hand features", "Random Qwen", "PatchTST", "MLP", "Frozen Qwen (Ours)"]
    acc = [0.128, 0.289, 0.175, 0.300, 0.678]
    err = [0.010, 0.015, 0.040, 0.035, 0.075]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.bar(methods, acc, yerr=err, capsize=4, color=["#999999", "#888888", "#66c2a5", "#fc8d62", "#1f77b4"])
    ax.set_ylabel("Oracle-hit accuracy (zero-shot, OOD)")
    ax.set_ylim(0, 0.85)
    for b, v in zip(bars, acc): ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.3f}", ha="center")
    ax.set_title("Zero-shot expert routing on unseen dynamics")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    save(fig, str(FIG / "fig2_zero_shot_routing.pdf"))
    print("fig2 done")

# ---------------- Fig 3: layerwise ------------------
def fig3():
    pr = load(ICLR / "layerwise/qwen3_0.6b/summary.json")
    rd = load(ICLR / "layerwise/qwen3_0.6b_random/summary.json")
    def curve(d):
        acc = d["per_layer_accuracy"]; ks = sorted(int(k) for k in acc)
        return ks, [acc[str(k)] for k in ks]
    k1, v1 = curve(pr); k2, v2 = curve(rd)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(k1, v1, "-o", label="Pretrained", color="#1f77b4", ms=3)
    ax.plot(k2, v2, "-s", label="Random init", color="#d62728", ms=3)
    ax.axhline(0.2, ls=":", color="gray"); ax.text(1, 0.22, "chance")
    ax.set_xlabel("Transformer layer"); ax.set_ylabel("Probe accuracy")
    ax.set_title("Layer-wise temporal-structure recognition (Qwen3-0.6B)")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    save(fig, str(FIG / "fig3_layerwise.pdf"))
    print("fig3 done")

# ---------------- Fig 4: real-data routing (8B) ----------------
def fig4():
    d = load(ICLR / "multi_dataset_router/qwen3_8b/summary.json")
    names, feat, llm, best = [], [], [], []
    for n, r in d["datasets"].items():
        if "error" in r: continue
        names.append(n); feat.append(r["feature_router"]["mse"]); llm.append(r["probe_pretrained"]["mse"]); best.append(r["best_single_mse"])
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(names))
    ax.bar(x - 0.25, feat, 0.25, label="Hand features", color="#999999")
    ax.bar(x, llm, 0.25, label="Frozen LLM router", color="#1f77b4")
    ax.bar(x + 0.25, best, 0.25, label="Best single expert", color="#cccccc")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("Routing MSE (log)")
    ax.legend(fontsize=8); ax.set_title("Zero-shot routing on 15 real datasets (Qwen3-8B)")
    plt.tight_layout(); save(fig, str(FIG / "fig4_real_data_routing.pdf"))
    print("fig4 done")

# ---------------- Fig 5: asymmetry sweeps ----------------
def fig5():
    d = load(ICLR / "asymmetry/qwen3_0.6b/summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    def by_axis(key, vals):
        out = {}
        for c in d["configs"]:
            if c[key] in vals: out[c[key]] = (c["recognition_acc"], c["gen_mse_on_complete"]/c["oracle_mse"])
        return out
    axes[0].set_title("Context length")
    for k, (r, g) in sorted(by_axis("context_len", [32, 64, 128]).items()):
        axes[0].plot([k, k], [r, g/10], marker="o", label=f"ctx={k}")
    axes[0].set_ylim(0, 1); axes[0].set_xlabel("context"); axes[0].set_ylabel("rate")
    for ax, key, vals, label in [
        (axes[0], "context_len", [32, 64, 128], "recog / (gen ratio/10)"),
        (axes[1], "noise", [0.02, 0.05, 0.1, 0.2], "noise"),
        (axes[2], "horizon", [8, 16, 32], "horizon"),
    ]:
        ax.set_title(f"{label} sweep")
        for k in sorted(vals):
            cs = [c for c in d["configs"] if c[key] == k]
            if not cs: continue
            c = cs[0]; r, g = c["recognition_acc"], c["gen_mse_on_complete"]/c["oracle_mse"]
            ax.scatter(k, r, color="#1f77b4", s=40)
            ax.scatter(k, min(g/10, 1), color="#d62728", s=40)
        ax.set_ylim(0, 1.05)
    axes[0].plot([], [], color="#1f77b4", label="recognition")
    axes[0].plot([], [], color="#d62728", label="generation ratio/10")
    axes[0].legend(fontsize=8)
    plt.tight_layout(); save(fig, str(FIG / "fig5_asymmetry_sweeps.pdf"))
    print("fig5 done")


# ---------------- Fig 6: PCA / t-SNE of representations ----------------
def fig6():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.colors as mcolors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    labels = ["trend", "periodic", "local", "mixture", "regime"]

    cache = ICLR / "probe/qwen3_0.6b/seed7/cache"
    h_pre = np.load(cache / "train_h_pretrained_seed7.npy")
    h_rnd = np.load(cache / "train_h_random_seed7.npy")
    # 90 train windows per kind, ordered by kind
    y = np.repeat(np.arange(5), 90)
    sample = np.random.RandomState(0).choice(len(y), size=min(len(y), 400), replace=False)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    for ax, h, title in [
        (axes[0], h_pre, "Pretrained \u2013 PCA"),
        (axes[1], h_rnd, "Random init \u2013 PCA"),
        (axes[2], h_pre, "Pretrained \u2013 t-SNE"),
        (axes[3], h_rnd, "Random init \u2013 t-SNE"),
    ]:
        pca = PCA(n_components=2).fit_transform(h)
        if "t-SNE" in title:
            proj = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(h)
        else:
            proj = pca
        for k in range(5):
            idx = np.where(y[sample] == k)[0]
            ax.scatter(proj[sample][idx, 0], proj[sample][idx, 1], s=7, c=colors[k],
                       label=labels[k], alpha=0.75, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(length=0)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    save(fig, str(FIG / "fig6_representation_tsne.pdf"))
    print("fig6 done")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("ALL FIGURES DONE")
