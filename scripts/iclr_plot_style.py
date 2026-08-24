"""Shared matplotlib style for ICLR figures: Times New Roman with
Liberation-Serif fallback (metric-compatible), academic font sizes.
"""
import matplotlib
import matplotlib.pyplot as plt


def apply_style():
    # Times New Roman preferred; Liberation Serif is the standard Linux
    # metric-compatible substitute (falls back to DejaVu Serif if neither).
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif", "DejaVu Serif Display"],
        "mathtext.fontset": "stix",           # STIX is Times-like for math
        "font.size": 9,                        # base text
        "axes.titlesize": 10,                  # panel titles
        "axes.labelsize": 9,                   # axis labels
        "xtick.labelsize": 8,                  # tick labels
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "axes.grid": False,
        "figure.facecolor": "white",
    })


def save(fig, path: str):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}")
