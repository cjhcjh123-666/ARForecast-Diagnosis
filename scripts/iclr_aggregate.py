"""ICLR: aggregate all results into a single markdown/JSON report.

Reads:
  - results/iclr/probe/<model>/seed*/summary.json  (recognition, gen, router)
  - results/iclr/router_from_cache/<model>/seed*/summary.json (router fallback)
  - results/iclr/multi_dataset_router/<model>/summary.json (15 datasets)
  - results/iclr/learned_router/seed*/summary.json (non-LLM baselines)
and writes results/iclr/AGGREGATE.md + AGGREGATE.json with means/stds across
seeds (statistical rigor).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "results" / "iclr"


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def main() -> None:
    lines = ["# ICLR 2027 Aggregate Results", ""]
    report = {}

    # ---- probe scale table ----
    lines.append("## Model scale: recognition / generation / routing (mean±std over seeds)")
    lines.append("")
    lines.append("| model | recog(pretr) | recog(random) | feat floor | gen parse | gen mse/oracle | router(pretr) | router(random) | router(feat) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    scale = {}
    for model in ["distilgpt2", "gpt2", "qwen3_0.6b", "qwen3_1.7b", "qwen3_8b"]:
        recog_p, recog_r, floor, parse, mse_ratio, rt_p, rt_r, rt_f = [], [], [], [], [], [], [], []
        for seed in [7, 17, 27, 37, 47]:
            # 8B probe data lives in the ICASSP result dir; others in ICLR dir
            probe_path = ROOT / "probe" / model / f"seed{seed}" / "summary.json"
            if model == "qwen3_8b":
                probe_path = REPO_ROOT / "results" / "icassp" / (
                    "probe" if seed == 7 else f"probe_seed{seed}"
                ) / "summary.json"
            d = load_json(probe_path)
            if not d:
                continue
            recog_p.append(d["probe"]["pretrained"]["accuracy"])
            recog_r.append(d["probe"]["random"]["accuracy"])
            floor.append(d["feature_floor"]["accuracy"])
            pg = d.get("paired_recognize_vs_generate")
            if pg:
                parse.append(pg["parse_rate"])
                if pg.get("mse_on_complete") and pg.get("oracle_mse"):
                    mse_ratio.append(pg["mse_on_complete"] / pg["oracle_mse"])
            z = d.get("zero_shot_router")
            if z:
                rt_p.append(z["probe_pretrained"]["routing_acc_vs_oracle"])
                rt_r.append(z["probe_random"]["routing_acc_vs_oracle"])
                rt_f.append(z["feature"]["routing_acc_vs_oracle"])
        # fallback router from cache if probe lacked it
        if not rt_p:
            cache_roots = [ROOT / "router_from_cache" / model]
            if model == "qwen3_8b":
                cache_roots = []
            for seed in [7, 17, 27, 37, 47]:
                for root in cache_roots:
                    d = load_json(root / f"seed{seed}" / "summary.json")
                    if d:
                        rt_p.append(d["probe_pretrained"]["acc"])
                        rt_r.append(d["probe_random"]["acc"])
                        rt_f.append(d["feature"]["acc"])
                        break
        scale[model] = {
            "recog_pretrained": recog_p, "recog_random": recog_r, "floor": floor,
            "parse": parse, "mse_ratio": mse_ratio,
            "router_pretrained": rt_p, "router_random": rt_r, "router_feature": rt_f,
        }
        lines.append(
            f"| {model} | {summarize(recog_p)} | {summarize(recog_r)} | {summarize(floor)} "
            f"| {summarize(parse)} | {summarize(mse_ratio)} | {summarize(rt_p)} "
            f"| {summarize(rt_r)} | {summarize(rt_f)} |"
        )
    lines.append("")
    report["scale"] = scale

    # ---- learned router baselines ----
    lines.append("## Non-LLM learned router baselines (zero-shot mixture/regime)")
    lines.append("")
    lines.append("| seed | method | in-domain acc | OOD acc vs oracle | OOD MSE |")
    lines.append("|---|---|---|---|---|")
    learned = {}
    for seed in [7, 17, 27, 37, 47]:
        d = load_json(ROOT / "learned_router" / f"seed{seed}" / "summary.json")
        if not d:
            continue
        learned[str(seed)] = d["baselines"]
        for meth in ["patchtst", "mlp", "feature"]:
            b = d["baselines"].get(meth)
            if b:
                ida = b.get("in_domain_acc")
                ida_s = f"{ida:.3f}" if isinstance(ida, (int, float)) else "n/a"
                lines.append(f"| {seed} | {meth} | {ida_s} | {b['acc_vs_oracle']:.3f} | {b['mse']:.3f} |")
    lines.append("")
    report["learned_router"] = learned

    # ---- multi-dataset router ----
    lines.append("## Real-data zero-shot routing (15 datasets, seed 7)")
    lines.append("")
    lines.append("| dataset | oracle | best-single | feat acc/mse | LLM acc/mse | rand acc/mse |")
    lines.append("|---|---|---|---|---|---|")
    multi = {}
    for model in ["qwen3_0.6b", "qwen3_8b"]:
        d = load_json(ROOT / "multi_dataset_router" / model / "summary.json")
        if not d:
            continue
        lines.append(f"### {model}")
        lines.append("| dataset | oracle | best-single | feat acc/mse | LLM acc/mse | rand acc/mse |")
        lines.append("|---|---|---|---|---|---|")
        multi[model] = d["datasets"]
        for name, row in d["datasets"].items():
            if "error" in row:
                lines.append(f"| {name} | ERROR | | | | |")
                continue
            f, p, r = row["feature_router"], row["probe_pretrained"], row["probe_random"]
            lines.append(
                f"| {name} | {row['oracle_mse']:.3f} | {row['best_single_name']}/{row['best_single_mse']:.3f} "
                f"| {f['acc_vs_oracle']:.3f}/{f['mse']:.3f} | {p['acc_vs_oracle']:.3f}/{p['mse']:.3f} "
                f"| {r['acc_vs_oracle']:.3f}/{r['mse']:.3f} |"
            )
        lines.append("")
    report["multi_dataset"] = multi

    out_md = ROOT / "AGGREGATE.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "AGGREGATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={out_md}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
