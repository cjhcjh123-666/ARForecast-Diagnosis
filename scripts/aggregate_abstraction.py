"""Aggregate probe shards into the Level-1/1b result tables (paired pretrained - random)."""
from __future__ import annotations
import csv, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from temporal_abstraction.statistics import paired_bootstrap, bh_fdr

PRI = REPO / "results/temporal_abstraction/primitives"
TAB = REPO / "results/temporal_abstraction/tables"
TAB.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    for f in sorted(PRI.glob("probes_*.csv")):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        print("no probe shards yet"); return
    with open(PRI / "probes_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"merged {len(rows)} probe rows from {len(set(r['model'] for r in rows))} models")

    # long table with paired deltas: per (model, task, metric, layer)
    key = lambda r: (r["model"], r["task"], r["metric"], r["layer"])
    by = defaultdict(lambda: defaultdict(list))          # key -> init -> [scores]
    for r in rows:
        by[key(r)][r["pretrained_or_random"]].append((int(r["seed"]), float(r["score"])))
    out = []
    pvals, idxs = [], []
    for k, d in by.items():
        pre = dict(d.get("pretrained", [])); rnd = dict(d.get("random", []))
        seeds = sorted(set(pre) & set(rnd))
        if not seeds:
            # handcrafted reference or missing branch: report raw mean only
            for init, vals in d.items():
                out.append(dict(model=k[0], task=k[1], metric=k[2], layer=k[3], init=init, n_seeds=len(vals),
                                mean=round(float(np.mean([v for _, v in vals])), 4), delta="", ci_lo="", ci_hi="",
                                q_value=""))
            continue
        dl = [pre[s] - rnd[s] for s in seeds]
        st = paired_bootstrap(dl, n=10000, seed=0)
        pvals.append(np.nan); idxs.append(len(out))
        out.append(dict(model=k[0], task=k[1], metric=k[2], layer=k[3], init="pretrained_minus_random",
                        n_seeds=len(seeds), mean=round(float(np.mean([pre[s] for s in seeds])), 4),
                        delta=round(st["mean"], 4), ci_lo=round(st["lo"], 4), ci_hi=round(st["hi"], 4), q_value=""))
        out[-1]["pre_mean"] = round(float(np.mean([pre[s] for s in seeds])), 4)
        out[-1]["rnd_mean"] = round(float(np.mean([rnd[s] for s in seeds])), 4)
        out[-1]["per_seed_delta"] = ";".join(f"s{s}:{pre[s]-rnd[s]:+.3f}" for s in seeds)
    FIELDS = ["model","task","metric","layer","init","n_seeds","pre_mean","rnd_mean","mean","delta","ci_lo","ci_hi","q_value","per_seed_delta"]
    with open(TAB / "level1_probes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(out)
    print("wrote", TAB / "level1_probes.csv", len(out), "rows")
    # quick summary
    for k in sorted({(r["model"], r["task"], r["metric"], r["layer"]) for r in out}):
        m = [r for r in out if (r["model"], r["task"], r["metric"], r["layer"]) == k and r["init"] == "pretrained_minus_random"]
        if m:
            print(f"{k[0]:<18} {k[1]:<28} {k[2]:<18} {k[3]:<6} P={m[0]['pre_mean']:.3f} R={m[0]['rnd_mean']:.3f} delta={m[0]['delta']:+.3f}")


if __name__ == "__main__":
    main()
