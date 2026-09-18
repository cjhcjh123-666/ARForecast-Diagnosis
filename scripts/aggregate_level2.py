"""Aggregate Level-2 (relations + composition) probe shards: paired deltas + CI."""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from temporal_abstraction.statistics import paired_bootstrap, bh_fdr

REL = REPO / "results/temporal_abstraction/relations"
TAB = REPO / "results/temporal_abstraction/tables"
TAB.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    for f in sorted(REL.glob("level2_*.csv")):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        print("no level-2 shards yet"); return
    with open(REL / "level2_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    by = defaultdict(lambda: defaultdict(dict))     # (model, task) -> init -> seed -> score
    fam = {}
    for r in rows:
        by[(r["model"], r["task"])][r["init"]][int(r["seed"])] = float(r["score"])
        fam[r["task"]] = r["family"]
    out = []
    for (m, t), d in sorted(by.items()):
        pre, rnd = d.get("pretrained", {}), d.get("random", {})
        seeds = sorted(set(pre) & set(rnd))
        if not seeds:
            continue
        dl = [pre[s] - rnd[s] for s in seeds]
        st = paired_bootstrap(dl, n=10000, seed=0)
        out.append(dict(model=m, task=t, family=fam[t], n_seeds=len(seeds),
                        pre_mean=round(float(np.mean([pre[s] for s in seeds])), 4),
                        rnd_mean=round(float(np.mean([rnd[s] for s in seeds])), 4),
                        delta=round(st["mean"], 4), ci_lo=round(st["lo"], 4), ci_hi=round(st["hi"], 4),
                        per_seed=";".join(f"s{s}:{pre[s]-rnd[s]:+.3f}" for s in seeds)))
    with open(TAB / "level2_probes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("wrote", TAB / "level2_probes.csv", len(out), "rows")
    print(f"{'task':<28}{'family':<22}{'P':>7}{'R':>7}{'delta':>8}  models>0")
    for t in sorted({r["task"] for r in out}):
        sub = [r for r in out if r["task"] == t]
        p = np.mean([r["pre_mean"] for r in sub]); r_ = np.mean([r["rnd_mean"] for r in sub])
        d_ = np.mean([r["delta"] for r in sub]); pos = sum(1 for r in sub if r["delta"] > 0)
        print(f"{t:<28}{sub[0]['family']:<22}{p:7.3f}{r_:7.3f}{d_:+8.3f}  {pos}/{len(sub)}")


if __name__ == "__main__":
    main()
