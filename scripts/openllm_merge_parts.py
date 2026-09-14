"""Merge per-shard CSVs produced by the parallel runs into the final result tables."""
from __future__ import annotations
import csv, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results/open_llm_suite/raw"
GROUPS = {
    "tableB_all": ["model", "init", "seed", "test"],
    "router_capacity_all": ["model", "init", "seed"],
    "dimension_matching": ["model", "init", "seed", "proj", "dim"],
    "pooling_sensitivity": ["model", "init", "seed", "pooling"],
    "oracle_stability_routing": ["model", "init", "seed", "test"],
}
for base, key in GROUPS.items():
    parts = sorted(RAW.glob(base + "_*.csv"))
    rows, seen, fields = [], set(), None
    for f in parts:
        for r in csv.DictReader(open(f)):
            sig = tuple(r[k] for k in key)
            if sig in seen:
                continue
            seen.add(sig); rows.append(r)
            fields = fields or list(r.keys())
    if not rows:
        print("no parts for", base); continue
    out = RAW / (base + ".csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("merged %d rows from %d parts -> %s" % (len(rows), len(parts), out.name))
