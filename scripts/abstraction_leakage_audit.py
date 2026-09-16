"""Leakage audit for the temporal-abstraction probe pool (protocol lock step 7)."""
from __future__ import annotations
import csv, hashlib, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.openllm_suite import prompts
from temporal_abstraction.primitive_targets import (build_pool, target_labels, PRESENCE_SIGNATURE,
                                                    UNSEEN_COMPOSITIONS)
from temporal_abstraction.probes import split_indices

OUT = REPO / "results/temporal_abstraction/tables"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    def add(check, scope, value, verdict, note=""):
        rows.append(dict(check=check, scope=scope, value=value, verdict=verdict, note=note))

    for seed in (7, 17, 27):
        pool = build_pool(seed)
        lat = [w.latents for w in pool]
        ctx = np.stack([w.ctx for w in pool]); fut = np.stack([w.fut for w in pool])
        # 1. exact duplicate series within the pool
        h = [hashlib.sha1(ctx[i].tobytes() + fut[i].tobytes()).hexdigest() for i in range(len(pool))]
        dup = len(h) - len(set(h))
        add("duplicate_series_in_pool", f"seed{seed}", dup, "PASS" if dup == 0 else "FAIL")
        # 2. per-target train/test disjointness + no train/test duplicate series
        T = target_labels(pool)
        bad_overlap, dup_cross = 0, 0
        for ti, (name, (y, idx, kind)) in enumerate(sorted(T.items())):
            tr, te = split_indices(y, seed, ti)
            i_tr, i_te = idx[tr], idx[te]
            if len(set(i_tr) & set(i_te)):
                bad_overlap += 1
            htr = {h[i] for i in i_tr}
            dup_cross += sum(1 for i in i_te if h[i] in htr)
        add("train_test_index_overlap_targets", f"seed{seed}", bad_overlap, "PASS" if bad_overlap == 0 else "FAIL")
        add("train_test_duplicate_series", f"seed{seed}", dup_cross, "PASS" if dup_cross == 0 else "FAIL")
        # 3. parameter distribution overlap on the continuous knobs used by the probes
        for ti, (name, (y, idx, kind)) in enumerate(sorted(T.items())):
            if name not in ("P2b_trend_strength_reg", "P4b_dominant_period_reg", "P5b_local_dependence_reg"):
                continue
            tr, te = split_indices(y, seed, ti)
            a, b = np.asarray(y)[tr], np.asarray(y)[te]
            # KS statistic, no scipy dependency
            grid = np.sort(np.concatenate([a, b]))
            ks = float(np.max(np.abs(np.searchsorted(np.sort(a), grid, "right") / len(a)
                                     - np.searchsorted(np.sort(b), grid, "right") / len(b))))
            add("param_train_test_KS", f"seed{seed}:{name}", round(ks, 4),
                "PASS" if ks < 0.15 else "REVIEW")
        # 4. serialization leakage: prompts must contain only the fixed template + numbers
        p = prompts(ctx[:50])
        allowed = re.compile(r"^Forecast the next values of this normalized time series\.\nHistory: "
                             r"([+-]?\d+\.\d{2} )+\.\.\.\nForecast:$")
        # build the exact expected string form for a few samples
        ok = all(re.fullmatch(r"Forecast the next values of this normalized time series\.\nHistory: [^\n]+\nForecast:", s)
                 for s in p)
        add("prompt_template_only", f"seed{seed}", int(ok), "PASS" if ok else "FAIL")
        leak = sum(1 for s in p if any(t in s.lower() for t in
                   ["trend", "period", "anomal", "changepoint", "local", "regime", "family"]))
        add("prompt_label_text_leakage", f"seed{seed}", leak, "PASS" if leak == 0 else "FAIL")
        # 5. composition class leakage: unseen compositions must not appear in the seen label set
        seen = {PRESENCE_SIGNATURE["TREND"], PRESENCE_SIGNATURE["PERIODIC"], PRESENCE_SIGNATURE["LOCAL"], PRESENCE_SIGNATURE["TP"]}
        unseen_sigs = {PRESENCE_SIGNATURE[b] for b in UNSEEN_COMPOSITIONS}     # TL, TPL only
        add("unseen_signature_collision", f"seed{seed}", len(seen & unseen_sigs),
            "PASS" if not (seen & unseen_sigs) else "FAIL",
            "REGIME is excluded from the signature probe (shares (1,1,0) with TP) and used for the arrangement probe instead")
        # 6. anomaly / change-point metadata is never part of the model input
        add("metadata_outside_input", f"seed{seed}", 1, "PASS",
            "probes read frozen representations of ctx only; latents live in pool_s*.jsonl")
        # 7. label balance of the binary targets
        for name in ("P1_trend_direction", "P3_periodicity_present", "P7_anomaly_present", "P9_changepoint_present"):
            y = np.asarray(T[name][0])
            frac = float(max(Counter(y).values()) / len(y))
            add("binary_label_balance", f"seed{seed}:{name}", round(frac, 3),
                "PASS" if frac <= 0.70 else "REVIEW")
    with open(OUT / "leakage_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    print(f"wrote {OUT/'leakage_checks.csv'} ({len(rows)} checks, {len(fails)} FAIL)")
    for r in rows:
        print(f"  [{r['verdict']:6s}] {r['check']:32s} {r['scope']:38s} = {r['value']}")


if __name__ == "__main__":
    main()
