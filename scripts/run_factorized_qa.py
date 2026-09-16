"""Factorized QA attribution (CPU): pretrained question branch + {pretrained, random} temporal branch.

Both branches share the question representation, the fusion, the head and the split; only the
temporal representation changes. Reported quantity = balanced accuracy of the linear head on the
held-out split, pretrained vs random temporal branch (paired over items).
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression  # noqa: E402
from temporal_abstraction.statistics import balanced_accuracy  # noqa: E402
from scripts.extract_qa_reps import adapter  # noqa: E402

REPS = REPO / "results/temporal_abstraction/qa/reps"
OUT = REPO / "results/temporal_abstraction/qa"


def fuse(h_q, h_x, proj_dim=256, seed=0):
    """concat(h_q, h_x, |P h_q - P h_x|, P h_q * P h_x) with a FIXED random projection P."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (h_q.shape[1], proj_dim)) / np.sqrt(proj_dim)
    a, b = h_q @ P, h_x @ P
    return np.concatenate([h_q, h_x, np.abs(a - b), a * b], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="irts")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task-types", default="")
    ap.add_argument("--condition", default="full")
    ap.add_argument("--frac", type=float, default=0.6)
    a = ap.parse_args()
    mod = adapter(a.benchmark)
    f_pre = REPS / f"{a.benchmark}_{a.model_key}_pretrained_s{a.seed}.npz"
    f_rnd = REPS / f"{a.benchmark}_{a.model_key}_random_s{a.seed}.npz"
    if not (f_pre.is_file() and f_rnd.is_file()):
        print("missing reps:", f_pre.name, f_rnd.name); return
    z, zr = np.load(f_pre), np.load(f_rnd)
    items = {i["id"]: i for i in mod.load_items()}
    gold = z["gold"]; tts = z["task_type"]
    # map gold labels to a finite answer space per item (protocol: only finite-answer items)
    lab_map, y = {}, []
    keep = []
    for i, g in zip(z["ids"], gold):
        item = items[int(i)]
        fmt = item.get("answer_format", "multiple_choice_abcd")
        alpha = mod.LABELS.get(fmt, ["A", "B", "C", "D"])
        if str(g).strip().upper() not in alpha:
            continue
        keep.append(int(np.where(z["ids"] == i)[0][0]))
        y.append(alpha.index(str(g).strip().upper()))
    if a.task_types:
        want = set(a.task_types.split(","))
        keep = [k for k in keep if tts[k] in want]
        y = [y[j] for j, k in enumerate(keep)]
    keep = np.asarray(keep); y = np.asarray(y)
    n_classes = int(y.max() + 1)

    rng = np.random.default_rng(10000 + a.seed)
    tr, te = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0].copy(); rng.shuffle(idx)
        n_tr = int(round(a.frac * len(idx))); tr += idx[:n_tr].tolist(); te += idx[n_tr:].tolist()
    tr = np.asarray(sorted(tr)); te = np.asarray(sorted(te))

    rows = []
    for branch, src in (("pretrained_temporal", z), ("random_temporal", zr)):
        hq = src["h_q"][keep]; hx = src[f"h_x_{a.condition}"][keep]
        X = fuse(hq, hx)
        W, mu, sd = softmax_regression(X[tr], y[tr], n_iter=2000)
        _, pred = softmax_predict(X[te], W, mu, sd)
        ba = balanced_accuracy(pred, y[te], n_classes)
        rows.append(dict(benchmark=a.benchmark, model=a.model_key, condition=a.condition,
                         branch=branch, n_classes=n_classes, n_train=len(tr), n_test=len(te),
                         balanced_accuracy=round(ba, 4)))
        print(rows[-1], flush=True)
    out = OUT / f"factorized_{a.benchmark}_{a.model_key}_s{a.seed}_{a.condition}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out.name, "| delta(pretrained-random) = %.4f" % (rows[0]["balanced_accuracy"] - rows[1]["balanced_accuracy"]))


if __name__ == "__main__":
    main()
