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

ROOT = REPO / "results/temporal_abstraction/corrected_v2"
REPS = ROOT / "qa/reps"
OUT = ROOT / "qa"
FORMAT_LABELS = {
    "multiple_choice_abcd": ["A", "B", "C", "D"],
    "multiple_choice_abc": ["A", "B", "C"],
    "multiple_choice_ab": ["A", "B"],
    "true_false": ["T", "F"],
}


def fuse(h_q, h_x, proj_dim=256, seed=0):
    """concat(h_q, h_x, |P h_q - P h_x|, P h_q * P h_x) with a FIXED random projection P."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (h_q.shape[1], proj_dim)) / np.sqrt(proj_dim)
    a, b = h_q @ P, h_x @ P
    return np.concatenate([h_q, h_x, np.abs(a - b), a * b], axis=1)


def select_finite_items(ids, gold, task_type, items_by_id, wanted=None, labels=None):
    """Return source row indices and labels together, preserving alignment after filtering."""
    labels = labels or FORMAT_LABELS
    wanted = set(wanted or [])
    keep, y = [], []
    for source_index, (item_id, answer, task) in enumerate(zip(ids, gold, task_type)):
        if wanted and str(task) not in wanted:
            continue
        item = items_by_id[int(item_id)]
        alphabet = labels.get(item.get("answer_format", "multiple_choice_abcd"), [])
        answer = str(answer).strip().upper()
        if answer not in alphabet:
            continue
        keep.append(source_index)
        y.append(alphabet.index(answer))
    return np.asarray(keep, dtype=int), np.asarray(y, dtype=int)


def validate_paired_caches(pre, random):
    for key in ("ids", "gold", "task_type"):
        if key not in pre or key not in random or not np.array_equal(pre[key], random[key]):
            raise ValueError(f"pretrained/random QA caches are misaligned for {key}")


def factorized_inputs(pretrained_cache, temporal_cache, condition, keep):
    """Use the pretrained question branch regardless of temporal-branch initialization."""
    hq = pretrained_cache["h_q"][keep]
    # With no temporal evidence, keep the nuisance branch fixed too.
    source = pretrained_cache if condition == "question_only" else temporal_cache
    hx = source[f"h_x_{condition}"][keep]
    return hq, hx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="irts")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task-types", default="")
    ap.add_argument("--condition", default="full")
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--reps-dir", type=Path, default=REPS)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    mod = adapter(a.benchmark)
    f_pre = a.reps_dir / f"{a.benchmark}_{a.model_key}_pretrained_s{a.seed}.npz"
    f_rnd = a.reps_dir / f"{a.benchmark}_{a.model_key}_random_s{a.seed}.npz"
    if not (f_pre.is_file() and f_rnd.is_file()):
        print("missing reps:", f_pre.name, f_rnd.name); return
    z, zr = np.load(f_pre), np.load(f_rnd)
    validate_paired_caches(z, zr)
    items = {i["id"]: i for i in mod.load_items()}
    want = set(filter(None, a.task_types.split(",")))
    keep, y = select_finite_items(
        z["ids"], z["gold"], z["task_type"], items, want, mod.LABELS
    )
    if not len(keep):
        raise ValueError("no finite-answer items remain after task filtering")
    n_classes = int(y.max() + 1)

    rng = np.random.default_rng(10000 + a.seed)
    tr, te = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0].copy(); rng.shuffle(idx)
        n_tr = int(round(a.frac * len(idx))); tr += idx[:n_tr].tolist(); te += idx[n_tr:].tolist()
    tr = np.asarray(sorted(tr)); te = np.asarray(sorted(te))

    branch_features = []
    for branch, src in (("pretrained_temporal", z), ("random_temporal", zr)):
        hq, hx = factorized_inputs(z, src, a.condition, keep)
        branch_features.append((branch, fuse(hq, hx, seed=a.seed)))
    # One shared readout is fit on both temporal branches, then evaluated separately.
    train_x = np.concatenate([X[tr] for _, X in branch_features])
    train_y = np.concatenate([y[tr] for _ in branch_features])
    W, mu, sd = softmax_regression(train_x, train_y, n_iter=2000)
    rows = []
    for branch, X in branch_features:
        _, pred = softmax_predict(X[te], W, mu, sd)
        ba = balanced_accuracy(pred, y[te], n_classes)
        rows.append(dict(benchmark=a.benchmark, model=a.model_key, condition=a.condition,
                         branch=branch, n_classes=n_classes, n_train=len(tr), n_test=len(te),
                         balanced_accuracy=round(ba, 4), protocol_version="corrected_v2_shared_head"))
        print(rows[-1], flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"factorized_{a.benchmark}_{a.model_key}_s{a.seed}_{a.condition}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out.name, "| delta(pretrained-random) = %.4f" % (rows[0]["balanced_accuracy"] - rows[1]["balanced_accuracy"]))


if __name__ == "__main__":
    main()
