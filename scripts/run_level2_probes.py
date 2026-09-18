"""Corrected Level-2 relation/composition probes from frozen final-layer caches.

Protocol v2 uses one persisted manifest per seed. The manifest fixes windows, pairs, labels and
splits across processes, models and initializations. Legacy outputs remain under
``results/temporal_abstraction/relations`` and are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.linear_probe import softmax_predict, softmax_regression
from temporal_abstraction.statistics import balanced_accuracy

POOL = REPO / "results/temporal_abstraction/primitives"
REPS = REPO / "results/temporal_abstraction/reps"
ROOT = REPO / "results/temporal_abstraction/corrected_v2"
OUT = ROOT / "relations"
MANIFESTS = ROOT / "manifests"
PROTOCOL_VERSION = "level2_corrected_v2"
SEEN = ["TREND", "PERIODIC", "LOCAL", "TP"]
UNSEEN = ["TL", "TPL"]
SIG = {
    "TREND": (1, 0, 0), "PERIODIC": (0, 1, 0), "LOCAL": (0, 0, 1),
    "TP": (1, 1, 0), "TL": (1, 0, 1), "TPL": (1, 1, 1),
    "REGIME": (1, 1, 0),
}


def stable_seed(*parts) -> int:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def load_pool(seed, pool_dir=POOL):
    source = pool_dir / f"pool_s{seed}.jsonl"
    lat = [json.loads(line) for line in source.read_text().splitlines()]
    idx = {}
    for i, row in enumerate(lat):
        idx.setdefault(row["block"], []).append(i)
    return lat, idx, source


def split_windows(ids, rng, frac=0.6):
    ids = np.asarray(ids, dtype=int)
    ids = ids[rng.permutation(len(ids))]
    n = int(round(frac * len(ids)))
    return ids[:n], ids[n:]


def pairs_r1(lat, idx, rng, n):
    out, pool = [], list(idx["TREND"])
    while len(out) < n:
        a, b = rng.choice(pool, 2, replace=False)
        da, db = abs(lat[a]["slope"]), abs(lat[b]["slope"])
        if abs(da - db) >= 0.03:
            out.append((int(a), int(b), int(da > db)))
    return out


def pairs_r2(lat, idx, rng, n):
    out = []
    while len(out) < n:
        periodic, control = rng.choice(idx["PERIODIC"]), rng.choice(idx["APERIODIC"])
        out.append((int(periodic), int(control), 1) if rng.random() < 0.5
                   else (int(control), int(periodic), 0))
    return out


def pairs_r4(lat, idx, rng, n):
    fams, out = ["TREND", "PERIODIC", "LOCAL"], []
    while len(out) < n:
        if rng.random() < 0.5:
            family = str(rng.choice(fams))
            a, b = rng.choice(idx[family], 2, replace=False)
            out.append((int(a), int(b), 1))
        else:
            f1, f2 = rng.choice(fams, 2, replace=False)
            out.append((int(rng.choice(idx[f1])), int(rng.choice(idx[f2])), 0))
    return out


def pairs_r5(lat, idx, rng, n):
    out, pool = [], list(idx["PERIODIC"])
    while len(out) < n:
        a, b = rng.choice(pool, 2, replace=False)
        out.append((int(a), int(b), int(lat[a]["period"] == lat[b]["period"])))
    return out


def pairs_r6(lat, idx, rng, n):
    out = []
    while len(out) < n:
        anomalous, clean = rng.choice(idx["ANOM"]), rng.choice(idx["ANOM_CLEAN"])
        out.append((int(anomalous), int(clean), 1) if rng.random() < 0.5
                   else (int(clean), int(anomalous), 0))
    return out


def _latent_vector(row):
    family = row["block"]
    if family == "TREND":
        return np.asarray([abs(float(row["slope"])) / 0.18])
    if family == "PERIODIC":
        return np.asarray([float(row["period"]) / 32.0, float(row["amp"]) / 2.0])
    if family == "LOCAL":
        return np.asarray([float(row["phi"]), float(row["sigma"]) / 0.15])
    raise ValueError(family)


def pairs_r7(lat, idx, rng, n):
    """Similarity ordering: which candidate is closer to an anchor in latent-parameter space."""
    fams, out = ["TREND", "PERIODIC", "LOCAL"], []
    while len(out) < n:
        family = str(rng.choice(fams))
        anchor, b, c = rng.choice(idx[family], 3, replace=False)
        va, vb, vc = _latent_vector(lat[anchor]), _latent_vector(lat[b]), _latent_vector(lat[c])
        db, dc = float(np.linalg.norm(va - vb)), float(np.linalg.norm(va - vc))
        if abs(db - dc) < 0.05:
            continue
        if rng.random() < 0.5:
            out.append((int(anchor), int(b), int(c), int(db < dc)))
        else:
            out.append((int(anchor), int(c), int(b), int(dc < db)))
    return out


RELATIONS = [
    ("R1_stronger_trend", pairs_r1),
    ("R2_periodicity_present", pairs_r2),
    ("R4_same_family", pairs_r4),
    ("R5_same_period", pairs_r5),
    ("R6_anomaly_vs_clean", pairs_r6),
    ("R7_similarity_ordering", pairs_r7),
]


def valid_c1_axes(y_test):
    y_test = np.asarray(y_test)
    return [j for j in range(y_test.shape[1]) if len(np.unique(y_test[:, j])) > 1]


def _json_hash(payload) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def build_manifest(lat, idx, seed, n_pairs, pool_sha256):
    relations = {}
    for name, builder in RELATIONS:
        train_pool, test_pool = {}, {}
        for block in sorted(idx):
            train_pool[block], test_pool[block] = split_windows(
                idx[block], np.random.default_rng(stable_seed(name, block, seed))
            )
        relations[name] = {}
        for split, pool in (("train", train_pool), ("test", test_pool)):
            relations[name][split] = builder(
                lat, pool, np.random.default_rng(stable_seed(name, split, seed)), n_pairs // 2
            )

    rng = np.random.default_rng(stable_seed("C1", seed))
    c1_train = np.concatenate([np.asarray(idx[block], dtype=int) for block in SEEN])
    c1_train = c1_train[rng.permutation(len(c1_train))][:int(0.75 * len(c1_train))]
    c1_test = np.concatenate([np.asarray(idx[block], dtype=int) for block in UNSEEN])
    y_test = np.asarray([SIG[lat[i]["block"]] for i in c1_test])

    rng = np.random.default_rng(stable_seed("C3", seed))
    train_rows, test_rows = [], []
    for label, block in enumerate(("TP", "REGIME")):
        train_ids, test_ids = split_windows(idx[block], rng)
        train_rows.extend([[int(i), label] for i in train_ids])
        test_rows.extend([[int(i), label] for i in test_ids])
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "n_pairs": n_pairs,
        "pool_sha256": pool_sha256,
        "invalid_tasks": {
            "R3_higher_volatility": "context standardisation removes absolute volatility scale"
        },
        "relations": relations,
        "c1": {
            "train_ids": c1_train.astype(int).tolist(),
            "test_ids": c1_test.astype(int).tolist(),
            "valid_axes": valid_c1_axes(y_test),
            "excluded_constant_axes": [j for j in range(3) if j not in valid_c1_axes(y_test)],
        },
        "c3": {"train": train_rows, "test": test_rows},
    }
    payload["manifest_sha256"] = _json_hash(payload)
    return payload


def load_or_create_manifest(seed, n_pairs, manifest_dir=MANIFESTS, pool_dir=POOL):
    lat, idx, source = load_pool(seed, pool_dir)
    pool_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"level2_s{seed}.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        expected_hash = payload.pop("manifest_sha256")
        actual_hash = _json_hash(payload)
        payload["manifest_sha256"] = expected_hash
        if expected_hash != actual_hash:
            raise ValueError(f"corrupt manifest: {path}")
        if (payload["protocol_version"] != PROTOCOL_VERSION or
                payload["n_pairs"] != n_pairs or payload["pool_sha256"] != pool_sha):
            raise ValueError(f"manifest protocol/source mismatch: {path}")
        return payload, lat, idx, path
    payload = build_manifest(lat, idx, seed, n_pairs, pool_sha)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return payload, lat, idx, path


def fit_eval(x_train, y_train, x_test, y_test):
    W, mu, sd = softmax_regression(x_train, y_train, n_iter=2000)
    _, pred = softmax_predict(x_test, W, mu, sd)
    return balanced_accuracy(pred, y_test, int(max(y_train.max() + 1, 2)))


def relation_features(H, rows):
    if len(rows[0]) == 3:
        return np.concatenate([H[[r[0] for r in rows]], H[[r[1] for r in rows]]], axis=1)
    if len(rows[0]) == 4:
        return np.concatenate([
            H[[r[0] for r in rows]], H[[r[1] for r in rows]], H[[r[2] for r in rows]]
        ], axis=1)
    raise ValueError("unsupported relation arity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init", choices=("pretrained", "random"))
    parser.add_argument("--n-pairs", type=int, default=1200)
    parser.add_argument("--layer", default="final")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--reps-dir", type=Path, default=REPS)
    parser.add_argument("--pool-dir", type=Path, default=POOL)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFESTS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    manifest, lat, _idx, manifest_path = load_or_create_manifest(
        args.seed, args.n_pairs, args.manifest_dir, args.pool_dir
    )
    print(f"manifest {manifest_path} sha256={manifest['manifest_sha256']}", flush=True)
    if args.manifest_only:
        return
    if not args.model_key or not args.init:
        parser.error("--model-key and --init are required unless --manifest-only is used")

    rep_path = args.reps_dir / f"{args.model_key}_{args.init}_s{args.seed}.npz"
    if not rep_path.is_file():
        raise SystemExit(f"missing {rep_path}")
    z = np.load(rep_path)
    layer_ids = z["layer_ids"].tolist()
    layer_id = layer_ids[-1] if args.layer == "final" else layer_ids[0]
    H = z[f"h_L{layer_id}"].astype(np.float32)
    common = dict(
        model=args.model_key, init=args.init, seed=args.seed,
        protocol_version=PROTOCOL_VERSION, manifest_sha256=manifest["manifest_sha256"],
    )
    rows = []

    for name, _ in RELATIONS:
        train = manifest["relations"][name]["train"]
        test = manifest["relations"][name]["test"]
        y_train = np.asarray([row[-1] for row in train], dtype=int)
        y_test = np.asarray([row[-1] for row in test], dtype=int)
        score = fit_eval(relation_features(H, train), y_train, relation_features(H, test), y_test)
        rows.append(dict(**common, task=name, family="relation", n_train=len(train), n_test=len(test),
                         metric="balanced_accuracy", score=round(score, 4)))
        print(f"[{args.model_key}/{args.init}/s{args.seed}] {name} BA={score:.3f}", flush=True)

    train_ids = np.asarray(manifest["c1"]["train_ids"], dtype=int)
    test_ids = np.asarray(manifest["c1"]["test_ids"], dtype=int)
    y_train = np.asarray([SIG[lat[i]["block"]] for i in train_ids])
    y_test = np.asarray([SIG[lat[i]["block"]] for i in test_ids])
    labels = ["trend_present", "periodic_present", "local_present"]
    predictions = []
    for axis in range(3):
        W, mu, sd = softmax_regression(H[train_ids], y_train[:, axis], n_iter=2000)
        _, pred = softmax_predict(H[test_ids], W, mu, sd)
        predictions.append(pred)
        if axis in manifest["c1"]["valid_axes"]:
            score = balanced_accuracy(pred, y_test[:, axis], 2)
            rows.append(dict(**common, task=f"C1_signature_{labels[axis]}",
                             family="composition_unseen", n_train=len(train_ids), n_test=len(test_ids),
                             metric="balanced_accuracy", score=round(score, 4)))
    pred_signature = np.stack(predictions, axis=1)
    rows.append(dict(**common, task="C1_exact_signature", family="composition_unseen",
                     n_train=len(train_ids), n_test=len(test_ids), metric="exact_match",
                     score=round(float((pred_signature == y_test).all(axis=1).mean()), 4)))

    c3_train, c3_test = manifest["c3"]["train"], manifest["c3"]["test"]
    c3_train_ids = np.asarray([row[0] for row in c3_train], dtype=int)
    c3_test_ids = np.asarray([row[0] for row in c3_test], dtype=int)
    c3_y_train = np.asarray([row[1] for row in c3_train], dtype=int)
    c3_y_test = np.asarray([row[1] for row in c3_test], dtype=int)
    score = fit_eval(H[c3_train_ids], c3_y_train, H[c3_test_ids], c3_y_test)
    rows.append(dict(**common, task="C3_arrangement_tp_vs_regime",
                     family="composition_arrangement", n_train=len(c3_train), n_test=len(c3_test),
                     metric="balanced_accuracy", score=round(score, 4)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"level2_{args.model_key}_{args.init}_s{args.seed}.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
