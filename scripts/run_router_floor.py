"""Floor experiment: can a cheap feature + softmax router pick the right expert?

This is the decisive control for the LLM-as-Temporal-Router paper.  Before any
LLM is involved we measure (a) how well hand-crafted statistics predict the
oracle expert, and (b) the downstream forecast cost of the cheap router.  If
the cheap floor already reaches ~100% oracle routing accuracy, the paper must
pivot to a zero-shot / unseen-dynamics story; this script produces exactly the
numbers needed to make that call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.features import FEATURE_NAMES, temporal_features
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import CrossChannelExpert, LocalExpert, PeriodicExpert, TrendExpert


def univariate_experts() -> list:
    return [TrendExpert(), PeriodicExpert(), LocalExpert()]


def expert_matrix_univariate(contexts: np.ndarray, futures: np.ndarray) -> np.ndarray:
    """Return per-expert MSE [N, E] for univariate windows."""
    experts = univariate_experts()
    errors = np.zeros((len(contexts), len(experts)), dtype=np.float64)
    for j, expert in enumerate(experts):
        for i in range(len(contexts)):
            pred = expert.predict(contexts[i], futures.shape[1])
            errors[i, j] = float(np.mean((pred - futures[i]) ** 2))
    return errors


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def run_synthetic(args: argparse.Namespace) -> dict:
    kinds = ["trend", "periodic", "local", "mixture", "regime"]
    train_c, train_f, train_kind = build_labeled_windows(
        kinds, args.context_len, args.horizon, args.n_train_per_kind, args.seed
    )
    test_c, test_f, test_kind = build_labeled_windows(
        kinds, args.context_len, args.horizon, args.n_test_per_kind, args.seed + 1000
    )
    horizon = args.horizon

    train_feats = np.stack([temporal_features(x) for x in train_c])
    test_feats = np.stack([temporal_features(x) for x in test_c])
    train_err = expert_matrix_univariate(train_c, train_f)
    test_err = expert_matrix_univariate(test_c, test_f)

    train_oracle = train_err.argmin(axis=1)
    test_oracle = test_err.argmin(axis=1)

    weights, mean, std = softmax_regression(train_feats, train_oracle)
    _, train_pred = softmax_predict(train_feats, weights, mean, std)
    test_prob, test_pred = softmax_predict(test_feats, weights, mean, std)

    # ---- routing accuracy ----
    oracle_acc = float(np.mean(test_pred == test_oracle))
    intended = np.isin(test_kind, [0, 1, 2])
    intent_acc = float(np.mean(test_pred[intended] == test_kind[intended]))
    per_kind = {}
    for k, name in enumerate(kinds):
        mask = test_kind == k
        per_kind[name] = {
            "windows": int(mask.sum()),
            "oracle_acc": float(np.mean(test_pred[mask] == test_oracle[mask])),
            "oracle_dist": [
                int((test_oracle[mask] == j).sum()) for j in range(3)
            ],
        }

    # ---- downstream forecast cost ----
    best_single_idx = int(train_err.mean(axis=0).argmin())
    majority_idx = int(np.bincount(test_oracle).argmax())
    oracle_mse = float(np.mean(test_err.min(axis=1)))
    router_mse = float(np.mean(test_err[np.arange(len(test_err)), test_pred]))
    best_single_mse = float(np.mean(test_err[:, best_single_idx]))
    majority_mse = float(np.mean(test_err[:, majority_idx]))
    uniform_mse = float(np.mean(test_err.mean(axis=1)))

    # perfect-intent upper bound (only over the 3 intended kinds)
    intended_rows = np.where(intended)[0]
    intent_mse = float(
        np.mean(test_err[intended_rows, test_kind[intended_rows]])
    )

    # ---- zero-shot check: train on the 3 clean kinds, route mixture/regime ----
    clean = train_kind < 3
    zs_weights, zs_mean, zs_std = softmax_regression(
        train_feats[clean], train_oracle[clean]
    )
    hard = test_kind >= 3
    zs_prob, zs_pred = softmax_predict(test_feats[hard], zs_weights, zs_mean, zs_std)
    zs_oracle_acc = float(np.mean(zs_pred == test_oracle[hard]))
    zs_mse = float(np.mean(test_err[hard, zs_pred]))

    summary = {
        "seed": args.seed,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "n_train_per_kind": args.n_train_per_kind,
        "n_test_per_kind": args.n_test_per_kind,
        "feature_names": FEATURE_NAMES,
        "oracle_routing_accuracy": oracle_acc,
        "intent_routing_accuracy": intent_acc,
        "per_kind": per_kind,
        "best_single_expert": {
            "name": univariate_experts()[best_single_idx].name,
            "index": best_single_idx,
            "mse": best_single_mse,
        },
        "majority_expert_mse": majority_mse,
        "uniform_ensemble_mse": uniform_mse,
        "oracle_mse": oracle_mse,
        "router_mse": router_mse,
        "intent_upper_bound_mse": intent_mse,
        "regret_vs_oracle_pct": 100.0 * (router_mse / oracle_mse - 1.0),
        "zero_shot": {
            "on": ["mixture", "regime"],
            "oracle_routing_accuracy": zs_oracle_acc,
            "mse": zs_mse,
        },
    }
    return summary


def print_synthetic(summary: dict) -> None:
    print("\n================ ROUTER FLOOR (SYNTHETIC) ================")
    print(f"seed={summary['seed']} context={summary['context_len']} "
          f"horizon={summary['horizon']}")
    print(f"oracle routing accuracy : {summary['oracle_routing_accuracy']:.4f}")
    print(f"intent routing accuracy : {summary['intent_routing_accuracy']:.4f}")
    print("per-kind oracle acc:")
    for name, row in summary["per_kind"].items():
        print(f"  {name:10s} n={row['windows']:4d} acc={row['oracle_acc']:.4f} "
              f"oracle_dist={row['oracle_dist']}")
    print("\nforecast MSE (downstream):")
    print(f"  oracle (upper bound)   : {summary['oracle_mse']:.5f}")
    print(f"  feature router         : {summary['router_mse']:.5f} "
          f"(regret {summary['regret_vs_oracle_pct']:.2f}%)")
    print(f"  best single expert     : {summary['best_single_expert']['mse']:.5f} "
          f"({summary['best_single_expert']['name']})")
    print(f"  majority expert        : {summary['majority_expert_mse']:.5f}")
    print(f"  uniform ensemble       : {summary['uniform_ensemble_mse']:.5f}")
    print(f"  perfect-intent (3 cls) : {summary['intent_upper_bound_mse']:.5f}")
    zs = summary["zero_shot"]
    print(f"\nzero-shot -> mixture/regime: acc_vs_oracle={zs['oracle_routing_accuracy']:.4f} "
          f"mse={zs['mse']:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-train-per-kind", type=int, default=200)
    parser.add_argument("--n-test-per-kind", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/router_floor"))
    args = parser.parse_args()

    summary = run_synthetic(args)
    print_synthetic(summary)
    out = args.output_dir / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nsaved -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
