"""Aggregate Qwen experiment artifacts across seeds and initializations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_qwen_results import summarize_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [summarize_run(root) for root in args.roots]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["initialization"])].append(row)
    aggregate = {}
    for initialization, members in groups.items():
        rmse = np.asarray([row["rmse"] for row in members if row["rmse"] is not None], dtype=float)
        amplitude = np.asarray(
            [row["spectral"]["amplitude_mae"] for row in members if row["spectral"] is not None],
            dtype=float,
        )
        aggregate[initialization] = {
            "runs": len(members),
            "parse_success_rate_mean": float(np.mean([row["parse_success_rate"] for row in members])),
            "rmse_mean": None if not len(rmse) else float(np.mean(rmse)),
            "rmse_std": None if len(rmse) < 2 else float(np.std(rmse, ddof=1)),
            "spectral_amplitude_mae_mean": None if not len(amplitude) else float(np.mean(amplitude)),
            "members": members,
        }
    payload = {"runs": rows, "aggregate": aggregate}
    serialized = json.dumps(payload, indent=2)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
