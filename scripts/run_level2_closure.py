"""Resume corrected Level-2 coverage with bounded CPU concurrency and per-run logs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/temporal_abstraction/corrected_v2/relations"
LOGS = REPO / "logs/abstraction/corrected_v2_level2"
MODELS = [
    "qwen3_8b_base", "llama31_8b", "llama32_3b", "gemma2_9b", "gemma2_2b",
    "mistral_7b_v03", "deepseek_llm_7b", "olmo2_7b", "olmo2_13b",
    "deepseek_v2_lite",
]


def run_one(model: str, seed: int, init: str, blas_threads: int) -> tuple[str, int]:
    tag = f"{model}_{init}_s{seed}"
    out = OUT / f"level2_{tag}.csv"
    if out.is_file():
        return tag, 0
    env = dict(os.environ)
    value = str(blas_threads)
    env.update(OMP_NUM_THREADS=value, MKL_NUM_THREADS=value, OPENBLAS_NUM_THREADS=value)
    command = [
        sys.executable, "-u", "scripts/run_level2_probes.py",
        "--model-key", model, "--seed", str(seed), "--init", init,
    ]
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{tag}.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    return tag, result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=8)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--inits", default="pretrained,random")
    args = parser.parse_args()
    models = [x for x in args.models.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    inits = [x for x in args.inits.split(",") if x]
    jobs = [(model, seed, init) for model in models for seed in seeds for init in inits]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, model, seed, init, args.blas_threads): (model, seed, init)
            for model, seed, init in jobs
        }
        for future in as_completed(futures):
            tag, code = future.result()
            print(f"[{tag}] {'done' if code == 0 else f'FAILED({code})'}", flush=True)
            if code:
                failures.append(tag)
    if failures:
        raise SystemExit("failed runs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
