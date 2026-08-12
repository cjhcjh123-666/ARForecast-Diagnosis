#!/usr/bin/env bash
# Run the controlled pretrained-vs-random Qwen3-8B LoRA forecast sweep for the
# ICASSP paper.  Usage: bash scripts/run_icassp_forecast_sweep.sh <gpu> <job...>
# where each job is a triple: <dataset> <init> <seed>.
set -u

GPU="$1"
shift

while [ "$#" -ge 3 ]; do
  DATASET="$1"
  INIT="$2"
  SEED="$3"
  shift 3
  if [[ "$INIT" == *random* ]]; then
    RAND_FLAG="--random-init"
  else
    RAND_FLAG=""
  fi
  OUT="results/icassp/${DATASET}/${INIT}/seed${SEED}"
  mkdir -p "${OUT}"
  echo "=== [$(date +%H:%M:%S)] gpu=${GPU} ${DATASET}/${INIT}/seed${SEED} ==="
  /public/duyinglong/miniconda3/envs/wavellm/bin/python scripts/run_qwen_experiment.py \
    --device "cuda:${GPU}" --epochs 5 --max-train-windows 256 --max-val-windows 64 \
    --max-test-windows 64 --seed "${SEED}" --dataset "${DATASET}" --output-dir "${OUT}" \
    ${RAND_FLAG}
done
