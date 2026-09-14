# Open-LLM Suite — Reproducibility Notes (2026-09-11)

- Git: repo `ARForecast-Diagnosis`, branch `main`; audit-stage commit `fd27169` (plus new commits from this round).
- Env: `/public/duyinglong/miniconda3/envs/wavellm` (torch 2.4.1+cu121, transformers 4.52.4, huggingface_hub 0.36.2, hf_transfer 0.1.9, numpy).
- Downloads: `HF_ENDPOINT=https://hf-mirror.com`, `no_proxy=*`, `HF_HUB_DISABLE_XET=1`, `scripts/openllm_download_one.py`;
  cache `/9950backfile/chenjiahui/hf_cache/hub`; revisions pinned via `HfApi.model_info(...).sha`
  (DeepSeek-LLM-7B `7683fea62db869066ddaff6a41d032262c490d4f`, DeepSeek-V2-Lite `604d5664dddd88a0433dbae533b7fe9472482de0`,
  Mistral-7B-v0.3 `caa1feb0e54d415e2df31207e5f4e273e33509b1`, OLMo-2-7B `7df9a82518afdecae4e8c026b27adccc8c1f0032`,
  OLMo-2-13B `3fefddc1bf18a30e1d9b91000271630718f2aa8b`).
- Protocol frozen from the existing paper pipeline: `data/dynamics.py`, `scripts/run_frozen_probe.py`
  (`_format_values`, `_prompts`, parser `(?<![0-9])[+-]?\d+\.\d{2}`), `analysis/linear_probe.py`, `models/experts.py`.
- New runner: `scripts/openllm_suite.py` (A clean/shuffle, B linear+MLP readout, C family-label routing,
  C2 oracle-label routing, R MLP64 router, pooling sensitivity, native greedy generation).
  Random control = from-config architecture-native init (same tokenizer/serialization/inference).
- API: `scripts/discover_api_models.py` (listing only, no paid calls); frozen list `configs/api_models_frozen.yaml`;
  paid runs require `RUN_PAID_API=1` plus `API_MAX_REQUESTS` or `API_BUDGET_USD`.

## Update 2026-09-14 — completed suite, exact commands

Environment (unchanged): `/public/duyinglong/miniconda3/envs/wavellm/bin/python`
(torch 2.4.1+cu121, transformers 4.52.4), 8 × A800-80GB.

Model paths actually used (local dirs, revision-pinned):

| model key | local path |
|---|---|
| deepseek_llm_7b | `hf_cache/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/7683fea62db869066ddaff6a41d032262c490d4f` |
| deepseek_v2_lite | `hf_cache/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0` |
| mistral_7b_v03 | `hf_cache/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1` |
| olmo2_7b / olmo2_13b | `hf_cache/hub/models--allenai--OLMo-2-1124-{7B,13B}/snapshots/{7df9a82…,3fefddc…}` |
| gemma2_9b / gemma2_2b | `model_cache_modelscope/LLM-Research/gemma-2-{9b,2b}` (ModelScope mirror, provenance JSONs alongside the results) |
| llama31_8b / llama32_3b | `model_cache_modelscope/LLM-Research/{Meta-Llama-3___1-8B,Llama-3___2-3B}` |

Representation + attribution (per model × pretrained/random, seeds 7/17/27):

```bash
$PY -u scripts/openllm_suite.py --model-key <key> --model-id <hf id> --path <local dir> \
    --device cuda:<i> --seeds 7,17,27 --tasks A,B,C,C2,R --inits pretrained,random
```

Real-world zero-shot routing (router trained only on the 270 synthetic clean primitives):

```bash
$PY -u scripts/openllm_realworld_features.py --model-key <key> ...   # 15 datasets × 2 inits
$PY -u scripts/openllm_realworld_analysis.py                          # paired bootstrap + BH-FDR
```

Native greedy generation (historical prompt/parser, 40 class-stratified windows):

```bash
$PY -u scripts/openllm_native.py --model-key <key> --model-id <hf id> --path <local dir> \
    --device cuda:<i> --inits pretrained,random --windows 40
```

Report / tables / figures (all numbers re-derived from CSVs, nothing typed by hand):

```bash
$PY scripts/openllm_make_report.py     # docs/open_llm_all_results.md
$PY scripts/openllm_figures.py         # Figure A/C/D + tables/robustness_matrix.csv
$PY scripts/openllm_export.py          # all_metrics_long.csv, all_metrics_wide.csv, paper_tables.tex
```

Decision rules kept throughout: BF16 only (no quantized hidden states), random control = from-config
architecture-native init with the *same* tokenizer / serialization / attention implementation / inference code,
router trained only on clean primitives, OOD never used for training or tuning, all negative results retained.

## Update 2026-09-14 (round 2) — robustness, scale, seeds, oracle stability

All analyses below are CPU-only and read the frozen feature dumps (no new LM forward passes
except the feature extractions listed in the first table above).

| script | what it produces |
|---|---|
| `scripts/openllm_features_full.py --extra-pool mean` | clean750 + bal3 + bal3n features (last-token **and** mean pooled) per (model, init, seed) |
| `scripts/openllm_remaining_experiments.py` | `tableB_all.csv` (3 vs 5 experts), `router_capacity_all.csv` (linear vs MLP64), `dimension_matching.csv` (random projection + PCA), `pooling_sensitivity.csv`. Env vars `OPENLLM_MODELS / OPENLLM_SEEDS / OPENLLM_INITS / OPENLLM_OUT_SUFFIX` shard it per (model, seed, init). |
| `scripts/openllm_scale_and_seeds.py` | `scale_sweep.csv` (Qwen3 0.6B/1.7B/8B) and `ten_seed_headline.csv` (Qwen3-8B, seeds 7..97) |
| `scripts/openllm_oracle_stability.py` | `oracle_stability.csv` (K=20 future resamples per window: stability, margin, majority label) and `oracle_stability_routing.csv` (family vs single-future vs expected-risk supervision) |
| `scripts/openllm_merge_parts.py` | merges all per-shard CSVs into the final tables |

Parallel execution recipe (this is how the round was actually run — 24-way sharding turns a
~90 min sequential job into a few minutes):

```bash
for m in <model keys>; do for s in 7 17 27; do for i in pretrained random; do
  OPENLLM_MODELS=$m OPENLLM_SEEDS=$s OPENLLM_INITS=$i OPENLLM_OUT_SUFFIX=__${m}_s${s}_${i} \
    setsid nohup $PY -u scripts/openllm_remaining_experiments.py > logs/fine/${m}_s${s}_${i}.log 2>&1 &
done; done; done
$PY scripts/openllm_merge_parts.py
```

Environment note: DeepSeek-LLM-7B ships `pytorch_model.bin`, which torch 2.4 refuses to load
(`CVE-2025-32434` guard) — its feature extraction was run with the `chatts` env (torch 2.6).
Everything else used `wavellm`. Shard logs are kept in `logs/fine/` and `logs/stab_*.log`.
