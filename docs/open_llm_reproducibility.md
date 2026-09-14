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
