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
