# Open-LLM Cross-Family Attribution Suite — Stage P0-0 Repository & Environment Audit

Date: 2026-09-11. Repo: `/9950backfile/chenjiahui/ARForecast-Diagnosis`
Git commit at audit start: `fd27169` (branch `main`, remote `github-cjhcjh123-666/cjhcjh123-666/ARForecast-Diagnosis`).

## 1. Environment
- Python envs: `/public/duyinglong/miniconda3/envs/wavellm` (torch 2.4.1+cu121, transformers 4.52.4, einops 0.4.0),
  `chatts` (torch 2.6.0+cu124, transformers 5.16.1, hydra, gluonts) used for Moirai/uni2ts in the prior round.
- GPU: **8 × NVIDIA A800-SXM4-80GB** (81920 MiB each), driver 535.261.03, CUDA 12.2.
  At audit time each card had ~9.1–9.5 GB used by other users (≈72 GB free/card).
- Disk: `/9950backfile` 113 TB available; HF cache `/9950backfile/chenjiahui/hf_cache` (252 GB used).
- Network: huggingface.co is **not** directly reachable (DNS/TLS blocked, proxy 127.0.0.1:7890 fails TLS);
  **hf-mirror.com works directly with proxies disabled** (verified HTTP 200 + snapshot download).
  `api2.aigcbest.top` is reachable directly (HTTP 200 on `/v1/models` with the provided key).
- HF_TOKEN: **unset** (no token file). Gated repos therefore cannot be fetched (see §5).

## 2. Existing code entry points (reused, not reimplemented)
| stage | file |
|---|---|
| synthetic generator / standardization | `data/dynamics.py` (`generate_window`, `build_labeled_windows`, `_standardize`) |
| prompt serialization + parser + hidden extraction | `scripts/run_frozen_probe.py` (`_format_values`, `_prompts`, `extract_last_hidden`, parser `(?<![0-9])[+-]?\d+\.\d{2}`) |
| probe/router utilities | `analysis/linear_probe.py` (`softmax_regression`, `softmax_predict`, `accuracy`) |
| expert bank | `models/experts.py` (`TrendExpert`, `PeriodicExpert`, `LocalExpert`) |
| Qwen/Chronos/TimesFM/MOMENT/Moirai unified runner (prior round) | `scripts/iclr_tsfm_deliver.py`, `scripts/iclr_tsfm_oracleC.py`, `scripts/iclr_tsfm_init_audit.py` |
| reviewer follow-ups (margin, 5-expert, dim, MLP, 10-seed) | `scripts/tsfm_*` (see `tsfm_reviewer_followup.md`) |
| real-data zero-shot routing windows | `results/iclr/multi_dataset_router/qwen3_8b_official/windows_qwen3_8b_official_s7/*.npz` |
| **new** open-LLM runner (this round) | `scripts/openllm_suite.py`, `scripts/openllm_smoke.py`, `scripts/openllm_download.py` |

## 3. Frozen protocol (must not change)
- C=64, H=16, context-only standardization (mean/std of the 64-length context).
- 5 families × 150/seed: trend, periodic, local AR, mixture, regime; train 90 / test 60 per family
  → clean 5-way 450/300; router trains on the 270 clean trend/periodic/local windows.
- LM serialization: `_format_values` = clip to ±9.99, 2 decimals, explicit sign, space-separated;
  prompt = `Forecast the next values of this normalized time series.\nHistory: <values>\nForecast:`.
- Representation: final transformer layer, final non-padding token (mean-pool variant as sensitivity).
- Pretrained vs matched random: random = architecture-native **from-config** construction (no released
  weights), seeded; tokenizer/serialization/inference identical.
- OOD sets: `results/iclr/e4_balanced3/windows/bal3_s{seed}.npz` (mixture/regime→T/P + AR_weak_sine→L)
  and `.../e4_balanced3_natural/windows/bal3n_s{seed}.npz` (local = periodic+AR), 40/40/40 per seed.
- Main seeds 7/17/27; headline replication 7…97 (10 seeds; extra windows already generated 2026-09-07).

## 4. Existing models & results (retained, not replaced)
- Local: `models/qwen3-8b-base` (official Qwen3-8B-Base), qwen3-0.6b/1.7b in `/9950backfile/chenjiahui/hf_cache`,
  TSFMs (Chronos T5 small/base, Bolt small, TimesFM 2.5, MOMENT-1-large, Moirai-1.1-R-small) via
  `/public/chenjiahui/SAFER-TS/code/model_cache` and WaveFormer paths.
- Deliverables: `results/iclr/tsfm_deliver_v2/` (metrics 324 rows, 84 per-window npz, audit, init_audit),
  `tsfm_suite_deliverables_v3.zip`, `tsfm_qwen_features.zip`, `tsfm_reviewer_followup.md`.

## 5. Model access status (P0/P1 open LMs)
Download path: `HF_ENDPOINT=https://hf-mirror.com`, proxies disabled, `scripts/openllm_download.py`,
cache `/9950backfile/chenjiahui/hf_cache/hub`, revisions pinned to resolved SHA.

| key | model_id | gated | status (2026-09-11, in progress) |
|---|---|---|---|
| qwen3_8b_base | Qwen/Qwen3-8B-Base | no | already local (`models/qwen3-8b-base`) |
| deepseek_llm_7b | deepseek-ai/deepseek-llm-7b-base | no | downloading |
| deepseek_v2_lite | deepseek-ai/DeepSeek-V2-Lite (MoE) | no | queued |
| mistral_7b_v03 | mistralai/Mistral-7B-v0.3 | no | queued |
| olmo2_7b | allenai/OLMo-2-1124-7B | no | queued |
| llama31_8b | meta-llama/Llama-3.1-8B | **yes** | BLOCKED_GATED_ACCESS (no HF_TOKEN) unless token provided |
| gemma2_9b | google/gemma-2-9b | **yes** | BLOCKED_GATED_ACCESS (no HF_TOKEN) |
| llama32_3b | meta-llama/Llama-3.2-3B | **yes** | BLOCKED_GATED_ACCESS |
| olmo2_13b | allenai/OLMo-2-1124-13B | no | queued (P1) |

**Action needed from user:** export `HF_TOKEN=<token with Llama/Gemma access>` (or make the mirror-accessible
copies available) to unblock the Llama/Gemma rows of the matched-size table. All other P0 models proceed without it.

## 6. API (direct numerical generation only)
- Key present in `api.txt`; must be passed via `AIGCBEST_API_KEY` (never committed).
- `GET https://api2.aigcbest.top/v1/models` returns HTTP 200 (OpenAI-compatible listing).
- Paid generation gated behind `RUN_PAID_API=1` + `API_MAX_REQUESTS`/`API_BUDGET_USD`.

## 7. What already exists vs what this round adds
Already: Qwen3 family + 6 TSFMs on clean/shuffle/readout/family-label/oracle-label/5-expert/dim/MLP/10-seed/
real-world/native; margin & bank sensitivity; init audit; per-window data.
New this round: open base LMs (DeepSeek dense/MoE, Mistral, OLMo; Llama/Gemma if unblocked) run through the
**same** protocol: matched random, clean/shuffle/readout, family/oracle routing, MLP router, pooling, native
generation, real-world zero-shot routing, cross-family statistics + paper tables/figures, plus an API
direct-generation zoo (separate, non-attribution table).
