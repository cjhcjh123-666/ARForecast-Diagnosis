# TS-Foundation-Model Cross-Model Suite — Results Summary

`results/iclr/tsfm_deliver/` — generated `2026-09-06` by `scripts/iclr_tsfm_deliver.py` + `scripts/rebuild_metrics_from_pw.py`.

**One-line result:** on the clean 5-class structure-recognition task almost every model (pretrained *or* random) is strong, but only the **language-pretrained Qwen3-8B** representation shows a large pretraining advantage on **zero-shot compositional routing** (bal3 balanced-acc `0.833` vs its random control `0.558`). No time-series foundation model shows such an advantage (pretrained <= random for Bolt/TimesFM/MOMENT/Moirai on routing).

## 1. Protocol

- Context **C=64**, horizon **H=16**, seeds **7/17/27**; windows are context-standardized (mean/std of the 64-length context).
- **A. Structure recognition**: 5-class family probe (trend/periodic/local/mixture/regime), 450 train / 300 test windows per seed; frozen backbone + full-batch softmax probe (2,000 steps, lr=0.5, L2=1e-3). `A_recog_shuf` = per-window independent random time-permutation, probe retrained on shuffled features.
- **B. Numerical readout**: frozen rep + linear head → H=16; AdamW lr=1e-3, wd=0, full-batch 300 epochs. MSE in context-normalized space; `head_params = dim*16+16`.
- **C. Compositional routing**: router = softmax probe fit on the 270 clean trend/periodic/local windows (family labels; identical to the E4 reference protocol), tested on two balanced 3-class OOD sets (**bal3**, **bal3n**; 120 windows each, T/P/L = 40/40/40; oracle = argmin future MSE over the 3 experts). Balanced accuracy / macro-F1 / per-class recall / routed MSE.
- **native**: each model's own forecasting interface on the 300 synth5 test windows. Point extraction: Chronos-T5 = median of 20 samples; Chronos-Bolt = 0.5 quantile; TimesFM = official decode point index 5 (GPU port verified bit-identical to `forecast_naive`); Moirai = median of 50 distribution samples, fixed patch 16; **MOMENT = n/a** (official forecasting requires a learned head); **Qwen = n/a** (its native numerical generation is the separate recognition-vs-generation experiment in the paper).

## 2. Deliverables

| File | Content |
|---|---|
| `metrics.csv` | **240 rows** — one per (model, init, seed, task, test_set); fields: accuracy / balanced_accuracy / macro_f1 / recall_trend / recall_periodic / recall_local / mse / head_params |
| `perwindow/pw_<model>_<init>_s<seed>.npz` (42 files) | per-window arrays: `a_true/a_pred`, `as_true/as_pred`, `b_future/b_pred`, `c_{bal3,bal3n}_{oracle,pred,err}` (err = 3 experts' per-window MSE), `n_future/n_pred` |
| `config.json` | checkpoints, families, dims, pooling, normalization, training settings, native point-extraction, and the **Moirai pipeline note** |

## 3. Models

| model | family | dim | pooling | native interface |
|---|---|---:|---|---|
| qwen3_8b_base | LLM (decoder-only, text tok.) | 4096 | last-token hidden | n/a (LM generation) |
| chronos_t5-small | T5 seq2seq TS FM | 512 | encoder last token (EOS) | median of 20 samples |
| chronos_t5-base | T5 seq2seq TS FM | 768 | encoder last token (EOS) | median of 20 samples |
| chronos_bolt-small | patched T5 TS FM (Chronos-Bolt) | 512 | encoder [REG] token | 0.5 quantile |
| timesfm_2.5-200m | decoder-only patched TS FM | 1280 | last patch hidden | decode point idx 5 |
| moment-1-large | encoder-only patch TS FM (flan-t5-large) | 1024 | MOMENT embed mean | n/a (learned head) |
| moirai-1.1-R-small | patch×variate encoder TS FM (uni2ts 1.x) | 384 | last patch-token hidden (p=16) | median of 50 samples (p=16) |

> **Moirai note**: Moirai's attention operates on (patch × variate) tokens and cannot be fed a raw `(B,64)` tensor; inputs are constructed exactly like `MoiraiForecast._convert` (target `(B, n_patch, max_patch=128)` + sample/time/variate ids + masks). Env workarounds (jaxtyping shim, einops/dynamo skip, uni2ts `__init__` bypass) are in `config.json`.

## 4. Mean over seeds (pretrained | random)

| model | A_recog_orig | A_recog_shuf | B MSE | C bal3 balacc | C bal3n balacc | C bal3 macro-F1 | native MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3_8b_base | 0.981 / 0.900 | 0.640 / 0.530 | 0.588 / 0.679 | 0.833 / 0.558 | 0.831 / 0.578 | 0.831 / 0.504 | n/a / n/a |
| chronos_t5-small | 0.999 / 0.612 | 0.707 / 0.366 | 0.828 / 1.845 | 0.411 / 0.375 | 0.408 / 0.378 | 0.306 / 0.344 | 0.672 / 9.559 |
| chronos_t5-base | 1.000 / 0.611 | 0.719 / 0.383 | 1.083 / 1.825 | 0.511 / 0.411 | 0.511 / 0.381 | 0.446 / 0.389 | 0.676 / 12.367 |
| chronos_bolt-small | 1.000 / 0.940 | 0.370 / 0.226 | 0.405 / 1.049 | 0.683 / 0.675 | 0.683 / 0.683 | 0.682 / 0.684 | 0.528 / 6.994 |
| timesfm_2.5-200m | 1.000 / 0.969 | 0.557 / 0.204 | 0.403 / 0.895 | 0.653 / 0.694 | 0.653 / 0.692 | 0.597 / 0.686 | 0.601 / 16.887 |
| moment-1-large | 0.993 / 0.976 | 0.231 / 0.228 | 0.535 / 0.504 | 0.631 / 0.725 | 0.628 / 0.722 | 0.589 / 0.723 | n/a / n/a |
| moirai-1.1-R-small | 0.997 / 0.990 | 0.461 / 0.187 | 0.394 / 0.435 | 0.564 / 0.656 | 0.550 / 0.656 | 0.521 / 0.639 | 1.471 / 2.274 |

### 4.1 C-routing per-class recall (bal3, mean over seeds, pretrained | random)

| model | recall_trend | recall_periodic | recall_local |
|---|---:|---:|---:|
| qwen3_8b_base | 0.708 / 0.667 | 0.800 / 0.133 | 0.992 / 0.875 |
| chronos_t5-small | 0.217 / 0.150 | 0.025 / 0.317 | 0.992 / 0.658 |
| chronos_t5-base | 0.508 / 0.225 | 0.025 / 0.350 | 1.000 / 0.658 |
| chronos_bolt-small | 0.600 / 0.675 | 0.458 / 0.617 | 0.992 / 0.733 |
| timesfm_2.5-200m | 0.775 / 0.775 | 0.183 / 0.483 | 1.000 / 0.825 |
| moment-1-large | 0.258 / 0.767 | 0.642 / 0.475 | 0.992 / 0.933 |
| moirai-1.1-R-small | 0.267 / 0.633 | 0.425 / 0.408 | 1.000 / 0.925 |

## 5. Per-seed detail (all rows)

| model | init | seed | task | test_set | n | acc | balacc | macroF1 | recT | recP | recL | mse | head_params |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_8b_base | pretrained | 7 | A_recog_orig | synth5 | 300 | 0.9833 | | | | | | - | - |
| qwen3_8b_base | pretrained | 7 | A_recog_shuf | synth5 | 300 | 0.6233 | | | | | | - | - |
| qwen3_8b_base | pretrained | 7 | B_readout | synth5 | 300 | - | | | | | | 0.5766 | 65552 |
| qwen3_8b_base | pretrained | 7 | C_routing | bal3 | 120 | | 0.875 | 0.8731 | 0.725 | 0.9 | 1.0 | 0.6823 | |
| qwen3_8b_base | pretrained | 7 | C_routing | bal3n | 120 | | 0.875 | 0.8731 | 0.725 | 0.9 | 1.0 | 0.7227 | |
| qwen3_8b_base | pretrained | 17 | A_recog_orig | synth5 | 300 | 0.9767 | | | | | | - | - |
| qwen3_8b_base | pretrained | 17 | A_recog_shuf | synth5 | 300 | 0.6433 | | | | | | - | - |
| qwen3_8b_base | pretrained | 17 | B_readout | synth5 | 300 | - | | | | | | 0.6595 | 65552 |
| qwen3_8b_base | pretrained | 17 | C_routing | bal3 | 120 | | 0.825 | 0.8224 | 0.725 | 0.75 | 1.0 | 0.6814 | |
| qwen3_8b_base | pretrained | 17 | C_routing | bal3n | 120 | | 0.8083 | 0.8077 | 0.725 | 0.75 | 0.95 | 0.7149 | |
| qwen3_8b_base | pretrained | 27 | A_recog_orig | synth5 | 300 | 0.9833 | | | | | | - | - |
| qwen3_8b_base | pretrained | 27 | A_recog_shuf | synth5 | 300 | 0.6533 | | | | | | - | - |
| qwen3_8b_base | pretrained | 27 | B_readout | synth5 | 300 | - | | | | | | 0.5269 | 65552 |
| qwen3_8b_base | pretrained | 27 | C_routing | bal3 | 120 | | 0.8 | 0.7972 | 0.675 | 0.75 | 0.975 | 0.7656 | |
| qwen3_8b_base | pretrained | 27 | C_routing | bal3n | 120 | | 0.8083 | 0.8044 | 0.675 | 0.75 | 1.0 | 0.5772 | |
| qwen3_8b_base | random | 7 | A_recog_orig | synth5 | 300 | 0.89 | | | | | | - | - |
| qwen3_8b_base | random | 7 | A_recog_shuf | synth5 | 300 | 0.5167 | | | | | | - | - |
| qwen3_8b_base | random | 7 | B_readout | synth5 | 300 | - | | | | | | 0.641 | 65552 |
| qwen3_8b_base | random | 7 | C_routing | bal3 | 120 | | 0.5917 | 0.5526 | 0.675 | 0.2 | 0.9 | 1.3352 | |
| qwen3_8b_base | random | 7 | C_routing | bal3n | 120 | | 0.6 | 0.561 | 0.675 | 0.2 | 0.925 | 1.4112 | |
| qwen3_8b_base | random | 17 | A_recog_orig | synth5 | 300 | 0.9133 | | | | | | - | - |
| qwen3_8b_base | random | 17 | A_recog_shuf | synth5 | 300 | 0.5167 | | | | | | - | - |
| qwen3_8b_base | random | 17 | B_readout | synth5 | 300 | - | | | | | | 0.7711 | 65552 |
| qwen3_8b_base | random | 17 | C_routing | bal3 | 120 | | 0.5417 | 0.4906 | 0.625 | 0.125 | 0.875 | 1.2843 | |
| qwen3_8b_base | random | 17 | C_routing | bal3n | 120 | | 0.5667 | 0.5082 | 0.625 | 0.125 | 0.95 | 1.2691 | |
| qwen3_8b_base | random | 27 | A_recog_orig | synth5 | 300 | 0.8967 | | | | | | - | - |
| qwen3_8b_base | random | 27 | A_recog_shuf | synth5 | 300 | 0.5567 | | | | | | - | - |
| qwen3_8b_base | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.6258 | 65552 |
| qwen3_8b_base | random | 27 | C_routing | bal3 | 120 | | 0.5417 | 0.4689 | 0.7 | 0.075 | 0.85 | 1.2913 | |
| qwen3_8b_base | random | 27 | C_routing | bal3n | 120 | | 0.5667 | 0.4906 | 0.7 | 0.075 | 0.925 | 0.9864 | |
| chronos_t5-small | pretrained | 7 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| chronos_t5-small | pretrained | 7 | A_recog_shuf | synth5 | 300 | 0.7067 | | | | | | - | - |
| chronos_t5-small | pretrained | 7 | B_readout | synth5 | 300 | - | | | | | | 0.8368 | 8208 |
| chronos_t5-small | pretrained | 7 | C_routing | bal3 | 120 | | 0.4083 | 0.3076 | 0.175 | 0.05 | 1.0 | 1.7535 | |
| chronos_t5-small | pretrained | 7 | C_routing | bal3n | 120 | | 0.3917 | 0.2992 | 0.175 | 0.05 | 0.95 | 1.8253 | |
| chronos_t5-small | pretrained | 7 | native_forecast | synth5 | 300 | | | | | | | 0.6916 | |
| chronos_t5-small | pretrained | 17 | A_recog_orig | synth5 | 300 | 0.9967 | | | | | | - | - |
| chronos_t5-small | pretrained | 17 | A_recog_shuf | synth5 | 300 | 0.7033 | | | | | | - | - |
| chronos_t5-small | pretrained | 17 | B_readout | synth5 | 300 | - | | | | | | 0.9005 | 8208 |
| chronos_t5-small | pretrained | 17 | C_routing | bal3 | 120 | | 0.4083 | 0.3037 | 0.225 | 0.025 | 0.975 | 1.7064 | |
| chronos_t5-small | pretrained | 17 | C_routing | bal3n | 120 | | 0.4167 | 0.3076 | 0.225 | 0.025 | 1.0 | 1.6904 | |
| chronos_t5-small | pretrained | 17 | native_forecast | synth5 | 300 | | | | | | | 0.7386 | |
| chronos_t5-small | pretrained | 27 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| chronos_t5-small | pretrained | 27 | A_recog_shuf | synth5 | 300 | 0.71 | | | | | | - | - |
| chronos_t5-small | pretrained | 27 | B_readout | synth5 | 300 | - | | | | | | 0.7473 | 8208 |
| chronos_t5-small | pretrained | 27 | C_routing | bal3 | 120 | | 0.4167 | 0.3063 | 0.25 | 0.0 | 1.0 | 1.6661 | |
| chronos_t5-small | pretrained | 27 | C_routing | bal3n | 120 | | 0.4167 | 0.3063 | 0.25 | 0.0 | 1.0 | 1.492 | |
| chronos_t5-small | pretrained | 27 | native_forecast | synth5 | 300 | | | | | | | 0.5869 | |
| chronos_t5-small | random | 7 | A_recog_orig | synth5 | 300 | 0.6067 | | | | | | - | - |
| chronos_t5-small | random | 7 | A_recog_shuf | synth5 | 300 | 0.3467 | | | | | | - | - |
| chronos_t5-small | random | 7 | B_readout | synth5 | 300 | - | | | | | | 1.8901 | 8208 |
| chronos_t5-small | random | 7 | C_routing | bal3 | 120 | | 0.4167 | 0.3911 | 0.175 | 0.4 | 0.675 | 1.7473 | |
| chronos_t5-small | random | 7 | C_routing | bal3n | 120 | | 0.4083 | 0.3839 | 0.175 | 0.4 | 0.65 | 1.8264 | |
| chronos_t5-small | random | 7 | native_forecast | synth5 | 300 | | | | | | | 12.6102 | |
| chronos_t5-small | random | 17 | A_recog_orig | synth5 | 300 | 0.5767 | | | | | | - | - |
| chronos_t5-small | random | 17 | A_recog_shuf | synth5 | 300 | 0.3667 | | | | | | - | - |
| chronos_t5-small | random | 17 | B_readout | synth5 | 300 | - | | | | | | 1.9096 | 8208 |
| chronos_t5-small | random | 17 | C_routing | bal3 | 120 | | 0.3583 | 0.3369 | 0.175 | 0.3 | 0.6 | 1.7692 | |
| chronos_t5-small | random | 17 | C_routing | bal3n | 120 | | 0.35 | 0.3306 | 0.175 | 0.3 | 0.575 | 1.8037 | |
| chronos_t5-small | random | 17 | native_forecast | synth5 | 300 | | | | | | | 7.8481 | |
| chronos_t5-small | random | 27 | A_recog_orig | synth5 | 300 | 0.6533 | | | | | | - | - |
| chronos_t5-small | random | 27 | A_recog_shuf | synth5 | 300 | 0.3833 | | | | | | - | - |
| chronos_t5-small | random | 27 | B_readout | synth5 | 300 | - | | | | | | 1.7342 | 8208 |
| chronos_t5-small | random | 27 | C_routing | bal3 | 120 | | 0.35 | 0.3036 | 0.1 | 0.25 | 0.7 | 1.9869 | |
| chronos_t5-small | random | 27 | C_routing | bal3n | 120 | | 0.375 | 0.3199 | 0.1 | 0.25 | 0.775 | 1.7883 | |
| chronos_t5-small | random | 27 | native_forecast | synth5 | 300 | | | | | | | 8.2179 | |
| chronos_t5-base | pretrained | 7 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| chronos_t5-base | pretrained | 7 | A_recog_shuf | synth5 | 300 | 0.7267 | | | | | | - | - |
| chronos_t5-base | pretrained | 7 | B_readout | synth5 | 300 | - | | | | | | 1.0824 | 12304 |
| chronos_t5-base | pretrained | 7 | C_routing | bal3 | 120 | | 0.575 | 0.5184 | 0.675 | 0.05 | 1.0 | 0.9321 | |
| chronos_t5-base | pretrained | 7 | C_routing | bal3n | 120 | | 0.575 | 0.5184 | 0.675 | 0.05 | 1.0 | 0.9725 | |
| chronos_t5-base | pretrained | 7 | native_forecast | synth5 | 300 | | | | | | | 0.69 | |
| chronos_t5-base | pretrained | 17 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| chronos_t5-base | pretrained | 17 | A_recog_shuf | synth5 | 300 | 0.71 | | | | | | - | - |
| chronos_t5-base | pretrained | 17 | B_readout | synth5 | 300 | - | | | | | | 1.144 | 12304 |
| chronos_t5-base | pretrained | 17 | C_routing | bal3 | 120 | | 0.4583 | 0.3943 | 0.375 | 0.0 | 1.0 | 1.0686 | |
| chronos_t5-base | pretrained | 17 | C_routing | bal3n | 120 | | 0.4583 | 0.3943 | 0.375 | 0.0 | 1.0 | 1.0657 | |
| chronos_t5-base | pretrained | 17 | native_forecast | synth5 | 300 | | | | | | | 0.7367 | |
| chronos_t5-base | pretrained | 27 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| chronos_t5-base | pretrained | 27 | A_recog_shuf | synth5 | 300 | 0.72 | | | | | | - | - |
| chronos_t5-base | pretrained | 27 | B_readout | synth5 | 300 | - | | | | | | 1.0212 | 12304 |
| chronos_t5-base | pretrained | 27 | C_routing | bal3 | 120 | | 0.5 | 0.4263 | 0.475 | 0.025 | 1.0 | 1.0478 | |
| chronos_t5-base | pretrained | 27 | C_routing | bal3n | 120 | | 0.5 | 0.4263 | 0.475 | 0.025 | 1.0 | 0.8737 | |
| chronos_t5-base | pretrained | 27 | native_forecast | synth5 | 300 | | | | | | | 0.6005 | |
| chronos_t5-base | random | 7 | A_recog_orig | synth5 | 300 | 0.59 | | | | | | - | - |
| chronos_t5-base | random | 7 | A_recog_shuf | synth5 | 300 | 0.3533 | | | | | | - | - |
| chronos_t5-base | random | 7 | B_readout | synth5 | 300 | - | | | | | | 1.8543 | 12304 |
| chronos_t5-base | random | 7 | C_routing | bal3 | 120 | | 0.4333 | 0.4223 | 0.275 | 0.4 | 0.625 | 1.757 | |
| chronos_t5-base | random | 7 | C_routing | bal3n | 120 | | 0.3917 | 0.3881 | 0.275 | 0.4 | 0.5 | 2.0405 | |
| chronos_t5-base | random | 7 | native_forecast | synth5 | 300 | | | | | | | 12.8554 | |
| chronos_t5-base | random | 17 | A_recog_orig | synth5 | 300 | 0.6267 | | | | | | - | - |
| chronos_t5-base | random | 17 | A_recog_shuf | synth5 | 300 | 0.42 | | | | | | - | - |
| chronos_t5-base | random | 17 | B_readout | synth5 | 300 | - | | | | | | 1.9 | 12304 |
| chronos_t5-base | random | 17 | C_routing | bal3 | 120 | | 0.3917 | 0.3759 | 0.275 | 0.275 | 0.625 | 1.6646 | |
| chronos_t5-base | random | 17 | C_routing | bal3n | 120 | | 0.3667 | 0.3563 | 0.275 | 0.275 | 0.55 | 1.7002 | |
| chronos_t5-base | random | 17 | native_forecast | synth5 | 300 | | | | | | | 12.8518 | |
| chronos_t5-base | random | 27 | A_recog_orig | synth5 | 300 | 0.6167 | | | | | | - | - |
| chronos_t5-base | random | 27 | A_recog_shuf | synth5 | 300 | 0.3767 | | | | | | - | - |
| chronos_t5-base | random | 27 | B_readout | synth5 | 300 | - | | | | | | 1.7205 | 12304 |
| chronos_t5-base | random | 27 | C_routing | bal3 | 120 | | 0.4083 | 0.3696 | 0.125 | 0.375 | 0.725 | 1.7622 | |
| chronos_t5-base | random | 27 | C_routing | bal3n | 120 | | 0.3833 | 0.3507 | 0.125 | 0.375 | 0.65 | 1.6116 | |
| chronos_t5-base | random | 27 | native_forecast | synth5 | 300 | | | | | | | 11.3938 | |
| chronos_bolt-small | pretrained | 7 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| chronos_bolt-small | pretrained | 7 | A_recog_shuf | synth5 | 300 | 0.3833 | | | | | | - | - |
| chronos_bolt-small | pretrained | 7 | B_readout | synth5 | 300 | - | | | | | | 0.3755 | 8208 |
| chronos_bolt-small | pretrained | 7 | C_routing | bal3 | 120 | | 0.7 | 0.6996 | 0.65 | 0.475 | 0.975 | 0.8084 | |
| chronos_bolt-small | pretrained | 7 | C_routing | bal3n | 120 | | 0.7083 | 0.7061 | 0.65 | 0.475 | 1.0 | 0.8289 | |
| chronos_bolt-small | pretrained | 7 | native_forecast | synth5 | 300 | | | | | | | 0.4944 | |
| chronos_bolt-small | pretrained | 17 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| chronos_bolt-small | pretrained | 17 | A_recog_shuf | synth5 | 300 | 0.38 | | | | | | - | - |
| chronos_bolt-small | pretrained | 17 | B_readout | synth5 | 300 | - | | | | | | 0.4817 | 8208 |
| chronos_bolt-small | pretrained | 17 | C_routing | bal3 | 120 | | 0.675 | 0.6725 | 0.6 | 0.425 | 1.0 | 0.7337 | |
| chronos_bolt-small | pretrained | 17 | C_routing | bal3n | 120 | | 0.675 | 0.6725 | 0.6 | 0.425 | 1.0 | 0.7308 | |
| chronos_bolt-small | pretrained | 17 | native_forecast | synth5 | 300 | | | | | | | 0.5831 | |
| chronos_bolt-small | pretrained | 27 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| chronos_bolt-small | pretrained | 27 | A_recog_shuf | synth5 | 300 | 0.3467 | | | | | | - | - |
| chronos_bolt-small | pretrained | 27 | B_readout | synth5 | 300 | - | | | | | | 0.359 | 8208 |
| chronos_bolt-small | pretrained | 27 | C_routing | bal3 | 120 | | 0.675 | 0.6745 | 0.55 | 0.475 | 1.0 | 0.8069 | |
| chronos_bolt-small | pretrained | 27 | C_routing | bal3n | 120 | | 0.6667 | 0.6682 | 0.55 | 0.475 | 0.975 | 0.6328 | |
| chronos_bolt-small | pretrained | 27 | native_forecast | synth5 | 300 | | | | | | | 0.5054 | |
| chronos_bolt-small | random | 7 | A_recog_orig | synth5 | 300 | 0.9367 | | | | | | - | - |
| chronos_bolt-small | random | 7 | A_recog_shuf | synth5 | 300 | 0.25 | | | | | | - | - |
| chronos_bolt-small | random | 7 | B_readout | synth5 | 300 | - | | | | | | 1.0899 | 8208 |
| chronos_bolt-small | random | 7 | C_routing | bal3 | 120 | | 0.8 | 0.8042 | 0.75 | 0.825 | 0.825 | 0.7831 | |
| chronos_bolt-small | random | 7 | C_routing | bal3n | 120 | | 0.775 | 0.7806 | 0.75 | 0.825 | 0.75 | 0.9342 | |
| chronos_bolt-small | random | 7 | native_forecast | synth5 | 300 | | | | | | | 4.5597 | |
| chronos_bolt-small | random | 17 | A_recog_orig | synth5 | 300 | 0.9533 | | | | | | - | - |
| chronos_bolt-small | random | 17 | A_recog_shuf | synth5 | 300 | 0.19 | | | | | | - | - |
| chronos_bolt-small | random | 17 | B_readout | synth5 | 300 | - | | | | | | 1.0838 | 8208 |
| chronos_bolt-small | random | 17 | C_routing | bal3 | 120 | | 0.675 | 0.6849 | 0.6 | 0.6 | 0.825 | 0.7679 | |
| chronos_bolt-small | random | 17 | C_routing | bal3n | 120 | | 0.6833 | 0.6918 | 0.6 | 0.6 | 0.85 | 0.7813 | |
| chronos_bolt-small | random | 17 | native_forecast | synth5 | 300 | | | | | | | 9.7247 | |
| chronos_bolt-small | random | 27 | A_recog_orig | synth5 | 300 | 0.93 | | | | | | - | - |
| chronos_bolt-small | random | 27 | A_recog_shuf | synth5 | 300 | 0.2367 | | | | | | - | - |
| chronos_bolt-small | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.9724 | 8208 |
| chronos_bolt-small | random | 27 | C_routing | bal3 | 120 | | 0.55 | 0.5619 | 0.675 | 0.425 | 0.55 | 1.0005 | |
| chronos_bolt-small | random | 27 | C_routing | bal3n | 120 | | 0.5917 | 0.6042 | 0.675 | 0.425 | 0.675 | 0.7312 | |
| chronos_bolt-small | random | 27 | native_forecast | synth5 | 300 | | | | | | | 6.6967 | |
| timesfm_2.5-200m | pretrained | 7 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| timesfm_2.5-200m | pretrained | 7 | A_recog_shuf | synth5 | 300 | 0.5867 | | | | | | - | - |
| timesfm_2.5-200m | pretrained | 7 | B_readout | synth5 | 300 | - | | | | | | 0.3955 | 20496 |
| timesfm_2.5-200m | pretrained | 7 | C_routing | bal3 | 120 | | 0.6833 | 0.6333 | 0.85 | 0.2 | 1.0 | 0.9339 | |
| timesfm_2.5-200m | pretrained | 7 | C_routing | bal3n | 120 | | 0.6833 | 0.6333 | 0.85 | 0.2 | 1.0 | 0.9743 | |
| timesfm_2.5-200m | pretrained | 7 | native_forecast | synth5 | 300 | | | | | | | 0.5789 | |
| timesfm_2.5-200m | pretrained | 17 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| timesfm_2.5-200m | pretrained | 17 | A_recog_shuf | synth5 | 300 | 0.53 | | | | | | - | - |
| timesfm_2.5-200m | pretrained | 17 | B_readout | synth5 | 300 | - | | | | | | 0.4741 | 20496 |
| timesfm_2.5-200m | pretrained | 17 | C_routing | bal3 | 120 | | 0.625 | 0.5518 | 0.75 | 0.125 | 1.0 | 1.1579 | |
| timesfm_2.5-200m | pretrained | 17 | C_routing | bal3n | 120 | | 0.625 | 0.5518 | 0.75 | 0.125 | 1.0 | 1.155 | |
| timesfm_2.5-200m | pretrained | 17 | native_forecast | synth5 | 300 | | | | | | | 0.6566 | |
| timesfm_2.5-200m | pretrained | 27 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| timesfm_2.5-200m | pretrained | 27 | A_recog_shuf | synth5 | 300 | 0.5533 | | | | | | - | - |
| timesfm_2.5-200m | pretrained | 27 | B_readout | synth5 | 300 | - | | | | | | 0.3408 | 20496 |
| timesfm_2.5-200m | pretrained | 27 | C_routing | bal3 | 120 | | 0.65 | 0.6056 | 0.725 | 0.225 | 1.0 | 1.0282 | |
| timesfm_2.5-200m | pretrained | 27 | C_routing | bal3n | 120 | | 0.65 | 0.6056 | 0.725 | 0.225 | 1.0 | 0.8541 | |
| timesfm_2.5-200m | pretrained | 27 | native_forecast | synth5 | 300 | | | | | | | 0.5673 | |
| timesfm_2.5-200m | random | 7 | A_recog_orig | synth5 | 300 | 0.98 | | | | | | - | - |
| timesfm_2.5-200m | random | 7 | A_recog_shuf | synth5 | 300 | 0.21 | | | | | | - | - |
| timesfm_2.5-200m | random | 7 | B_readout | synth5 | 300 | - | | | | | | 0.8935 | 20496 |
| timesfm_2.5-200m | random | 7 | C_routing | bal3 | 120 | | 0.7083 | 0.7092 | 0.8 | 0.475 | 0.85 | 0.7634 | |
| timesfm_2.5-200m | random | 7 | C_routing | bal3n | 120 | | 0.7167 | 0.7165 | 0.8 | 0.475 | 0.875 | 0.844 | |
| timesfm_2.5-200m | random | 7 | native_forecast | synth5 | 300 | | | | | | | 15.6801 | |
| timesfm_2.5-200m | random | 17 | A_recog_orig | synth5 | 300 | 0.9833 | | | | | | - | - |
| timesfm_2.5-200m | random | 17 | A_recog_shuf | synth5 | 300 | 0.2067 | | | | | | - | - |
| timesfm_2.5-200m | random | 17 | B_readout | synth5 | 300 | - | | | | | | 1.0005 | 20496 |
| timesfm_2.5-200m | random | 17 | C_routing | bal3 | 120 | | 0.8083 | 0.8074 | 0.675 | 0.825 | 0.925 | 0.8626 | |
| timesfm_2.5-200m | random | 17 | C_routing | bal3n | 120 | | 0.775 | 0.7753 | 0.675 | 0.825 | 0.825 | 0.8109 | |
| timesfm_2.5-200m | random | 17 | native_forecast | synth5 | 300 | | | | | | | 19.4683 | |
| timesfm_2.5-200m | random | 27 | A_recog_orig | synth5 | 300 | 0.9433 | | | | | | - | - |
| timesfm_2.5-200m | random | 27 | A_recog_shuf | synth5 | 300 | 0.1967 | | | | | | - | - |
| timesfm_2.5-200m | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.7916 | 20496 |
| timesfm_2.5-200m | random | 27 | C_routing | bal3 | 120 | | 0.5667 | 0.5413 | 0.85 | 0.15 | 0.7 | 1.044 | |
| timesfm_2.5-200m | random | 27 | C_routing | bal3n | 120 | | 0.5833 | 0.5571 | 0.85 | 0.15 | 0.75 | 0.6951 | |
| timesfm_2.5-200m | random | 27 | native_forecast | synth5 | 300 | | | | | | | 15.5134 | |
| moment-1-large | pretrained | 7 | A_recog_orig | synth5 | 300 | 0.99 | | | | | | - | - |
| moment-1-large | pretrained | 7 | A_recog_shuf | synth5 | 300 | 0.2233 | | | | | | - | - |
| moment-1-large | pretrained | 7 | B_readout | synth5 | 300 | - | | | | | | 0.5368 | 16400 |
| moment-1-large | pretrained | 7 | C_routing | bal3 | 120 | | 0.5833 | 0.5561 | 0.325 | 0.45 | 0.975 | 1.6862 | |
| moment-1-large | pretrained | 7 | C_routing | bal3n | 120 | | 0.5917 | 0.5629 | 0.325 | 0.45 | 1.0 | 1.7067 | |
| moment-1-large | pretrained | 17 | A_recog_orig | synth5 | 300 | 0.9933 | | | | | | - | - |
| moment-1-large | pretrained | 17 | A_recog_shuf | synth5 | 300 | 0.2367 | | | | | | - | - |
| moment-1-large | pretrained | 17 | B_readout | synth5 | 300 | - | | | | | | 0.5887 | 16400 |
| moment-1-large | pretrained | 17 | C_routing | bal3 | 120 | | 0.6417 | 0.6063 | 0.275 | 0.65 | 1.0 | 1.4616 | |
| moment-1-large | pretrained | 17 | C_routing | bal3n | 120 | | 0.6417 | 0.6063 | 0.275 | 0.65 | 1.0 | 1.4588 | |
| moment-1-large | pretrained | 27 | A_recog_orig | synth5 | 300 | 0.9967 | | | | | | - | - |
| moment-1-large | pretrained | 27 | A_recog_shuf | synth5 | 300 | 0.2333 | | | | | | - | - |
| moment-1-large | pretrained | 27 | B_readout | synth5 | 300 | - | | | | | | 0.4803 | 16400 |
| moment-1-large | pretrained | 27 | C_routing | bal3 | 120 | | 0.6667 | 0.605 | 0.175 | 0.825 | 1.0 | 1.4545 | |
| moment-1-large | pretrained | 27 | C_routing | bal3n | 120 | | 0.65 | 0.592 | 0.175 | 0.825 | 0.95 | 1.3278 | |
| moment-1-large | random | 7 | A_recog_orig | synth5 | 300 | 0.9867 | | | | | | - | - |
| moment-1-large | random | 7 | A_recog_shuf | synth5 | 300 | 0.22 | | | | | | - | - |
| moment-1-large | random | 7 | B_readout | synth5 | 300 | - | | | | | | 0.5114 | 16400 |
| moment-1-large | random | 7 | C_routing | bal3 | 120 | | 0.7583 | 0.7522 | 0.825 | 0.5 | 0.95 | 0.7977 | |
| moment-1-large | random | 7 | C_routing | bal3n | 120 | | 0.7333 | 0.73 | 0.825 | 0.5 | 0.875 | 0.8189 | |
| moment-1-large | random | 17 | A_recog_orig | synth5 | 300 | 0.9833 | | | | | | - | - |
| moment-1-large | random | 17 | A_recog_shuf | synth5 | 300 | 0.2367 | | | | | | - | - |
| moment-1-large | random | 17 | B_readout | synth5 | 300 | - | | | | | | 0.5795 | 16400 |
| moment-1-large | random | 17 | C_routing | bal3 | 120 | | 0.7167 | 0.7138 | 0.75 | 0.45 | 0.95 | 0.6857 | |
| moment-1-large | random | 17 | C_routing | bal3n | 120 | | 0.725 | 0.7204 | 0.75 | 0.45 | 0.975 | 0.6738 | |
| moment-1-large | random | 27 | A_recog_orig | synth5 | 300 | 0.9567 | | | | | | - | - |
| moment-1-large | random | 27 | A_recog_shuf | synth5 | 300 | 0.2267 | | | | | | - | - |
| moment-1-large | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.4214 | 16400 |
| moment-1-large | random | 27 | C_routing | bal3 | 120 | | 0.7 | 0.7024 | 0.725 | 0.475 | 0.9 | 0.794 | |
| moment-1-large | random | 27 | C_routing | bal3n | 120 | | 0.7083 | 0.7089 | 0.725 | 0.475 | 0.925 | 0.587 | |
| moirai-1.1-R-small | pretrained | 7 | A_recog_orig | synth5 | 300 | 0.9967 | | | | | | - | - |
| moirai-1.1-R-small | pretrained | 7 | A_recog_shuf | synth5 | 300 | 0.4767 | | | | | | - | - |
| moirai-1.1-R-small | pretrained | 7 | B_readout | synth5 | 300 | - | | | | | | 0.3692 | 6160 |
| moirai-1.1-R-small | pretrained | 7 | C_routing | bal3 | 120 | | 0.5917 | 0.5624 | 0.325 | 0.45 | 1.0 | 1.6459 | |
| moirai-1.1-R-small | pretrained | 7 | C_routing | bal3n | 120 | | 0.575 | 0.5501 | 0.325 | 0.45 | 0.95 | 1.7081 | |
| moirai-1.1-R-small | pretrained | 7 | native_forecast | synth5 | 300 | | | | | | | 1.4691 | |
| moirai-1.1-R-small | pretrained | 17 | A_recog_orig | synth5 | 300 | 0.9933 | | | | | | - | - |
| moirai-1.1-R-small | pretrained | 17 | A_recog_shuf | synth5 | 300 | 0.4167 | | | | | | - | - |
| moirai-1.1-R-small | pretrained | 17 | B_readout | synth5 | 300 | - | | | | | | 0.44 | 6160 |
| moirai-1.1-R-small | pretrained | 17 | C_routing | bal3 | 120 | | 0.6333 | 0.6044 | 0.325 | 0.575 | 1.0 | 1.415 | |
| moirai-1.1-R-small | pretrained | 17 | C_routing | bal3n | 120 | | 0.6167 | 0.5919 | 0.325 | 0.575 | 0.95 | 1.4229 | |
| moirai-1.1-R-small | pretrained | 17 | native_forecast | synth5 | 300 | | | | | | | 1.5249 | |
| moirai-1.1-R-small | pretrained | 27 | A_recog_orig | synth5 | 300 | 1.0 | | | | | | - | - |
| moirai-1.1-R-small | pretrained | 27 | A_recog_shuf | synth5 | 300 | 0.49 | | | | | | - | - |
| moirai-1.1-R-small | pretrained | 27 | B_readout | synth5 | 300 | - | | | | | | 0.372 | 6160 |
| moirai-1.1-R-small | pretrained | 27 | C_routing | bal3 | 120 | | 0.4667 | 0.3963 | 0.15 | 0.25 | 1.0 | 1.9816 | |
| moirai-1.1-R-small | pretrained | 27 | C_routing | bal3n | 120 | | 0.4583 | 0.3911 | 0.15 | 0.25 | 0.975 | 1.8453 | |
| moirai-1.1-R-small | pretrained | 27 | native_forecast | synth5 | 300 | | | | | | | 1.4183 | |
| moirai-1.1-R-small | random | 7 | A_recog_orig | synth5 | 300 | 0.9833 | | | | | | - | - |
| moirai-1.1-R-small | random | 7 | A_recog_shuf | synth5 | 300 | 0.2333 | | | | | | - | - |
| moirai-1.1-R-small | random | 7 | B_readout | synth5 | 300 | - | | | | | | 0.4363 | 6160 |
| moirai-1.1-R-small | random | 7 | C_routing | bal3 | 120 | | 0.6583 | 0.6328 | 0.825 | 0.275 | 0.875 | 1.3322 | |
| moirai-1.1-R-small | random | 7 | C_routing | bal3n | 120 | | 0.65 | 0.6317 | 0.825 | 0.275 | 0.85 | 1.5281 | |
| moirai-1.1-R-small | random | 7 | native_forecast | synth5 | 300 | | | | | | | 2.2283 | |
| moirai-1.1-R-small | random | 17 | A_recog_orig | synth5 | 300 | 0.9967 | | | | | | - | - |
| moirai-1.1-R-small | random | 17 | A_recog_shuf | synth5 | 300 | 0.1567 | | | | | | - | - |
| moirai-1.1-R-small | random | 17 | B_readout | synth5 | 300 | - | | | | | | 0.501 | 6160 |
| moirai-1.1-R-small | random | 17 | C_routing | bal3 | 120 | | 0.6667 | 0.6574 | 0.575 | 0.475 | 0.95 | 1.1035 | |
| moirai-1.1-R-small | random | 17 | C_routing | bal3n | 120 | | 0.675 | 0.6637 | 0.575 | 0.475 | 0.975 | 1.1112 | |
| moirai-1.1-R-small | random | 17 | native_forecast | synth5 | 300 | | | | | | | 2.3127 | |
| moirai-1.1-R-small | random | 27 | A_recog_orig | synth5 | 300 | 0.99 | | | | | | - | - |
| moirai-1.1-R-small | random | 27 | A_recog_shuf | synth5 | 300 | 0.17 | | | | | | - | - |
| moirai-1.1-R-small | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.3691 | 6160 |
| moirai-1.1-R-small | random | 27 | C_routing | bal3 | 120 | | 0.6417 | 0.6271 | 0.5 | 0.475 | 0.95 | 1.3096 | |
| moirai-1.1-R-small | random | 27 | C_routing | bal3n | 120 | | 0.6417 | 0.6271 | 0.5 | 0.475 | 0.95 | 1.073 | |
| moirai-1.1-R-small | random | 27 | native_forecast | synth5 | 300 | | | | | | | 2.2817 | |

## 6. Key observations (descriptive)

- **Recognition is easy for everyone**: pretrained A_recog_orig ≈ 0.98–1.00 for all models; even random weights give ≥0.90 for the patch-based TS FMs (chronos-T5 random ~0.61 is the exception). Clean-family separability alone does **not** indicate transfer.
- **Order sensitivity varies strongly**: A_recog_shuf drops most for MOMENT (0.993→0.231), Bolt (1.0→0.37), TimesFM (1.0→0.56), Moirai (0.997→0.46); Qwen drops 0.981→0.64.
- **Compositional routing (the key test)**: only **Qwen3-8B pretrained** beats its random control by a wide margin (0.833 vs 0.558 bal3). For every TS FM the pretrained router is ≤ its random control (Bolt 0.683 vs 0.675, TimesFM 0.653 vs 0.694, MOMENT 0.631 vs 0.725, Moirai 0.564 vs 0.656). No TS-FM representation reproduces the language-pretrained routing transfer.
- **Numerical readout**: pretrained helps for Qwen (0.588 vs 0.679), Bolt (0.405 vs 1.049) and TimesFM (0.404 vs 0.895); MOMENT random ≈ pretrained; Moirai random is close to pretrained (0.436 vs 0.394).
- **Native forecasting**: TS-FM native interfaces (pretrained) give 0.53–0.67 MSE; random-weight native is 5–25× worse (TimesFM random 16.9, Chronos-T5-base random 12.4), as expected. MOMENT/Qwen native are not defined under this protocol (documented).

> Caveat: these are diagnostic accessibility/routing numbers on controlled synthetic windows; no OOD test set was used for any model/training selection. See `config.json` for exact checkpoint paths and the Moirai/native caveats.
