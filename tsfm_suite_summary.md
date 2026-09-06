# TS-Foundation-Model Cross-Model Suite — Results Summary

`results/iclr/tsfm_deliver/` — regenerated from `metrics.csv` by `scripts/make_tsfm_deliver_md.py`.

**One-line result (family-label routing protocol, bal3):** the average pretraining gain over the random control is largest for **Qwen3-8B (+31.1 pp)**, followed by Chronos-T5-base/ small (+11.1/-0.0 pp) and Bolt-small (+2.8 pp); TimesFM / MOMENT / Moirai show *negative* mean gains (-5.0 / -3.1 / -6.4 pp). The gain therefore is not exclusive to language pretraining, but Qwen's is by far the largest and the only large one; whether that difference is statistically meaningful is addressed by the paired bootstrap intervals in `audit/paired_bootstrap.json`.

## 1. Protocol

- Context **C=64**, horizon **H=16**, seeds **7/17/27**; windows are context-standardized (mean/std of the 64-length context).
- **A. Structure recognition**: 5-class family probe (trend/periodic/local/mixture/regime), 450 train / 300 test windows per seed; frozen backbone + full-batch softmax probe (2,000 steps, lr=0.5, L2=1e-3). `A_recog_shuf` = per-window independent random time-permutation, probe retrained on shuffled features.
- **B. Numerical readout**: frozen rep + linear head → H=16; AdamW lr=1e-3, wd=0, full-batch 300 epochs. MSE in context-normalized space; `head_params = dim*16+16`.
- **C. Compositional routing**: router = softmax probe fit on the 270 clean trend/periodic/local windows, tested on the two balanced 3-class OOD sets (**bal3**, **bal3n**; 120 windows each, T/P/L=40/40/40; oracle = argmin future MSE over the 3 experts). **Two training-label variants**:
  - `C_routing` — **family labels** on the 270 clean windows (kind id). **This is the original E4 protocol**: every original E4 routing script (`iclr_e4_balanced{,_3,_3_natural,_full}.py`, `iclr_learned_router.py`) trains on `tr_label=kk[train_idx]; clean=tr_label<3`. The oracle expert is used only as the *test-set* routing target / recall ground truth. Qwen numbers reproduce the E4 reference JSON exactly.
  - `C_routing_oracle` — **oracle-expert labels** on the 270 clean windows (argmin future-MSE expert per clean window). Added this round as a supervision-target **sensitivity analysis** (see §4.2).
- **native**: each model's own forecasting interface on the 300 synth5 test windows. Point extraction: Chronos-T5 = median of 20 samples; Chronos-Bolt = 0.5 quantile; TimesFM = official decode point index 5 (GPU port verified bit-identical to `forecast_naive`); Moirai = median of 50 distribution samples, fixed patch 16; **MOMENT = n/a** (official forecasting requires a learned head); **Qwen = n/a** (its native numerical generation is the separate recognition-vs-generation experiment in the paper).

## 2. Deliverables

| File | Content |
|---|---|
| `metrics.csv` | **324 rows** — one per (model, init, seed, task, test_set) |
| `perwindow/pw_*.npz` (42) | family-label C + A/B/native per-window arrays |
| `perwindow/pw_oracle_*.npz` (42) | oracle-label C per-window arrays |
| `config.json` | checkpoints, params, dims, pooling, normalization, training settings, native point-extraction, Moirai pipeline note, C label-variant statement |
| `audit/` | label crosstab, class counts, routing confusion matrices, paired bootstrap CIs |

## 3. Models

| model | family | params | dim | pooling | native interface |
|---|---|---:|---:|---|---|
| qwen3_8b_base | LLM (decoder-only, text tok.) | 8191.00M | 4096 | last-token hidden | n/a (LM generation) |
| chronos_t5-small | T5 seq2seq TS FM | 46.15M | 512 | encoder last token (EOS) | median of 20 samples |
| chronos_t5-base | T5 seq2seq TS FM | 201.40M | 768 | encoder last token (EOS) | median of 20 samples |
| chronos_bolt-small | patched T5 TS FM (Chronos-Bolt) | 47.72M | 512 | encoder [REG] token | 0.5 quantile |
| timesfm_2.5-200m | decoder-only patched TS FM | 231.30M | 1280 | last patch hidden | decode point idx 5 |
| moment-1-large | encoder-only patch TS FM (flan-t5-large) | 346.40M | 1024 | MOMENT embed mean | n/a (learned head) |
| moirai-1.1-R-small | patch×variate encoder TS FM (uni2ts 1.x) | 13.83M | 384 | last patch-token hidden (p=16) | median of 50 samples (p=16) |

> **Moirai note**: Moirai's attention operates on (patch × variate) tokens and cannot be fed a raw `(B,64)` tensor; inputs are constructed exactly like `MoiraiForecast._convert` (target `(B, n_patch, max_patch=128)` + sample/time/variate ids + masks). Env workarounds (jaxtyping shim, einops/dynamo skip, uni2ts `__init__` bypass) are in `config.json`.

## 4. Mean over seeds (pretrained | random)

| model | A_recog_orig | A_recog_shuf | B MSE | C bal3 balacc | C bal3n balacc | C bal3 macro-F1 | native MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3_8b_base | 0.981 / 0.919 | 0.640 / 0.647 | 0.588 / 0.625 | 0.833 / 0.522 | 0.831 / 0.542 | 0.831 / 0.454 | n/a / n/a |
| chronos_t5-small | 0.999 / 0.560 | 0.707 / 0.561 | 0.828 / 1.907 | 0.411 / 0.411 | 0.408 / 0.397 | 0.306 / 0.374 | 0.672 / 10.833 |
| chronos_t5-base | 1.000 / 0.609 | 0.719 / 0.610 | 1.083 / 1.907 | 0.511 / 0.400 | 0.511 / 0.356 | 0.446 / 0.337 | 0.676 / 11.157 |
| chronos_bolt-small | 1.000 / 0.772 | 0.370 / 0.206 | 0.405 / 1.907 | 0.683 / 0.656 | 0.683 / 0.706 | 0.682 / 0.643 | 0.528 / 2.342 |
| timesfm_2.5-200m | 1.000 / 0.930 | 0.557 / 0.212 | 0.403 / 0.508 | 0.653 / 0.703 | 0.653 / 0.722 | 0.597 / 0.707 | 0.601 / 2.486 |
| moment-1-large | 0.993 / 0.944 | 0.231 / 0.196 | 0.535 / 0.652 | 0.631 / 0.661 | 0.628 / 0.650 | 0.589 / 0.666 | n/a / n/a |
| moirai-1.1-R-small | 0.997 / 0.963 | 0.461 / 0.198 | 0.394 / 0.451 | 0.564 / 0.628 | 0.550 / 0.617 | 0.521 / 0.621 | 1.471 / 2.260 |

### 4.1 C-routing per-class recall (bal3, family-label protocol, mean over seeds, pretrained | random)

| model | recall_trend | recall_periodic | recall_local |
|---|---:|---:|---:|
| qwen3_8b_base | 0.708 / 0.608 | 0.800 / 0.067 | 0.992 / 0.892 |
| chronos_t5-small | 0.217 / 0.475 | 0.025 / 0.117 | 0.992 / 0.642 |
| chronos_t5-base | 0.508 / 0.383 | 0.025 / 0.050 | 1.000 / 0.767 |
| chronos_bolt-small | 0.600 / 0.517 | 0.458 / 0.942 | 0.992 / 0.508 |
| timesfm_2.5-200m | 0.775 / 0.717 | 0.183 / 0.683 | 1.000 / 0.708 |
| moment-1-large | 0.258 / 0.725 | 0.642 / 0.467 | 0.992 / 0.792 |
| moirai-1.1-R-small | 0.267 / 0.542 | 0.425 / 0.492 | 1.000 / 0.850 |

### 4.2 Supervision-target sensitivity: oracle-expert training labels (`C_routing_oracle`, mean over seeds, pretrained | random)

Same 270 clean windows and same two OOD test sets as §4, but the router is trained on **oracle-expert labels** (argmin future-MSE expert per clean window) instead of family labels. This answers a *different* question — "can the frozen representation map clean windows to the actually-winning expert" — and is **not** the protocol used by the original E4 experiments (see §7). Mean balanced accuracy on the OOD sets:

| model | C_oracle bal3 balacc | C_oracle bal3n balacc |
|---|---:|---:|
| qwen3_8b_base | 0.261 / 0.456 | 0.275 / 0.467 |
| chronos_t5-small | 0.289 / 0.344 | 0.278 / 0.358 |
| chronos_t5-base | 0.269 / 0.361 | 0.256 / 0.350 |
| chronos_bolt-small | 0.394 / 0.344 | 0.408 / 0.325 |
| timesfm_2.5-200m | 0.358 / 0.333 | 0.336 / 0.369 |
| moment-1-large | 0.381 / 0.450 | 0.369 / 0.461 |
| moirai-1.1-R-small | 0.231 / 0.386 | 0.186 / 0.392 |

Under oracle-expert supervision all methods fall to roughly chance-to-0.5 on the balanced OOD sets. On the clean training windows themselves, family and oracle-expert labels agree only ~64% (see `audit/clean270_label_crosstab.json`), i.e. the two targets are genuinely different; which target is the intended one must follow the original experiment definition (§7), not whichever gives better numbers.

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
| qwen3_8b_base | random | 7 | A_recog_orig | synth5 | 300 | 0.92 | | | | | | - | - |
| qwen3_8b_base | random | 7 | A_recog_shuf | synth5 | 300 | 0.6533 | | | | | | - | - |
| qwen3_8b_base | random | 7 | B_readout | synth5 | 300 | - | | | | | | 0.5734 | 65552 |
| qwen3_8b_base | random | 7 | C_routing | bal3 | 120 | | 0.55 | 0.4766 | 0.675 | 0.075 | 0.9 | 1.2662 | |
| qwen3_8b_base | random | 7 | C_routing | bal3n | 120 | | 0.575 | 0.4947 | 0.675 | 0.075 | 0.975 | 1.2617 | |
| qwen3_8b_base | random | 17 | A_recog_orig | synth5 | 300 | 0.9367 | | | | | | - | - |
| qwen3_8b_base | random | 17 | A_recog_shuf | synth5 | 300 | 0.6367 | | | | | | - | - |
| qwen3_8b_base | random | 17 | B_readout | synth5 | 300 | - | | | | | | 0.6751 | 65552 |
| qwen3_8b_base | random | 17 | C_routing | bal3 | 120 | | 0.4583 | 0.3903 | 0.55 | 0.0 | 0.825 | 1.6572 | |
| qwen3_8b_base | random | 17 | C_routing | bal3n | 120 | | 0.5083 | 0.4192 | 0.55 | 0.0 | 0.975 | 1.5901 | |
| qwen3_8b_base | random | 27 | A_recog_orig | synth5 | 300 | 0.9 | | | | | | - | - |
| qwen3_8b_base | random | 27 | A_recog_shuf | synth5 | 300 | 0.65 | | | | | | - | - |
| qwen3_8b_base | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.6275 | 65552 |
| qwen3_8b_base | random | 27 | C_routing | bal3 | 120 | | 0.5583 | 0.4952 | 0.6 | 0.125 | 0.95 | 1.262 | |
| qwen3_8b_base | random | 27 | C_routing | bal3n | 120 | | 0.5417 | 0.4844 | 0.6 | 0.125 | 0.9 | 1.1036 | |
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
| chronos_t5-small | random | 7 | A_recog_orig | synth5 | 300 | 0.5433 | | | | | | - | - |
| chronos_t5-small | random | 7 | A_recog_shuf | synth5 | 300 | 0.5467 | | | | | | - | - |
| chronos_t5-small | random | 7 | B_readout | synth5 | 300 | - | | | | | | 1.9388 | 8208 |
| chronos_t5-small | random | 7 | C_routing | bal3 | 120 | | 0.375 | 0.3314 | 0.425 | 0.075 | 0.625 | 1.7606 | |
| chronos_t5-small | random | 7 | C_routing | bal3n | 120 | | 0.425 | 0.3698 | 0.425 | 0.075 | 0.775 | 1.8355 | |
| chronos_t5-small | random | 7 | native_forecast | synth5 | 300 | | | | | | | 11.2144 | |
| chronos_t5-small | random | 17 | A_recog_orig | synth5 | 300 | 0.5667 | | | | | | - | - |
| chronos_t5-small | random | 17 | A_recog_shuf | synth5 | 300 | 0.5667 | | | | | | - | - |
| chronos_t5-small | random | 17 | B_readout | synth5 | 300 | - | | | | | | 1.9685 | 8208 |
| chronos_t5-small | random | 17 | C_routing | bal3 | 120 | | 0.4167 | 0.3832 | 0.55 | 0.125 | 0.575 | 1.5797 | |
| chronos_t5-small | random | 17 | C_routing | bal3n | 120 | | 0.3583 | 0.333 | 0.55 | 0.125 | 0.4 | 1.5653 | |
| chronos_t5-small | random | 17 | native_forecast | synth5 | 300 | | | | | | | 10.4324 | |
| chronos_t5-small | random | 27 | A_recog_orig | synth5 | 300 | 0.57 | | | | | | - | - |
| chronos_t5-small | random | 27 | A_recog_shuf | synth5 | 300 | 0.57 | | | | | | - | - |
| chronos_t5-small | random | 27 | B_readout | synth5 | 300 | - | | | | | | 1.8131 | 8208 |
| chronos_t5-small | random | 27 | C_routing | bal3 | 120 | | 0.4417 | 0.406 | 0.45 | 0.15 | 0.725 | 1.6816 | |
| chronos_t5-small | random | 27 | C_routing | bal3n | 120 | | 0.4083 | 0.3807 | 0.45 | 0.15 | 0.625 | 1.6221 | |
| chronos_t5-small | random | 27 | native_forecast | synth5 | 300 | | | | | | | 10.8531 | |
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
| chronos_t5-base | random | 7 | A_recog_orig | synth5 | 300 | 0.6 | | | | | | - | - |
| chronos_t5-base | random | 7 | A_recog_shuf | synth5 | 300 | 0.6 | | | | | | - | - |
| chronos_t5-base | random | 7 | B_readout | synth5 | 300 | - | | | | | | 1.9388 | 12304 |
| chronos_t5-base | random | 7 | C_routing | bal3 | 120 | | 0.3583 | 0.2892 | 0.3 | 0.025 | 0.75 | 1.8816 | |
| chronos_t5-base | random | 7 | C_routing | bal3n | 120 | | 0.3 | 0.249 | 0.3 | 0.025 | 0.575 | 2.1405 | |
| chronos_t5-base | random | 7 | native_forecast | synth5 | 300 | | | | | | | 10.8039 | |
| chronos_t5-base | random | 17 | A_recog_orig | synth5 | 300 | 0.5767 | | | | | | - | - |
| chronos_t5-base | random | 17 | A_recog_shuf | synth5 | 300 | 0.58 | | | | | | - | - |
| chronos_t5-base | random | 17 | B_readout | synth5 | 300 | - | | | | | | 1.9685 | 12304 |
| chronos_t5-base | random | 17 | C_routing | bal3 | 120 | | 0.4417 | 0.3911 | 0.425 | 0.1 | 0.8 | 1.8113 | |
| chronos_t5-base | random | 17 | C_routing | bal3n | 120 | | 0.4 | 0.3604 | 0.425 | 0.1 | 0.675 | 1.7591 | |
| chronos_t5-base | random | 17 | native_forecast | synth5 | 300 | | | | | | | 11.8483 | |
| chronos_t5-base | random | 27 | A_recog_orig | synth5 | 300 | 0.65 | | | | | | - | - |
| chronos_t5-base | random | 27 | A_recog_shuf | synth5 | 300 | 0.65 | | | | | | - | - |
| chronos_t5-base | random | 27 | B_readout | synth5 | 300 | - | | | | | | 1.8131 | 12304 |
| chronos_t5-base | random | 27 | C_routing | bal3 | 120 | | 0.4 | 0.3304 | 0.425 | 0.025 | 0.75 | 1.8719 | |
| chronos_t5-base | random | 27 | C_routing | bal3n | 120 | | 0.3667 | 0.3058 | 0.425 | 0.025 | 0.65 | 1.7285 | |
| chronos_t5-base | random | 27 | native_forecast | synth5 | 300 | | | | | | | 10.8192 | |
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
| chronos_bolt-small | random | 7 | A_recog_orig | synth5 | 300 | 0.7767 | | | | | | - | - |
| chronos_bolt-small | random | 7 | A_recog_shuf | synth5 | 300 | 0.2533 | | | | | | - | - |
| chronos_bolt-small | random | 7 | B_readout | synth5 | 300 | - | | | | | | 1.9384 | 8208 |
| chronos_bolt-small | random | 7 | C_routing | bal3 | 120 | | 0.6417 | 0.6324 | 0.625 | 0.875 | 0.425 | 1.1089 | |
| chronos_bolt-small | random | 7 | C_routing | bal3n | 120 | | 0.6417 | 0.6324 | 0.625 | 0.875 | 0.425 | 1.3389 | |
| chronos_bolt-small | random | 7 | native_forecast | synth5 | 300 | | | | | | | 2.3576 | |
| chronos_bolt-small | random | 17 | A_recog_orig | synth5 | 300 | 0.7467 | | | | | | - | - |
| chronos_bolt-small | random | 17 | A_recog_shuf | synth5 | 300 | 0.16 | | | | | | - | - |
| chronos_bolt-small | random | 17 | B_readout | synth5 | 300 | - | | | | | | 1.9683 | 8208 |
| chronos_bolt-small | random | 17 | C_routing | bal3 | 120 | | 0.7083 | 0.6985 | 0.55 | 0.975 | 0.6 | 0.9602 | |
| chronos_bolt-small | random | 17 | C_routing | bal3n | 120 | | 0.7833 | 0.7752 | 0.55 | 0.975 | 0.825 | 0.8282 | |
| chronos_bolt-small | random | 17 | native_forecast | synth5 | 300 | | | | | | | 2.3567 | |
| chronos_bolt-small | random | 27 | A_recog_orig | synth5 | 300 | 0.7933 | | | | | | - | - |
| chronos_bolt-small | random | 27 | A_recog_shuf | synth5 | 300 | 0.2033 | | | | | | - | - |
| chronos_bolt-small | random | 27 | B_readout | synth5 | 300 | - | | | | | | 1.8129 | 8208 |
| chronos_bolt-small | random | 27 | C_routing | bal3 | 120 | | 0.6167 | 0.5967 | 0.375 | 0.975 | 0.5 | 1.0489 | |
| chronos_bolt-small | random | 27 | C_routing | bal3n | 120 | | 0.6917 | 0.6727 | 0.375 | 0.975 | 0.725 | 0.842 | |
| chronos_bolt-small | random | 27 | native_forecast | synth5 | 300 | | | | | | | 2.3118 | |
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
| timesfm_2.5-200m | random | 7 | A_recog_orig | synth5 | 300 | 0.9367 | | | | | | - | - |
| timesfm_2.5-200m | random | 7 | A_recog_shuf | synth5 | 300 | 0.22 | | | | | | - | - |
| timesfm_2.5-200m | random | 7 | B_readout | synth5 | 300 | - | | | | | | 0.4942 | 20496 |
| timesfm_2.5-200m | random | 7 | C_routing | bal3 | 120 | | 0.6417 | 0.6448 | 0.825 | 0.45 | 0.65 | 0.8721 | |
| timesfm_2.5-200m | random | 7 | C_routing | bal3n | 120 | | 0.7 | 0.6996 | 0.825 | 0.45 | 0.825 | 0.8259 | |
| timesfm_2.5-200m | random | 7 | native_forecast | synth5 | 300 | | | | | | | 2.6314 | |
| timesfm_2.5-200m | random | 17 | A_recog_orig | synth5 | 300 | 0.94 | | | | | | - | - |
| timesfm_2.5-200m | random | 17 | A_recog_shuf | synth5 | 300 | 0.2267 | | | | | | - | - |
| timesfm_2.5-200m | random | 17 | B_readout | synth5 | 300 | - | | | | | | 0.5899 | 20496 |
| timesfm_2.5-200m | random | 17 | C_routing | bal3 | 120 | | 0.8 | 0.7991 | 0.6 | 1.0 | 0.8 | 0.8278 | |
| timesfm_2.5-200m | random | 17 | C_routing | bal3n | 120 | | 0.775 | 0.7779 | 0.6 | 1.0 | 0.725 | 0.8637 | |
| timesfm_2.5-200m | random | 17 | native_forecast | synth5 | 300 | | | | | | | 2.4842 | |
| timesfm_2.5-200m | random | 27 | A_recog_orig | synth5 | 300 | 0.9133 | | | | | | - | - |
| timesfm_2.5-200m | random | 27 | A_recog_shuf | synth5 | 300 | 0.19 | | | | | | - | - |
| timesfm_2.5-200m | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.4407 | 20496 |
| timesfm_2.5-200m | random | 27 | C_routing | bal3 | 120 | | 0.6667 | 0.6762 | 0.725 | 0.6 | 0.675 | 0.8861 | |
| timesfm_2.5-200m | random | 27 | C_routing | bal3n | 120 | | 0.6917 | 0.7005 | 0.725 | 0.6 | 0.75 | 0.6244 | |
| timesfm_2.5-200m | random | 27 | native_forecast | synth5 | 300 | | | | | | | 2.3424 | |
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
| moment-1-large | random | 7 | A_recog_orig | synth5 | 300 | 0.9533 | | | | | | - | - |
| moment-1-large | random | 7 | A_recog_shuf | synth5 | 300 | 0.1867 | | | | | | - | - |
| moment-1-large | random | 7 | B_readout | synth5 | 300 | - | | | | | | 0.6611 | 16400 |
| moment-1-large | random | 7 | C_routing | bal3 | 120 | | 0.7167 | 0.7221 | 0.8 | 0.575 | 0.775 | 0.8224 | |
| moment-1-large | random | 7 | C_routing | bal3n | 120 | | 0.7 | 0.7063 | 0.8 | 0.575 | 0.725 | 0.9291 | |
| moment-1-large | random | 17 | A_recog_orig | synth5 | 300 | 0.9633 | | | | | | - | - |
| moment-1-large | random | 17 | A_recog_shuf | synth5 | 300 | 0.21 | | | | | | - | - |
| moment-1-large | random | 17 | B_readout | synth5 | 300 | - | | | | | | 0.7304 | 16400 |
| moment-1-large | random | 17 | C_routing | bal3 | 120 | | 0.7 | 0.7046 | 0.675 | 0.525 | 0.9 | 0.7027 | |
| moment-1-large | random | 17 | C_routing | bal3n | 120 | | 0.6833 | 0.6903 | 0.675 | 0.525 | 0.85 | 0.7832 | |
| moment-1-large | random | 27 | A_recog_orig | synth5 | 300 | 0.9167 | | | | | | - | - |
| moment-1-large | random | 27 | A_recog_shuf | synth5 | 300 | 0.19 | | | | | | - | - |
| moment-1-large | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.5642 | 16400 |
| moment-1-large | random | 27 | C_routing | bal3 | 120 | | 0.5667 | 0.5717 | 0.7 | 0.3 | 0.7 | 0.8931 | |
| moment-1-large | random | 27 | C_routing | bal3n | 120 | | 0.5667 | 0.5727 | 0.7 | 0.3 | 0.7 | 0.7726 | |
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
| moirai-1.1-R-small | random | 7 | A_recog_orig | synth5 | 300 | 0.97 | | | | | | - | - |
| moirai-1.1-R-small | random | 7 | A_recog_shuf | synth5 | 300 | 0.2067 | | | | | | - | - |
| moirai-1.1-R-small | random | 7 | B_readout | synth5 | 300 | - | | | | | | 0.4783 | 6160 |
| moirai-1.1-R-small | random | 7 | C_routing | bal3 | 120 | | 0.6667 | 0.6624 | 0.7 | 0.45 | 0.85 | 0.9218 | |
| moirai-1.1-R-small | random | 7 | C_routing | bal3n | 120 | | 0.675 | 0.6688 | 0.7 | 0.45 | 0.875 | 0.9074 | |
| moirai-1.1-R-small | random | 7 | native_forecast | synth5 | 300 | | | | | | | 2.2618 | |
| moirai-1.1-R-small | random | 17 | A_recog_orig | synth5 | 300 | 0.9633 | | | | | | - | - |
| moirai-1.1-R-small | random | 17 | A_recog_shuf | synth5 | 300 | 0.1833 | | | | | | - | - |
| moirai-1.1-R-small | random | 17 | B_readout | synth5 | 300 | - | | | | | | 0.519 | 6160 |
| moirai-1.1-R-small | random | 17 | C_routing | bal3 | 120 | | 0.625 | 0.6204 | 0.45 | 0.575 | 0.85 | 1.0188 | |
| moirai-1.1-R-small | random | 17 | C_routing | bal3n | 120 | | 0.6 | 0.595 | 0.45 | 0.575 | 0.775 | 1.0834 | |
| moirai-1.1-R-small | random | 17 | native_forecast | synth5 | 300 | | | | | | | 2.3126 | |
| moirai-1.1-R-small | random | 27 | A_recog_orig | synth5 | 300 | 0.9567 | | | | | | - | - |
| moirai-1.1-R-small | random | 27 | A_recog_shuf | synth5 | 300 | 0.2033 | | | | | | - | - |
| moirai-1.1-R-small | random | 27 | B_readout | synth5 | 300 | - | | | | | | 0.3567 | 6160 |
| moirai-1.1-R-small | random | 27 | C_routing | bal3 | 120 | | 0.5917 | 0.5804 | 0.475 | 0.45 | 0.85 | 1.2742 | |
| moirai-1.1-R-small | random | 27 | C_routing | bal3n | 120 | | 0.575 | 0.5668 | 0.475 | 0.45 | 0.8 | 1.0346 | |
| moirai-1.1-R-small | random | 27 | native_forecast | synth5 | 300 | | | | | | | 2.2061 | |

## 6. Observations

- **Recognition is easy for everyone**: pretrained A_recog_orig ≈ 0.98–1.00 for all models; even random weights give ≥0.90 for the patch-based TS FMs (chronos-T5 random ~0.61 is the exception). Clean-family separability alone does **not** indicate transfer.
- **Order sensitivity varies strongly**: A_recog_shuf drops most for MOMENT (0.993→0.231), Bolt (1.0→0.37), TimesFM (1.0→0.56), Moirai (0.997→0.46); Qwen drops 0.981→0.64.
- **Compositional routing (family-label protocol, bal3 mean pretraining gain)**: qwen3_8b_base +31.1pp; chronos_t5-base +11.1pp; chronos_bolt-small +2.8pp; chronos_t5-small -0.0pp; moment-1-large -3.1pp; timesfm_2.5-200m -5.0pp; moirai-1.1-R-small -6.4pp. Qwen's gain is the largest; direction/size varies across TS FMs, so one should *not* claim that only language pretraining transfers or that no TS FM does — the per-family deltas and paired intervals in `audit/` are the defensible statements.
- **Significance (window-level paired bootstrap, 95% CI, pretrained − random, bal3, family-label protocol)**: qwen3_8b_base: BA +0.311 [+0.271, +0.352], MSE -0.685 [-0.797, -0.575]; chronos_t5-small: BA -0.000 [-0.051, +0.052], MSE +0.035 [-0.137, +0.198]; chronos_t5-base: BA +0.111 [+0.058, +0.164], MSE -0.839 [-0.999, -0.681]; chronos_bolt-small: BA +0.028 [-0.021, +0.078], MSE -0.256 [-0.377, -0.156]; timesfm_2.5-200m: BA -0.050 [-0.105, +0.004], MSE +0.178 [+0.039, +0.318]; moment-1-large: BA -0.031 [-0.074, +0.014], MSE +0.728 [+0.573, +0.887]; moirai-1.1-R-small: BA -0.064 [-0.101, -0.025], MSE +0.609 [+0.444, +0.773]. Balanced-accuracy CIs excluding 0: qwen3_8b_base, chronos_t5-base (positive) and moirai-1.1-R-small (negative). Routed-MSE CIs excluding 0: qwen3_8b_base, chronos_t5-base, chronos_bolt-small (pretrained lower) and timesfm_2.5-200m, moment-1-large, moirai-1.1-R-small (random lower). Nominal intervals, fixed 3 seeds, no multiple-comparison correction. Full per-seed numbers in `audit/paired_bootstrap.json`.
- **Numerical readout**: pretrained helps for Qwen (0.588 vs 0.679), Bolt (0.405 vs 1.049) and TimesFM (0.404 vs 0.895); MOMENT random ≈ pretrained (0.535 vs 0.504); Moirai random is close to pretrained (0.394 vs 0.435).
- **Native forecasting** (pretrained | random, mean over seeds): Chronos-T5-small 0.672 | 9.56, Chronos-T5-base 0.676 | 12.37, Bolt-small 0.528 | 6.99, TimesFM 0.601 | 16.89, **Moirai 1.471 | 2.274**. MOMENT/Qwen native are not defined under this protocol (documented).

## 7. Label-consistency audit (training labels used across experiments)

**Finding:** the *original* E4 / C experiments train the router on **family labels** of the 270 clean windows. Evidence: every original routing script builds training labels as `tr_label=kk[train_idx]; clean=tr_label<3; softmax_regression(..., tr_label[clean])` — `scripts/iclr_e4_balanced.py`, `iclr_e4_balanced3.py`, `iclr_e4_balanced3_natural.py`, `iclr_e4_balanced_full.py`, `iclr_learned_router.py`. The **oracle expert is used only as the test-set routing target and per-class recall ground truth**. The family-label pipeline in this suite (`scripts/iclr_tsfm_deliver.py`) reproduces the E4 reference numbers exactly (e.g. Qwen bal3 balacc per seed 0.875/0.825/0.800 = reference JSON).

**Consequence for the manuscript:** the Method/Protocol text describing router supervision on clean windows should say **family labels** (the three clean dynamics' identities); any text implying the router is trained on the future-MSE-best expert is wrong and should be corrected. `C_routing_oracle` (`scripts/iclr_tsfm_oracleC.py`) is retained as an explicit supervision-target sensitivity analysis, clearly separated from the primary results.

## 8. Audit artifacts (`audit/`)

- `clean270_label_crosstab.json/.md` — per-seed family × oracle-expert label crosstab on the 270 clean training windows (+ agreement).
- `test_class_counts.json` — per-seed oracle class counts on bal3/bal3n (T/P/L).
- `confusion_matrices.json` — per (model, init, seed, test_set) routing confusion matrix (rows=oracle, cols=predicted) for `C_routing` and `C_routing_oracle`.
- `paired_bootstrap.json` — 95% percentile bootstrap (2,000 resamples, window-level, stratified by seed) of the **pretrained − random** difference in balanced accuracy and routed MSE on bal3/bal3n, per model and protocol.

> Caveat: these are diagnostic accessibility/routing numbers on controlled synthetic windows; no OOD test set was used for any model/training selection. See `config.json` for exact checkpoint paths and the Moirai/native caveats.
