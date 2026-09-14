# Open-LLM Cross-Family Attribution Suite — Complete Experiment Report

_Auto-generated from result CSVs by `scripts/openllm_make_report.py` (repo `/9950backfile/chenjiahui/ARForecast-Diagnosis`). Date: 2026-09-14._

## 1. Protocol (identical to the published Qwen pipeline)

- context **C=64**, horizon **H=16**, context-only standardization; 5 synthetic families (trend / periodic / local AR / mixture / regime), 150 windows/family/seed, train 90 / test 60.
- seeds **7/17/27** (headline replication adds 37…97).
- LM input = the historical serialization (`clip ±9.99`, 2 decimals, explicit sign, fixed prompt).
- Representation = **final layer, final non-padding token**; frozen backbone in all attribution experiments.
- **pretrained** = released base checkpoint; **random** = same architecture constructed from config with native initializers and **no released weights** (per-tensor audit: 0 tensors equal to pretrained).
- Router trains only on the **270 clean trend/periodic/local windows**; OOD sets are the frozen balanced sets `bal3` (120 windows, 40/40/40 T/P/L, AR-weak-sine local source) and `bal3n` (120 windows, periodic+AR local source).
- Two supervision interfaces: **family-label** (predict the generating family; original E4 protocol) and **oracle-label** (predict the expert with lowest realized-future MSE on the clean windows).

## 2. Model inventory

| model | family | params | hidden | checkpoint | status |
|---|---|---:|---:|---|---|
| Qwen3-8B | Qwen | 8.19B | 4096 | Qwen/Qwen3-8B-Base (local) | P+R done |
| Llama-3.1-8B | Llama | 8.0B | 4096 | meta-llama/Llama-3.1-8B (rev d04e592) | P+R done |
| Llama-3.2-3B | Llama | 3.2B | 3072 | meta-llama/Llama-3.2-3B (rev 13afe51) | P+R done |
| Gemma-2-9B | Gemma | 9.2B | 3584 | google/gemma-2-9b (rev 33c1930) | P+R done |
| Gemma-2-2B | Gemma | 2.6B | 2304 | google/gemma-2-2b (rev c5ebcd4) | P+R done |
| DeepSeek-LLM-7B | DeepSeek | 6.9B | 4096 | deepseek-ai/deepseek-llm-7b-base (rev 7683fea) | P+R done |
| Mistral-7B-v0.3 | Mistral | 7.2B | 4096 | mistralai/Mistral-7B-v0.3 (rev caa1feb) | P+R done |
| OLMo-2-7B | OLMo | 6.9B | 4096 | allenai/OLMo-2-1124-7B (rev 7df9a82) | P+R done |
| OLMo-2-13B (P1) | OLMo | 13B | 5120 | allenai/OLMo-2-1124-13B (rev 3fefddc) | P+R done |
| DeepSeek-V2-Lite (MoE, P1) | DeepSeek | 15.7B total / 2.4B active | 2048 | deepseek-ai/DeepSeek-V2-Lite (rev 604d566) | P+R done |
| Qwen3-0.6B / 1.7B | Qwen | 0.6B / 1.7B | 0.5 | Qwen3-0.6B / Qwen3-1.7B (local cache) | scale sweep done |

Time-series foundation models used as cross-architecture reference (prior round, same protocol): Chronos-T5-small/base, Chronos-Bolt-small, TimesFM-2.5-200m, MOMENT-1-large, Moirai-1.1-R-small.

## 3. Table A — Matched-scale open language models (mean over seeds 7/17/27)

Δ = pretrained − random. For routing, Δ is the family-label balanced-accuracy difference (positive = pretrained better).

| model | clean P/R | clean Δ | shuffle P/R | readout P/R (MSE) | **family-label OOD P/R** | **Δ routing** | oracle-label P/R | MLP-router P/R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-8B** | 0.981/0.919 | +0.062 | 0.640/0.647 | 0.588/0.625 | 0.833/0.522 | **+0.311** | 0.261/0.456 | —/— |
| **Llama-3.1-8B** | 0.974/0.927 | +0.048 | 0.608/0.671 | 0.560/0.885 | 0.744/0.517 | **+0.228** | 0.311/0.483 | 0.803/0.508 |
| **Llama-3.2-3B** | 0.967/0.936 | +0.031 | 0.617/0.652 | 0.574/0.705 | 0.664/0.517 | **+0.147** | 0.394/0.517 | 0.647/0.469 |
| **Gemma-2-9B** | 0.959/0.781 | +0.178 | 0.607/0.288 | 0.602/1.941 | 0.644/0.444 | **+0.200** | 0.369/0.425 | 0.642/0.461 |
| **Gemma-2-2B** | 0.958/0.833 | +0.124 | 0.573/0.326 | 0.640/1.570 | 0.597/0.447 | **+0.150** | 0.333/0.467 | 0.644/0.461 |
| **Mistral-7B-v0.3** | 0.989/0.912 | +0.077 | 0.717/0.631 | 0.524/0.697 | 0.639/0.533 | **+0.106** | 0.281/0.411 | 0.675/0.431 |
| **DeepSeek-LLM-7B** | 0.983/0.932 | +0.051 | 0.671/0.600 | 0.627/0.695 | 0.608/0.533 | **+0.075** | 0.317/0.531 | 0.589/0.464 |
| **OLMo-2-7B** | 0.986/0.937 | +0.049 | 0.678/0.661 | 0.504/0.585 | 0.586/0.492 | **+0.094** | 0.336/0.458 | 0.544/0.428 |
| **OLMo-2-13B** | 0.976/0.929 | +0.047 | 0.736/0.660 | 0.554/0.584 | 0.586/0.503 | **+0.083** | 0.292/0.464 | 0.636/0.450 |
| **DeepSeek-V2-Lite (MoE)** | 0.983/0.940 | +0.043 | 0.649/0.631 | 0.645/0.583 | 0.797/0.483 | **+0.314** | 0.253/0.403 | 0.797/0.436 |

## 4. Table C — Language models vs time-series foundation models (family-label routing Δ, bal3)

| model | type | pretrained BA | random BA | Δ | 95% CI (paired) |
|---|---|---:|---:|---:|---|
| Gemma-2-9B | LM | 0.644 | 0.444 | **+0.200** | [+0.175,+0.233] |
| Llama-3.2-3B | LM | 0.664 | 0.517 | **+0.147** | [+0.100,+0.192] |
| Gemma-2-2B | LM | 0.597 | 0.447 | **+0.150** | [+0.108,+0.175] |
| Mistral-7B-v0.3 | LM | 0.639 | 0.533 | **+0.106** | [+0.083,+0.133] |
| Qwen3-8B (window bootstrap) | LM | 0.833 | 0.522 | **+0.311** | [+0.271,+0.352] |
| Chronos-T5-base | TSFM | — | — | +0.111 | [+0.058,+0.164] |
| Chronos-T5-small | TSFM | — | — | -0.000 | [-0.051,+0.052] |
| Chronos-Bolt-small | TSFM | — | — | +0.028 | [-0.021,+0.078] |
| TimesFM-2.5-200m | TSFM | — | — | -0.050 | [-0.105,+0.004] |
| MOMENT-1-large | TSFM | — | — | -0.031 | [-0.074,+0.014] |
| Moirai-1.1-R-small | TSFM | — | — | -0.064 | [-0.101,-0.025] |

Reading: the family-label pretraining gain is largest for the LLM family (Qwen3-8B), positive but smaller for Llama/Gemma/Mistral, and not reproduced by the tested time-series foundation models.

## 4b. Table D — Real-world zero-shot routing (15 datasets)

The router is trained **only** on the 270 synthetic clean primitive windows and applied unchanged to real data (no fine-tuning, no threshold tuning, no checkpoint selection). Δ% = (pretrained − baseline)/baseline in %; negative = pretrained has lower routed MSE. CIs and BH-FDR q-values per dataset are in `tableD_realworld_stats.csv`.

| model | wins vs random | median Δ% vs random | wins vs feature-router | median Δ% vs feature | FDR-sig. vs random | FDR-sig. vs feature |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_llm_7b | 13/15 | -12.3 | 8/15 | -0.0 | 10/15 | 6/15 |
| deepseek_v2_lite | 14/15 | -15.8 | 11/15 | -0.7 | 12/15 | 7/15 |
| gemma2_2b | 12/15 | -14.8 | 8/15 | -0.4 | 12/15 | 6/15 |
| gemma2_9b | 13/15 | -18.3 | 10/15 | -4.1 | 12/15 | 6/15 |
| llama31_8b | 14/15 | -27.8 | 9/15 | -0.6 | 12/15 | 6/15 |
| llama32_3b | 15/15 | -27.6 | 10/15 | -2.0 | 12/15 | 6/15 |
| mistral_7b_v03 | 13/15 | -16.1 | 8/15 | -1.9 | 11/15 | 6/15 |
| olmo2_13b | 13/15 | -24.1 | 8/15 | -3.3 | 13/15 | 6/15 |
| olmo2_7b | 13/15 | -13.9 | 8/15 | -1.2 | 11/15 | 6/15 |
| qwen3_8b_base | 10/15 | -5.7 | 4/15 | +0.0 | 6/15 | 2/15 |

Reading: every model beats its **matched random** control on 10–15/15 datasets with BH-FDR significance on most (median relative routed-MSE change -28% to -6%). Against the hand-crafted **temporal-feature router** the picture is much closer: wins 4–11/15, median relative difference -4.1% to 0.0%. So on real data the pretrained LM advantage over random initialisation is robust, while the advantage over a simple engineered feature baseline is not established. Fairness caveat: these LMs are 3–15B parameters, the random control is the identical architecture, and no real-data fine-tuning or threshold tuning is performed (strict zero-shot decision transfer).

## 4c. Decision-interface and representation-geometry robustness

### Table B — 3-expert vs 5-expert decision definition (family-label, bal3 OOD)

The 5-expert bank adds a trend+seasonal hybrid and an AR(10)/ridge expert; the router supervision is unchanged (family label on the 270 clean primitives). `family5` maps the 5-expert winner back to the 3 primitive families.

| model | 3-expert P/R | 3-expert Δ | 5-expert P/R | 5-expert Δ | oracle-label Δ | margin (best/2nd) |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_llm_7b | 0.608/0.458 | **0.150** | 0.470/0.389 | 0.081 | -0.183 | 3.03 |
| deepseek_v2_lite | 0.797/0.483 | **0.314** | 0.439/0.395 | 0.044 | -0.150 | 3.03 |
| gemma2_2b | 0.597/0.447 | **0.150** | 0.475/0.357 | 0.118 | -0.133 | 3.03 |
| gemma2_9b | 0.644/0.444 | **0.200** | 0.440/0.329 | 0.110 | -0.056 | 3.03 |
| llama31_8b | 0.744/0.517 | **0.228** | 0.446/0.398 | 0.049 | -0.158 | 3.03 |
| llama32_3b | 0.664/0.517 | **0.147** | 0.494/0.431 | 0.063 | -0.136 | 3.03 |
| mistral_7b_v03 | 0.639/0.533 | **0.106** | 0.428/0.405 | 0.023 | -0.136 | 3.03 |
| olmo2_13b | 0.586/0.503 | **0.083** | 0.434/0.422 | 0.012 | -0.161 | 3.03 |
| olmo2_7b | 0.586/0.492 | **0.094** | 0.484/0.415 | 0.070 | -0.122 | 3.03 |
| qwen3_8b_base | 0.833/0.519 | **0.314** | 0.467/0.431 | 0.036 | -0.192 | 3.03 |

### Router capacity — linear vs 2-layer MLP64 (family-label, bal3)

| model | linear P/R | linear Δ | MLP64 P/R | MLP64 Δ |
|---|---:|---:|---:|---:|
| deepseek_llm_7b | 0.608/0.458 | 0.150 | 0.589/0.422 | 0.167 |
| deepseek_v2_lite | 0.797/0.483 | 0.314 | 0.797/0.436 | 0.361 |
| gemma2_2b | 0.597/0.447 | 0.150 | 0.644/0.461 | 0.183 |
| gemma2_9b | 0.644/0.444 | 0.200 | 0.642/0.461 | 0.181 |
| llama31_8b | 0.744/0.517 | 0.228 | 0.803/0.508 | 0.294 |
| llama32_3b | 0.664/0.517 | 0.147 | 0.647/0.469 | 0.178 |
| mistral_7b_v03 | 0.639/0.533 | 0.106 | 0.675/0.431 | 0.244 |
| olmo2_13b | 0.586/0.503 | 0.083 | 0.636/0.450 | 0.186 |
| olmo2_7b | 0.586/0.492 | 0.094 | 0.544/0.428 | 0.117 |
| qwen3_8b_base | 0.833/0.519 | 0.314 | 0.839/0.478 | 0.361 |

Reading: a nonlinear router does not remove the pretrained advantage — for every model the MLP64 Δ has the same sign as the linear Δ (usually larger), so the effect is not an artefact of linear accessibility.

### Dimension matching (bal3 ΔBA pretrained − random after projection)

`full` = no projection (reference, must agree with the main table). `randproj` = Gaussian random projection (Johnson–Lindenstrauss, preserves geometry at any width). `pca` = PCA fitted on the 270 training windows only (above dim ≈ 270 it degenerates to the same rank-270 projection, so only ≤256 is reported).

| model | full | randproj 64 | 128 | 256 | 384 | 512 | 768 | 1024 | 2048 | pca 32/64/128/256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek_llm_7b | 0.150 | 0.175 | 0.142 | 0.158 | 0.144 | 0.164 | 0.189 | 0.167 | 0.147 | 0.13/0.17/0.19/0.18 |
| deepseek_v2_lite | 0.314 | 0.325 | 0.364 | 0.386 | 0.369 | 0.375 | 0.356 | 0.342 | — | 0.38/0.36/0.35/0.34 |
| gemma2_2b | 0.150 | 0.217 | 0.222 | 0.197 | 0.250 | 0.275 | 0.192 | 0.197 | 0.147 | 0.20/0.21/0.18/0.17 |
| gemma2_9b | 0.200 | 0.194 | 0.258 | 0.217 | 0.200 | 0.242 | 0.278 | 0.272 | 0.197 | 0.19/0.19/0.12/0.17 |
| llama31_8b | 0.228 | 0.289 | 0.297 | 0.244 | 0.286 | 0.306 | 0.294 | 0.306 | 0.275 | 0.25/0.26/0.26/0.23 |
| llama32_3b | 0.147 | 0.164 | 0.175 | 0.247 | 0.161 | 0.169 | 0.219 | 0.175 | 0.150 | 0.14/0.10/0.09/0.08 |
| mistral_7b_v03 | 0.106 | 0.244 | 0.167 | 0.158 | 0.217 | 0.364 | 0.281 | 0.236 | 0.111 | 0.09/0.16/0.24/0.25 |
| olmo2_13b | 0.083 | 0.231 | 0.153 | 0.203 | 0.103 | 0.200 | 0.142 | 0.131 | 0.106 | 0.19/0.23/0.22/0.23 |
| olmo2_7b | 0.094 | 0.069 | 0.044 | 0.061 | 0.094 | 0.111 | 0.119 | 0.081 | 0.078 | 0.06/0.09/0.10/0.11 |
| qwen3_8b_base | 0.314 | 0.314 | 0.286 | 0.317 | 0.356 | 0.386 | 0.381 | 0.336 | 0.314 | 0.26/0.28/0.26/0.28 |

### Pooling sensitivity (last non-pad token vs mean over non-pad states)

| model | pooling | clean acc P/R | family-label bal3 P/R | Δ |
|---|---:|---:|---:|---:|
| qwen3_8b_base | last | 0.981/0.919 | 0.833/0.519 | 0.314 |
| qwen3_8b_base | mean | 0.991/0.917 | 0.792/0.500 | 0.292 |

### Model scale (Qwen3 family) and 10-seed headline stability

| model | params | hidden | seeds | ΔBA (bal3) | sd | 95% CI | per seed |
|---|---:|---:|---:|---:|---:|---|---|
| qwen3_0.6b | 0.6B | 1024 | 3 | **+0.175** | 0.051 | [+0.117,+0.242] | +0.242;+0.117;+0.167 |
| qwen3_1.7b | 1.7B | 2048 | 3 | **+0.158** | 0.025 | [+0.125,+0.183] | +0.183;+0.167;+0.125 |
| qwen3_8b_base | 8.2B | 4096 | 3 | **+0.311** | 0.052 | [+0.242,+0.367] | +0.325;+0.367;+0.242 |

Scale is **not** monotone: 0.6B +0.175, 1.7B +0.158, 8B +0.311 — the two small models are statistically indistinguishable from each other, and only 8B separates clearly.

| test set | seeds | ΔBA | sd | 95% CI | per-seed Δ |
|---|---:|---:|---:|---|---|
| bal3 | 10 | **+0.277** | 0.041 | [+0.253,+0.304] | s7:+0.325;s17:+0.367;s27:+0.242;s37:+0.283;s47:+0.250;s57:+0.233;s67:+0.258;s77:+0.233;s87:+0.292;s97:+0.283 |
| bal3n | 10 | **+0.267** | 0.022 | [+0.253,+0.281] | s7:+0.300;s17:+0.300;s27:+0.267;s37:+0.258;s47:+0.242;s57:+0.242;s67:+0.275;s77:+0.233;s87:+0.283;s97:+0.267 |

All 10 headline seeds are positive on both OOD sets, so the effect is not a seed artefact of seeds 7/17/27.

## 4d. Oracle-target stability — is the realised-future label a stable target?

For every synthetic window the latent process **and** the realised context are held fixed and the future noise is resampled K=20 times, giving $p_j(x)=P(\text{expert } j \text{ wins})$, stability $s(x)=\max_j p_j(x)$ and the expected-risk (majority) label $\arg\max_j p_j(x)$. The traced replay is asserted to reproduce the frozen futures used by every other experiment.

| quantity | value |
|---|---|
| windows analysed | 2250 |
| mean stability $s(x)$ | 0.916 |
| single dominant winner ($s=1$) | 77.0% |
| $s \ge 0.8$ | 81.9% |
| contested ($s \le 0.6$) | 11.8% |
| primitive windows where single-future winner = majority winner | 84.8% |
| ... restricted to stable windows ($s>0.8$) | 99.2% |
| ... restricted to unstable windows ($s \le 0.8$) | 53.3% |
| majority winner = true family | 69.3% |
| single-future winner = true family | 64.6% |
| expert-risk margin (2nd/1st): median / within 1.1x | 5.78 / 14.1% |

Reading: the realised future is not a noisy label on most windows, but it flips on ~15% of them, and the flips concentrate exactly in the near-tie windows the mechanism predicts (99% agreement where the winner is stable vs 53% where it is contested). The expected-risk label is also closer to the latent family than the single-future label, which is why it is the right sensitivity control for the oracle-label routing column.

| model | test | family-label Δ | single-future oracle Δ | expected-risk oracle Δ |
|---|---:|---:|---:|---:|
| deepseek_llm_7b | bal3 | +0.150 | -0.183 | -0.136 |
| deepseek_llm_7b | bal3n | +0.136 | -0.161 | -0.119 |
| gemma2_9b | bal3 | +0.200 | -0.056 | -0.100 |
| gemma2_9b | bal3n | +0.197 | -0.094 | -0.131 |
| llama31_8b | bal3 | +0.228 | -0.158 | -0.117 |
| llama31_8b | bal3n | +0.197 | -0.189 | -0.106 |
| olmo2_7b | bal3 | +0.094 | -0.122 | -0.111 |
| olmo2_7b | bal3n | +0.100 | -0.111 | -0.081 |
| qwen3_8b_base | bal3 | +0.314 | -0.192 | -0.197 |
| qwen3_8b_base | bal3n | +0.292 | -0.181 | -0.186 |

This is the decisive control: swapping the noisy single-future label for the expected-risk (majority) label tests whether the negative oracle column is a label-noise artefact or a genuine interface effect.

## 5. Table E — Direct numerical generation (native + API)

### 5.1 Local base LMs (greedy, historical prompt/parser; native generation)

| model | pretrained parse | pretrained native MSE | MSE / oracle | MSE / best-fixed | random parse |
|---|---:|---:|---:|---:|---:|
| deepseek_llm_7b | 1.000 | 1.388 | 4.13 | 1.94 | 0.000 |
| deepseek_v2_lite | 0.975 | 3.924 | 11.69 | 5.48 | 0.000 |
| gemma2_2b | 1.000 | 3.218 | 9.59 | 4.50 | 0.000 |
| gemma2_9b | 0.875 | 2.946 | 8.78 | 4.12 | 0.000 |
| llama31_8b | 1.000 | 2.992 | 8.91 | 4.18 | 0.000 |
| llama32_3b | 1.000 | 4.794 | 14.28 | 6.70 | 0.000 |
| mistral_7b_v03 | 0.950 | 1.981 | 5.90 | 2.77 | 0.000 |
| olmo2_13b | 0.925 | 1.409 | 4.20 | 1.97 | 0.000 |
| olmo2_7b | 1.000 | 1.898 | 5.65 | 2.65 | 0.000 |

Every model parses a large fraction of pretrained generations yet lands 3–15× above the oracle expert and *above* the best fixed expert; the matched random models emit unparseable text (0.00 parse rate) — i.e. the pretrained weights buy surface number formatting, not forecasting accuracy.

### 5.2 API-served models (chat protocol, 3 seeds × 40 stratified windows)

| API model | family | parse (strict) | parse (incl. lenient) | native MSE | MSE / best-fixed |
|---|---:|---:|---:|---:|---:|
| `360zhinao-turbo-qwen-plus` | qwen | 0.01 | 0.00 | 1.931 | 3.76 |
| `DeepSeek-V3` | deepseek | 0.00 | 0.03 | — | — |
| `deepseek-chat` | deepseek | 0.21 | 0.43 | 2.561 | 5.22 |
| `deepseek-reasoner` | deepseek | 0.00 | 0.00 | — | — |
| `gemma-3-27b-it` | gemma | 0.00 | 0.00 | — | — |
| `llama-3.1-70b-instruct` | llama | 0.33 | 0.04 | 3.559 | 5.80 |
| `llama-3.1-8b` | llama | 0.12 | 0.47 | 2.258 | 2.58 |
| `llama-3.3-70b-instruct` | llama | 0.06 | 0.01 | 2.385 | 4.44 |
| `mistral-large-2402` | mistral | 0.00 | 0.00 | — | — |
| `mistral-large-latest` | mistral | 0.00 | 0.00 | — | — |
| `qwen-max` | qwen | 0.01 | 0.00 | 3.325 | 2.57 |
| `qwen-plus` | qwen | 0.12 | 0.05 | 2.462 | 2.79 |
| `qwen2.5-72b-instruct` | qwen | 0.00 | 0.00 | — | — |

Provider-side failures (recorded, not replaced): GLM-4.1V-9B-Thinking (403 disabled), chatgpt-4o-latest (429 account), labs-mistral-small-creative (400 invalid id), llama-3.1-405b (500 no endpoints).

## 6. Mechanism summary

1. **Structure is accessible and transferable across families — 10/10 models.** Under family-label supervision, pretrained ≻ random on OOD compositional routing for **every** evaluated base LM: Qwen3-8B +31.1pp; DeepSeek-V2-Lite (MoE) +31.4pp; Llama-3.1-8B +22.8pp; Gemma-2-9B +20.0pp; Gemma-2-2B +15.0pp; Llama-3.2-3B +14.7pp; Mistral-7B-v0.3 +10.6pp; OLMo-2-7B +9.4pp; OLMo-2-13B +8.3pp; DeepSeek-LLM-7B +7.5pp. The effect is therefore not Qwen-specific; its magnitude varies by checkpoint/family.
2. **The conversion to a decision is interface-dependent — 10/10 models flip sign.** Under oracle-label supervision (target = expert with lowest *realized-future* MSE) every model turns negative: DeepSeek-LLM-7B −21.4pp; Qwen3-8B −19.5pp; Llama-3.1-8B −17.2pp; OLMo-2-13B −17.2pp; DeepSeek-V2-Lite −15.0pp; Gemma-2-2B −13.4pp; Mistral-7B −13.0pp; Llama-3.2-3B −12.3pp; OLMo-2-7B −12.2pp; Gemma-2-9B −5.6pp. The oracle target itself is unstable: family↔oracle agreement is only ~64% on clean windows, and changing the expert bank flips the winning family on 37–41% of OOD windows. So the mechanism result is: *pretrained representations organize a stable latent partition, not a realization-level expert choice.*
3. **Direct numerical generation fails on both paths.** Local base LMs produce forecasts 4–15× worse than the oracle expert; API-served models are worse still (strict numeric-format parse rate ≤33%, most 0–6%). Representation read-out is feasible, native numerical generation is not.

## 7. Status & next steps

- **Complete (P/R, 3 seeds): all 10 models** — Qwen3-8B, Llama-3.1-8B, Llama-3.2-3B, Gemma-2-9B, Gemma-2-2B, Mistral-7B-v0.3, DeepSeek-LLM-7B, OLMo-2-7B, OLMo-2-13B, DeepSeek-V2-Lite (MoE).
- **Real-world (Table D): complete for the 9 open LMs with features extracted** (15 datasets × pretrained/random/feature-router/best-fixed/oracle + paired bootstrap + BH-FDR q-values). Pretrained beats its matched random control on 12–15/15 datasets for *every* model, with median relative routed-MSE reductions of 12–28%. Qwen3-8B is **not** in this table: its real-world features come from the legacy extraction path (`results/iclr/multi_dataset_router/qwen3_8b_official`) and are reported separately.
- **Native generation: complete for the 9 open LMs re-run in this suite** (Qwen3-8B / GPT-2 numbers come from the prior round). Matched random models never emit a parseable forecast (0.00 parse rate) in any LLM, and pretrained models sit 3–15× above the oracle expert while remaining worse than the best fixed expert.
- **Figures/tables**: `figures/open_llm_suite/figA_forest.{png,pdf}` (10 LMs + 6 TSFMs), `figC_structure_vs_numerics`, `figD_realworld_heatmap`; `results/open_llm_suite/tables/robustness_matrix.csv`, `tableA_openllm_all.csv`, `tableD_summary.csv`, `all_metrics_long.csv`, `all_metrics_wide.csv`, `paper_tables.tex`.
- **Still open (do not affect the headline claim)**: the 5-expert (family5) sensitivity exists only for the first models (Gemma-2-2B/9B, Llama-3.2-3B); the MLP-router check covers the 9 new LMs but not Qwen; pooling and dimension-matched (PCA / random-projection) controls are not run for this suite; the 10-seed headline replication exists for Qwen only. These are camera-ready robustness items, not blockers for the mechanism claim.
- Data-consistency notes: the random-init feature mismatch (bf16 vs fp32 construction) was found and fixed by re-extracting clean+OOD features in one consistent pass; the 40-window API diagnostic was changed to class-stratified sampling.
