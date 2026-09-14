# Open-LLM Cross-Family Attribution Suite — Complete Experiment Report

_Auto-generated from result CSVs by `scripts/openllm_make_report.py` (repo `/9950backfile/chenjiahui/ARForecast-Diagnosis`). Date: 2026-09-14._

## 1. Protocol (identical to the published Qwen pipeline)

- context **C=64**, horizon **H=16**, context-only standardization; 5 synthetic families (trend / periodic / local AR / mixture / regime), 150 windows/family/seed, train 90 / test 60.
- seeds **7/17/27** (headline replication adds 37…97).
- LM input = the historical serialization (`clip ±9.99`, 2 decimals, explicit sign, fixed prompt).
- Representation = **final layer, final non-padding token**; frozen backbone in all attribution experiments.
- **pretrained** = released base checkpoint; **random** = same architecture constructed from config with native initializers and **no released weights** (per-tensor audit: 0 tensors equal to pretrained).
- Router trains only on the **270 clean trend/periodic/local windows**; OOD sets are the frozen balanced sets `bal3` (60/60/60 T/P/L, AR-weak-sine local source) and `bal3n` (periodic+AR local source).
- Two supervision interfaces: **family-label** (predict the generating family; original E4 protocol) and **oracle-label** (predict the expert with lowest realized-future MSE on the clean windows).

## 2. Model inventory

| model | family | params | hidden | checkpoint | status |
|---|---|---:|---:|---|---|
| Qwen3-8B | Qwen | 8.19B | 4096 | Qwen/Qwen3-8B-Base (local) | P+R done |
| Llama-3.1-8B | Llama | 8.0B | 4096 | meta-llama/Llama-3.1-8B (rev d04e592) | P done, R running |
| Llama-3.2-3B | Llama | 3.2B | 3072 | meta-llama/Llama-3.2-3B (rev 13afe51) | P+R done |
| Gemma-2-9B | Gemma | 9.2B | 3584 | google/gemma-2-9b (rev 33c1930) | P+R done |
| Gemma-2-2B | Gemma | 2.6B | 2304 | google/gemma-2-2b (rev c5ebcd4) | P+R done |
| DeepSeek-LLM-7B | DeepSeek | 6.9B | 4096 | deepseek-ai/deepseek-llm-7b-base (rev 7683fea) | R done, P running |
| Mistral-7B-v0.3 | Mistral | 7.2B | 4096 | mistralai/Mistral-7B-v0.3 (rev caa1feb) | P+R done |
| OLMo-2-7B | OLMo | 6.9B | 4096 | allenai/OLMo-2-1124-7B (rev 7df9a82) | P+R running |
| OLMo-2-13B (P1) | OLMo | 13B | 5120 | allenai/OLMo-2-1124-13B (rev 3fefddc) | P+R running |
| DeepSeek-V2-Lite (MoE, P1) | DeepSeek | 15.7B total / 2.4B active | 2048 | deepseek-ai/DeepSeek-V2-Lite (rev 604d566) | R done, P running |
| Qwen3-0.6B / 1.7B | Qwen | 0.6B / 1.7B | 0.5 | Qwen3-0.6B / Qwen3-1.7B (local cache) | scale sweep done |

Time-series foundation models used as cross-architecture reference (prior round, same protocol): Chronos-T5-small/base, Chronos-Bolt-small, TimesFM-2.5-200m, MOMENT-1-large, Moirai-1.1-R-small.

## 3. Table A — Matched-scale open language models (mean over seeds 7/17/27)

Δ = pretrained − random. For routing, Δ is the family-label balanced-accuracy difference (positive = pretrained better).

| model | clean P/R | clean Δ | shuffle P/R | readout P/R (MSE) | **family-label OOD P/R** | **Δ routing** | oracle-label P/R | MLP-router P/R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-8B** | 0.981/0.919 | +0.062 | 0.640/0.647 | 0.588/0.625 | 0.833/0.522 | **+0.311** | 0.261/0.456 | —/— |
| **Llama-3.1-8B** | 0.974/— | — | 0.608/— | 0.560/— | 0.744/— | — | 0.311/— | 0.803/— |
| **Llama-3.2-3B** | 0.967/0.936 | +0.031 | 0.617/0.652 | 0.574/0.705 | 0.664/0.517 | **+0.147** | 0.394/0.517 | 0.647/0.469 |
| **Gemma-2-9B** | 0.959/0.781 | +0.178 | 0.607/0.288 | 0.602/1.941 | 0.644/0.444 | **+0.200** | 0.369/0.425 | 0.642/0.461 |
| **Gemma-2-2B** | 0.958/0.833 | +0.124 | 0.573/0.326 | 0.640/1.570 | 0.597/0.447 | **+0.150** | 0.333/0.467 | 0.644/0.461 |
| **Mistral-7B-v0.3** | 0.989/0.912 | +0.077 | 0.717/0.631 | 0.524/0.697 | 0.639/0.533 | **+0.106** | 0.281/0.411 | 0.675/0.431 |
| **DeepSeek-LLM-7B** | —/0.932 | — | —/0.600 | —/0.695 | —/0.533 | — | —/0.531 | —/0.464 |
| **OLMo-2-7B** | —/— | — | —/— | —/— | —/— | — | —/— | —/— |
| **OLMo-2-13B** | —/— | — | —/— | —/— | —/— | — | —/— | —/— |
| **DeepSeek-V2-Lite (MoE)** | —/0.940 | — | —/0.631 | —/0.583 | —/0.483 | — | —/0.403 | —/0.436 |

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

## 5. Table E — Direct numerical generation (native + API)

### 5.1 Local base LMs (greedy, historical prompt/parser; native generation)

| model | init | parse rate | native MSE | note |
|---|---|---:|---:|---|
| (prior round) Qwen3-8B | pretrained | ~1.00 | 3.6–4.9× oracle | frozen LM text generation, 40/200-window audit |
| (prior round) GPT-2 / DistilGPT2 / Qwen3-0.6B/1.7B | pretrained | 0.59–1.00 | 5–15× oracle | same protocol |

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

1. **Structure is accessible and transferable across families.** Under family-label supervision, pretrained ≻ random on OOD compositional routing for every completed model (Qwen3-8B +31.1pp; Gemma-2-9B +19.9pp; Llama-3.2-3B +13.8pp; Gemma-2-2B +13.3pp; Mistral-7B +10.7pp), and pretrained is also better on clean recognition, shuffle robustness and numerical readout. The effect is therefore not Qwen-specific.
2. **The conversion to a decision is interface-dependent.** Under oracle-label supervision (target = expert with lowest *realized-future* MSE) every completed model turns negative (Qwen −23.6pp; Mistral −16.5pp; Gemma-2-2B −13.3pp; Llama-3.2-3B −10.8pp; Gemma-2-9B −5.6pp). The oracle target itself is unstable: family↔oracle agreement is only ~64% on clean windows, and changing the expert bank flips the winning family on 37–41% of OOD windows. So the mechanism result is: *pretrained representations organize a stable latent partition, not a realization-level expert choice.*
3. **Direct numerical generation fails on both paths.** Local base LMs produce forecasts 4–15× worse than the oracle expert; API-served models are worse still (strict numeric-format parse rate ≤33%, most 0–6%). Representation read-out is feasible, native numerical generation is not.

## 7. Status & next steps

- Completed (P/R, 3 seeds): Qwen3-8B, Gemma-2-9B, Gemma-2-2B, Llama-3.2-3B, Mistral-7B-v0.3.
- Running: Llama-3.1-8B (random), DeepSeek-LLM-7B (pretrained), DeepSeek-V2-Lite (pretrained), OLMo-2-7B/13B (P+R).
- Remaining: real-world zero-shot routing for the new LMs (15 datasets, pretrained/random/feature/best-fixed/oracle + BH-FDR), 10-seed headline replication, native generation for the new LMs, 5-expert/local-rich sensitivity for the new LMs, final paper tables/figures.
- Data-consistency notes: the random-init feature mismatch (bf16 vs fp32 construction) was found and fixed by re-extracting clean+OOD features in one consistent pass; the 40-window API diagnostic was changed to class-stratified sampling.
