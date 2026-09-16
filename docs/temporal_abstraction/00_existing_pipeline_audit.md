# 00 — Existing pipeline audit (ARForecast-Diagnosis)

_Audited 2026-09-16. Scope: what already exists in this repo, what can be reused for the
Temporal-Abstraction study, what must be added, and what must not be overwritten._

Repo: `/9950backfile/chenjiahui/ARForecast-Diagnosis` (branch `main`, HEAD `49a67b7`).
Environment: `/public/duyinglong/miniconda3/envs/wavellm` (torch 2.4.1, transformers 4.52.4);
DeepSeek-LLM-7B needs `chatts` (torch 2.6) because it ships `pytorch_model.bin`.
Hardware: 8 × A800-80GB.

## 1. Components that exist and are REUSABLE as-is

| component | file | what it gives us |
|---|---|---|
| model loading (pretrained / matched random) | `scripts/openllm_suite.py::load_lm` | pretrained = official base checkpoint; random = `AutoConfig.from_pretrained` → `AutoModelForCausalLM.from_config` with architecture-native init, `torch.manual_seed(seed)`, **same attention implementation** for both branches (required; gemma/dsv2 fixes documented in `docs/open_llm_failures_and_blockers.md` §9-10) |
| representation extraction | `scripts/openllm_suite.py::embed` | `output_hidden_states=True`, `use_cache=False`, bf16 forward → float32 numpy; pooling `last` (final non-pad token) or `mean` (masked mean) |
| LM serialization | `scripts/openllm_suite.py::_format_values`, `prompts` | clip ±9.99, 2 decimals, explicit sign, fixed prompt `"Forecast the next values of this normalized time series.\nHistory: …\nForecast:"` |
| prompt/parser for generation | same file | `NUM_RE = (?<![0-9])[+-]?\d+\.\d{2}`, `parse_numbers(text, h)` |
| frozen feature dumps | `results/open_llm_suite/features_full/full_<model>_<init>_s<seed>.npz` | 10 models × {pretrained, random} × seeds {7,17,27}: `h_clean` (750 windows, 5 families), `h_bal3`, `h_bal3n` (120 windows each), `bal3_oracle`, `bal3n_oracle`, `fut`, and (Qwen only) mean-pooled variants |
| router/probe trainer | `analysis/linear_probe.py::softmax_regression / softmax_predict / accuracy` | pure-numpy multinomial logistic regression, 2000 iters, l2 1e-3, internal standardization. **This is the locked probe for the new study.** |
| MLP head / router | `scripts/openllm_suite.py::mlp_router`, `lin_head`, `mlp_head` | robustness only |
| experts | `models/experts.py` (`TrendExpert`, `PeriodicExpert`, `LocalExpert`), `scripts/tsfm_expertbank_stability.py` (`TrendSeasonalExpert`, `ARExpert`) | 3-expert and 5-expert banks |
| window builder | `data/dynamics.py::build_labeled_windows / generate_window` | 5 families: trend / periodic / local AR / mixture / regime, C=64, H=16, 150 windows per family |
| frozen OOD sets | `results/iclr/e4_balanced3/windows/bal3_s*.npz`, `results/iclr/e4_balanced3_natural/windows/bal3n_s*.npz` | 120 windows each, 40/40/40 by future-defined oracle winner |
| real-world evaluation | `scripts/openllm_realworld_features.py`, `openllm_realworld_analysis.py` | 15 datasets × 10 models, zero-shot routing, paired bootstrap + BH-FDR |
| native generation / API zoo | `scripts/openllm_native.py`, `openllm_api_generate.py`, `openllm_api_aggregate.py` | direct numerical generation, both local and API |
| statistics + figures | `scripts/openllm_figures.py`, `openllm_export.py`, `openllm_make_report.py`, `openllm_final_report.py` | bootstrap CIs, FDR, forest plots, LaTeX tables, auto-generated markdown |
| parallel CPU recipe | `scripts/openllm_remaining_experiments.py` + `openllm_merge_parts.py` (env-var sharding) | 24-way sharding turned a ~90 min sequential analysis into minutes; **reuse this pattern for all new probe sweeps** |

Model checkpoint paths (revision-pinned): see `docs/open_llm_reproducibility.md` §"Update 2026-09-14".
Qwen3-8B is the local `models/qwen3-8b-base`; the other nine are read from
`/9950backfile/chenjiahui/hf_cache/hub/…` and `/9950backfile/chenjiahui/model_cache_modelscope/LLM-Research/…`.

## 2. Components that must be ADDED (nothing in the repo covers them)

| need | why it is missing | new file |
|---|---|---|
| latent ground-truth generator | `data/dynamics.py` draws its parameters inline from `rng` and **returns only (context, future)**. There is no way to read slope / period / φ / σ / anomaly index / change-point index back out. | `temporal_abstraction/generators.py` |
| signed trend, variable noise, anomalies, random-location change points | current families have **only positive slopes** (`uniform(0.03, 0.18)`), fixed noise per family (`σ=0.05` / `0.15`), no anomaly injection, and the only change point sits at a fixed `0.6·C` fraction. Probes P1/P6/P7-P10 are therefore degenerate on the existing generator. | same |
| primitive probe targets + splits | not implemented | `temporal_abstraction/primitive_targets.py` |
| pairwise relation targets | not implemented | `temporal_abstraction/relation_targets.py` |
| seen/unseen composition splits | the forecasting study has a *fixed* OOD set (bal3/bal3n) but no train/test split machinery over compositions | `temporal_abstraction/composition_splits.py` |
| layer-wise extraction | `embed()` always takes the **last** hidden state | `temporal_abstraction/layerwise.py` |
| probe/statistics runner | not implemented | `temporal_abstraction/probes.py`, `statistics.py` |
| QA adapters + QA runs | the repo has no QA code at all (the QA asset lives in the WaveTLM project, see `01_external_benchmark_audit.md`) | `temporal_abstraction/qa_adapters/*`, `native_qa.py`, `qa_factorized.py` |
| capability joint analysis | not implemented | `temporal_abstraction/capability_analysis.py` |

## 3. Existing results that MUST NOT be overwritten

Everything under `results/open_llm_suite/`, `results/iclr/`, `docs/open_llm_*`, and the four
deliverable zips. The temporal-abstraction study writes **only** to
`results/temporal_abstraction/**`, `temporal_abstraction/**`, `configs/temporal_abstraction/**`,
`docs/temporal_abstraction/**` and `scripts/abstraction_*` / `run_*` new scripts.
The existing `data/dynamics.py` generator is **frozen**: the new generator wraps it, never edits it,
so the forecasting numbers stay reproducible bit-for-bit.

## 4. Ground-truth availability of the current generator (STEP 3 of the brief)

| required probe target | available today? | action |
|---|---|---|
| P1 trend direction | NO — slope is always positive | add signed slope to the new generator |
| P2 trend strength | yes (slope magnitude) | reuse |
| P3 periodicity presence | yes (family ∈ {periodic, mixture}) | reuse |
| P4 dominant period | only {12, 24} | widen to {8,10,12,16,20,24,32} in the new generator |
| P5 local dependence | yes (φ for the AR family) | reuse |
| P6 noise / volatility | NO — σ fixed per family | sample σ per window from a range |
| P7 anomaly presence | NO | add spike/dip/flatline injection with a recorded index |
| P8 anomaly location | NO | same |
| P9 change-point presence | only via `regime` (always present, fixed 0.6·C) | add optional change point at a **uniform random index**, and non-change-point controls |
| P10 change-point location | NO | same |

Conclusion: **the new generator is a prerequisite for Level 1**; it must record every latent
parameter it draws (this is the ingredient that makes the labels ground truth instead of model-derived).
The existing families are reimplemented inside it with the *same* parameter ranges, so
"primitive / seen composition" windows remain distribution-compatible with the frozen bal3 sets.

## 5. Protocol inheritance (what carries over unchanged)

* context C=64, horizon H=16, context-only standardization;
* serialization: clip ±9.99, `%+.2f`, explicit sign, same prompt template;
* representation: frozen backbone, final layer, final non-padding token (mean-pooling = robustness);
* probe: linear (`softmax_regression`, 2000 iters, l2 1e-3), trained on the probe train split only;
* random control: from-config architecture-native init, same tokenizer/serialization/inference;
* seeds 7 / 17 / 27 for everything; no test-set-driven hyper-parameter choice.

## 6. Key risks inherited from the previous round (must be re-checked, not re-litigated)

1. Attention-implementation mismatch between pretrained and random branches was a real bug class
   (gemma2 sdpa→compile, DeepSeek sdpa unsupported). New extraction code must call `load_lm`, never
   build its own loader.
2. Random init differs per seed by design (`torch.manual_seed(seed)`), so a random control may not be
   reused across seeds.
3. `results/open_llm_suite/raw/shards/` shows the sharding convention used for parallel runs; keep it so
   that every reported table can be traced back to per-shard CSVs.
