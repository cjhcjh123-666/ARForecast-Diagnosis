# 02 — Protocol lock

_Locked 2026-09-16, before any probe result was computed. Machine-readable copy:
`configs/temporal_abstraction/protocol.yaml`. **No item below may be changed because of a result.**
Any bug fix must be logged, committed, and the affected experiments re-run from scratch._

## What is inherited from the forecasting suite (unchanged)

| item | value |
|---|---|
| context / horizon | C = 64, H = 16 |
| standardization | context-only mean/std (the LM never sees future statistics) |
| serialization | `openllm_suite.prompts` — clip ±9.99, `%+.2f`, explicit sign, same template |
| representation | frozen backbone, `output_hidden_states=True`, final **non-padding** token |
| random control | `AutoConfig.from_pretrained` → `from_config` (architecture-native init, `torch.manual_seed(seed)`), **same** attention implementation as the pretrained branch, same tokenizer/serialization/inference code |
| BF16 forward | yes; features cast to float32 for probing (float16 on disk) |
| seeds | 7, 17, 27 |

## Level 1 — probe pool (per seed, 4380 windows)

Blocks: TREND 500 · PERIODIC 600 · APERIODIC 300 · LOCAL 500 · TP 300 · TL 300 · TPL 300 ·
REGIME 300 · ANOM_CLEAN 320 · ANOM 320 · CP_CLEAN 320 · CP 320.

Latents recorded per window: family, block, slope/sign/noise, period/amp/phase, φ/σ,
`anomaly_present/type/index/width/mag/bin`, `changepoint_present/type/index/delta/bin`,
`periodicity_present`, plus the standardization constants. Ground truth comes from the generator,
never from an expert or a model.

Locked family parameter ranges (`temporal_abstraction/generators.py`): trend `|slope| ∈ [0.03,0.18]`
with **balanced sign**, noise ∈ {0.03, 0.08, 0.15}; periodic period ∈ {8,10,12,16,20,24,32},
amp ∈ [0.3,1.8]; local φ ∈ {0.55,0.70,0.85,0.95}, σ ∈ {0.05,0.15,0.30}; regime switch ∈ [0.3C,0.7C];
anomaly injected in **z-space** with magnitude ∈ [2.5,5.0] z-units and index uniform over the context;
change point = level shift (|δ| ∈ [1.5,3.5] z-units, random sign) or variance switch (×2.0–3.5) at a
uniform index in [0.2C, 0.8C]. `data/dynamics.py` is untouched; a `legacy=True` mode reproduces it exactly.

## Probe targets (13; no additions after this point)

| id | label | metric |
|---|---|---|
| P1 | trend direction (sign) | balanced accuracy |
| P2 | trend strength, 4 levels (\|slope\| cuts 0.06/0.10/0.14) | balanced accuracy (+R² variant) |
| P3 | periodicity present (periodic vs aperiodic) | balanced accuracy |
| P4 | dominant period, 7 levels | balanced accuracy (+R² variant) |
| P5 | local dependence, 4 φ levels | balanced accuracy (+R² variant) |
| P6 | noise level, 3 levels | balanced accuracy |
| P7 | anomaly present | balanced accuracy |
| P8 | anomaly location, 8 context bins | balanced accuracy |
| P9 | change-point present | balanced accuracy |
| P10 | change-point location, 8 context bins | balanced accuracy |

Probe = single linear readout (`analysis.linear_probe.softmax_regression`, 2000 iters, lr 0.5,
l2 1e-3, internal standardization; ridge α=1 for regression). **No per-target hyper-parameter search.**
Split = per-target stratified 60/40, drawn with `default_rng(10000 + 100·seed + target_index)`;
train/test indices are disjoint by construction and identical across models (the split depends only on
(target, seed)).

## Level 1b — layer-wise probes

Same pool, same probes, but representations read at `early(1)`, 25 %, 50 %, 75 % and final layer of
`config.num_hidden_layers`. Reported as probe score (and pretrained − random gain) versus normalized
depth. **No prior that deeper is better.**

## Level 2 — relations and composition (locked)

* Relations (pairwise, both series encoded by the same model): R1 stronger trend, R2 stronger
  periodicity, R3 higher volatility, R4 same/different family, R5 same/different period,
  R6 more anomalous, R7 similarity ordering, R8 lead–lag (series B = series A shifted by ±k).
  Representation = concat(h_a, h_b) (protocol-locked); probe = linear.
* Composition: seen = {T, P, L, TP}; unseen = {TL, TPL, REGIME}. The probe predicts the 3-bit
  presence signature (trend/periodic/local present) and is trained **only** on seen classes;
  primary metric = per-label balanced accuracy on unseen classes, plus exact-signature match.
  No parameter range is reserved for unseen classes (the unseen axis is the *combination*, not new
  parameter values).

## Level 3 — QA

* Primary: **IRTS-ToolBench** (1700 items, 10 task types, MC/TF). Replications: ARFBench (750),
  TSQA-classification (6209), SciTS-classification (1818).
* Conditions: **Full** (question + series), **Question-only**, **Shuffled TS** (values permuted, same
  multiset), **Reversed TS** (semantically applicable subset only).
* Native QA (generation) runs on the 5 representative models; raw response, parsed answer,
  correctness, category and condition are saved per item.
* **Factorized attribution** (the only QA protocol allowed to claim pretraining attribution):
  question branch always `E_pretrained(q)`; temporal branch `E_pretrained(x)` vs `E_random(x)`;
  identical fusion `concat(h_q, h_x, |W h_q − W h_x|, W h_q ⊙ W h_x)`, identical linear head, identical
  training split, identical hyper-parameters. Only the *temporal* representation changes.
* **Symbolic control**: the natural-language question is replaced by a canonical task code
  (e.g. `TASK=COMPARE_TREND_STRENGTH`) with no answer information; any pretrained advantage that
  survives this condition cannot be explained by language comprehension.
* One locked prompt template per benchmark; no per-benchmark prompt tuning.

## Statistics

Paired (same windows for pretrained and random), 10 000-resample bootstrap over windows with shared
indices, per-seed values always reported, BH-FDR across probes within a family. Effect sizes always
accompany p-values/q-values.

## Explicit prohibitions (from the brief)

designing per-primitive bespoke heads; per-benchmark prompt tuning; keeping only tasks that look good;
aggregating benchmarks into an arbitrary overall score; treating natural-language QA accuracy alone as
evidence of "temporal understanding"; deleting the existing forecasting results.
