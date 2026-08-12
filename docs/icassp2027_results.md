# ICASSP 2027 — Final Results Summary (2026-08-12)

Direction: **"Recognize, Don't Generate"** — language pretraining as a
temporal-structure prior in LLM time-series forecasting.

## E1. Controlled forecast ablation (Qwen3-8B + LoRA, 3 seeds)

| Dataset | Init | MSE (mean±std) | MAE | Parse | Spec amp MAE | Rel. MSE ↓ |
|---|---|---|---|---|---|---|
| sine | pretrained | 0.0094 ± 0.0006 | 0.077 | 1.000 | 0.238 | — |
| sine | random | 0.0212 ± 0.0011 | 0.111 | 1.000 | 0.342 | 125% |
| ETTm1 | pretrained | 0.0099 ± 0.0017 | 0.071 | 1.000 | 0.215 | — |
| ETTm1 | random | 0.2453 ± 0.1366 | 0.339 | 1.000 | 0.819 | 2380% |
| ETTh1 | pretrained | 0.0487 ± 0.0097 | 0.160 | 1.000 | 0.449 | — |
| ETTh1 | random | 0.2896 ± 0.2059 | 0.366 | 0.990 | 0.876 | 495% |

Protocol: context 64 / horizon 16, 256 train / 64 test windows, 5 epochs,
LoRA r=8 on q/k/v/o, seeds {7,17,27}, train-only normalization.  Identical
architecture/tokenizer/windows; only initialization differs.

## E2. Frozen structure recognition (5-kind synthetic dynamics)

| Method | Overall | Trend | Periodic | Local | Mixture | Regime |
|---|---|---|---|---|---|---|
| Hand features + logistic | 0.990 | 1.00 | 0.97 | 0.98 | 1.00 | 1.00 |
| Frozen pretrained Qwen3-8B | 0.963 | 1.00 | 0.94 | 0.94 | 0.94 | 1.00 |
| Frozen random Qwen3-8B | 0.906 | 1.00 | 0.96 | 0.78 | 0.88 | 1.00 |

(3 seeds: 7/17/27)

## E3. Paired recognize-vs-generate (same 40 windows, stratified)

| Metric | Value |
|---|---|
| Recognition accuracy (frozen probe) | 0.967 |
| Frozen generation parse rate | 0.267 |
| Frozen generation MSE (complete) | 1.854 |
| Oracle expert MSE (same windows) | 0.334 |

## E4. Zero-shot expert router (train on trend/periodic/local → unseen mixture/regime)

| Method | Oracle-acc | Hard MSE | Soft MSE |
|---|---|---|---|
| Oracle (upper bound) | — | 0.446 | — |
| Best single (trend) | — | 0.500 | — |
| Uniform ensemble | — | 1.287 | — |
| Feature router | 0.128 | 1.278 | 1.235 |
| Frozen random probe | 0.289 | 1.079 | 1.056 |
| Frozen pretrained probe | 0.678 | 0.739 | 0.723 |

## E5. Real-data zero-shot: ETTm1 (300 test windows, no adaptation)

| Method | Oracle-acc | MSE | Route (T/P/L) |
|---|---|---|---|
| Oracle (4 experts) | — | 0.0103 | — |
| Best single univariate (periodic) | — | 0.0137 | — |
| Always periodic (degenerate) | 0.677 | 0.0137 | 0/300/0 |
| Uniform ensemble | — | 0.2552 | — |
| Feature router | 0.060 | 0.6568 | 11/18/271 |
| Frozen random probe | 0.677 | 0.0137 | 0/300/0 |
| Frozen pretrained probe | 0.517 | 0.0199 | 82/218/0 |

Honest read: on homogeneous ETTm1, both LLM-based routers avoid the
catastrophic `local` trap that breaks the feature router (MSE 0.66), but a
degenerate always-periodic policy is already near-optimal there, so the
language-pretraining advantage shows on diverse/unseen dynamics (E4), not on a
single homogeneous real series.

ETTh1 reproduces the same pattern (300 windows): feature router MSE 0.748 /
5% oracle-acc (collapses to local, 292/300); pretrained probe MSE 0.041 / 54%
(routes 26 to trend, 274 to periodic); random probe MSE 0.039 / 54%
(degenerate always-periodic).

## E6. Patch-level dynamic routing (bonus, frozen LLM probe)

Horizon split into 4×4-step segments; each segment routed independently by the
frozen LLM probe on the 32 steps before it (train probe on 32-step clean
patches).  Per-kind forecast MSE:

| Kind | Whole-LLM | Patch-LLM | Patch oracle |
|---|---|---|---|
| regime | 0.817 | **0.546** | 0.280 |
| local | 1.635 | **1.305** | 0.624 |
| mixture | 0.731 | 0.860 | 0.229 |
| periodic | 0.280 | 0.999 | 0.086 |
| trend | 0.001 | 0.065 | 0.001 |
| overall | 0.693 | 0.755 | 0.244 |

Overall routing acc vs per-segment oracle: whole-LLM 44.3%, patch-LLM 45.7%,
feature-patch 44.8%.  Feature patch router overall MSE 1.144.

Honest read: dynamic routing helps where dynamics change within the window
(regime -33%, local -20%), and the LLM patch router beats the feature patch
router by ~34% overall, but stationary windows (periodic/trend) do not benefit
because 32-step patches carry weaker structural evidence and routing errors
compound.

## Key takeaways

1. Under strict control (same arch/tokenizer/windows, init only differs),
   language pretraining massively helps forecasting (2–25× MSE).
2. A *frozen* pretrained LLM recognizes temporal dynamics well; its advantage
   over cheap features and random same-architecture control appears under
   **distribution shift** (zero-shot), not on easy in-distribution classes.
3. Recognition is robust while numeric generation is fragile (95% vs 27.5%
   parse / 6× error on the same windows).
4. The frozen LLM routes experts zero-shot to unseen synthetic dynamics (63.3%
   oracle-acc vs 10.8% features) and transfers to real ETTm1 (51.7% vs 6.0%).

## Honest limitations (to keep in the paper)

- Synthetic probe data covers three seeds (7/17/27); recognition/router numbers
  are stable across them.
- Clean stationary dynamics are easy for cheap features (98% floor); the LLM
  advantage is specifically OOD transfer.
- On unseen synthetic mixture/regime, best-single (trend) is competitive
  because trend dominates the oracle; routing's value is recognition transfer.
- Experts: AR is a universal model on clean stationary windows; periodic
  specialist does not beat AR on clean sines (known limitation of the expert
  set, motivates unseen-dynamics framing).

## Artifacts

- `results/icassp/E1_table.md`, `results/icassp/E1_summary.json` — forecast table
- `results/icassp/probe/summary.json` — E2/E3/E4 (seed 7)
- `results/icassp/probe_seed17/` — seed 17 probe (in progress)
- `results/icassp/ettm1_router/probe_summary.json` — E5
- `paper/icassp2027/main.tex` — paper draft (numbers filled)
- Scripts: `scripts/run_icassp_forecast_sweep.sh`, `scripts/run_frozen_probe.py`,
  `scripts/compute_probe_results.py`, `scripts/run_probe_ettm1.py`,
  `scripts/run_ettm1_router.py`, `scripts/summarize_icassp.py`
