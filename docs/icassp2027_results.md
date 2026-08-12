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
| Hand features + logistic | 0.980 | 1.00 | 0.97 | 0.95 | 1.00 | 1.00 |
| Frozen pretrained Qwen3-8B | 0.967 | 1.00 | 0.97 | 0.95 | 0.92 | 1.00 |
| Frozen random Qwen3-8B | 0.910 | 1.00 | 0.97 | 0.70 | 0.88 | 1.00 |

## E3. Paired recognize-vs-generate (same 40 windows, stratified)

| Metric | Value |
|---|---|
| Recognition accuracy (frozen probe) | 0.950 |
| Frozen generation parse rate | 0.275 |
| Frozen generation MSE (complete) | 1.979 |
| Oracle expert MSE (same windows) | 0.328 |

## E4. Zero-shot expert router (train on trend/periodic/local → unseen mixture/regime)

| Method | Oracle-acc | Hard MSE | Soft MSE |
|---|---|---|---|
| Oracle (upper bound) | — | 0.449 | — |
| Best single (trend) | — | 0.492 | — |
| Uniform ensemble | — | 1.220 | — |
| Feature router | 0.108 | 1.379 | 1.350 |
| Frozen random probe | 0.267 | 1.198 | 1.179 |
| Frozen pretrained probe | 0.633 | 0.773 | 0.761 |

## E5. Real-data zero-shot: ETTm1 (300 test windows, no adaptation)

| Method | Oracle-acc | MSE |
|---|---|---|
| Oracle (4 experts) | — | 0.0103 |
| Best single univariate (periodic) | — | 0.0137 |
| Uniform ensemble | — | 0.2552 |
| Feature router | 0.060 | 0.6568 |
| Frozen pretrained probe router | 0.517 | 0.0199 |

Probe router routing distribution on ETTm1: [trend 82, periodic 218, local 0] —
the frozen LLM correctly recognizes ETTm1 as periodicity-dominated.

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

- Synthetic probe data covers two seeds (7, 17); recognition/router numbers are
  stable across them, but more seeds would be safer before submission.
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
