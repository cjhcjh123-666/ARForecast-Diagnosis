# ICASSP 2027 Paper Plan (working outline)

**Deadline: 2026-09-16** (Toronto, 2027-05-16..21).  4-page body + references.

---

## Working title

*Recognize, Don't Generate: Language Pretraining as a Temporal-Structure Prior
in LLM Time-Series Forecasting*

## Core thesis

In LLM-based time-series forecasting, language pretraining helps primarily
because it gives the model a **temporal-structure prior** (recognizing trend /
periodicity / local dependence), not because it makes the model a better
numeric *generator*.  Numeric-text generation is the fragile mode: pretrained
initialization fixes generation (parse coverage, accuracy), but even *frozen*
LLM representations already recognize temporal dynamics well above chance and
above a same-architecture random control — and can act as a zero-shot expert
router to unseen dynamics.

This positions the paper directly against the skeptical line (Tan et al.,
NeurIPS 2024: "Are LMs Actually Useful for TS Forecasting?"), which found no
benefit from language pretraining for popular LLM4TS methods.

## Contributions

1. **Controlled pretraining ablation** (same architecture, tokenizer, windows,
   LoRA protocol; only init differs): pretrained Qwen3-8B > random-init by a
   large margin on sine / ETTm1 / ETTh1 across seeds — under direct
   parameter-efficient fine-tuning, language pretraining *does* help.
2. **Frozen structure probe**: a linear probe on the frozen LLM's final-token
   hidden state (numeric-text history input) classifies temporal dynamics
   (trend / periodic / local / mixture / regime) far above a random-init
   same-architecture control and at least on par with hand-crafted features;
   leave-one-kind-out shows zero-shot structure generalization.
3. **Paired recognize-vs-generate**: on the *same windows*, recognition is
   accurate while frozen numeric generation is unreliable (low parse coverage,
   high error) — quantifying the asymmetry.
4. **Zero-shot expert router**: the frozen probe routes among lightweight
   statistical experts on unseen dynamics; routing accuracy and downstream MSE
   beat the feature floor and best-single-expert baselines.

## Experiments / tables

| # | Experiment | Settings | Output |
|---|-----------|----------|--------|
| E1 | Controlled forecast | sine / ETTm1 / ETTh1; context 64, horizon 16; 256/64 train/test windows; 5 epochs; LoRA; seeds {7,17,27}; init ∈ {pretrained, random} | Table 1: MSE (+MAE, parse rate, spectral amp MAE) |
| E2 | Frozen structure probe | synthetic 5-kind dynamics; linear probe on last-token hidden state; init ∈ {pretrained, random}; + hand-feature floor | Table 2: per-kind acc, overall acc, LOIO mean |
| E3 | Paired recognize vs generate | same synthetic windows; frozen greedy generation vs frozen probe | Table 3a: parse rate, mse, recognize acc |
| E4 | Zero-shot router | train router on {trend,periodic,local}, apply to {mixture,regime}; experts = trend/periodic/local | Table 3b: routing acc vs oracle, downstream MSE |

## Running / completed experiments

- `scripts/run_icassp_forecast_sweep.sh` on GPUs 4/5/6: 18 runs (3 datasets ×
  2 inits × 3 seeds), output under `results/icassp/<dataset>/<init>/seed<seed>/`.
- `scripts/run_frozen_probe.py` on GPU 7: probe + paired generation + zero-shot
  router, output under `results/icassp/probe/summary.json`.
- Existing assets to reuse: `docs/qwen_phase1_results.md`, `docs/robustness_sweep.md`
  (2-seed sine/ETTm1), `docs/instability_results.md` (text-rollout fragility),
  `docs/mitigation_results.md`.

## Related work to cite / position against

- Tan et al. NeurIPS 2024, "Are Language Models Actually Useful for Time Series
  Forecasting?" — our E1 is a direct controlled counterpoint under PEFT.
- Time-LLM (ICLR 2024), AutoTimes, CALF — LLM4TS methods (reprogramming /
  frozen / token-based).
- "Random Initialization Can't Catch Up" (arXiv 2506.21570), "LLM Pretraining
  Shapes a Generalizable Manifold" (arXiv 2605.20449) — same-arch init studies;
  we differ in (a) modern Qwen3-8B + numeric text + LoRA, (b) the
  recognize-vs-generate asymmetry, (c) zero-shot router application.
- LLM as router / MoE for TS: LEGO (NAACL 2025 Findings), FM-LLM (KBS 2026),
  DMoE-LLM, RSynLLM, MPL-MoE (ICASSP 2026) — we route *temporal dynamics* with a
  *frozen* LLM and measure zero-shot generalization to unseen dynamics.
- Spectral / signal-processing analysis fits ICASSP: FFT amplitude error,
  spectral entropy, dominant-period statistics.

## Open items / risks

- E1: wait for 18 runs; aggregate per-seed mean/std; confirm pretrained margin
  holds on ETTh1 (new dataset).
- E2/E3/E4: interpret probe numbers once GPU 7 finishes.
- Expert specialization caveat: AR is a universal model on clean stationary
  windows; the routing story must emphasize unseen-dynamics generalization
  (zero-shot) rather than in-distribution wins.
- Scope for 4 pages: possibly merge E3 into E2 discussion and keep E4 compact.

## Next steps

1. Poll GPU 7 probe results; sanity-check per-kind accuracies.
2. Aggregate E1 sweep into a table script (`scripts/summarize_icassp.py`).
3. Draft LaTeX skeleton (IEEEtran, 4 pages) and fill numbers.
4. Decide figures: (a) per-kind probe acc bars, (b) routing MSE bars, (c) maybe
   a spectral illustration.
