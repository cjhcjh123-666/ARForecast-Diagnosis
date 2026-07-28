# User Requirements

## Scope

- The first milestone is a minimal controlled experiment for autoregressive time-series forecasting.
- Use the same small decoder-only Transformer backbone for three output mechanisms:
  1. direct multi-horizon forecast;
  2. continuous scalar autoregressive forecast;
  3. fixed-width textual numeric token autoregressive forecast.
- Start with synthetic AR and sine data. Add ETTm1 as a real-data check.
- Report free-running rollout error by horizon. Keep teacher-forced training separate from free-running evaluation.
- Do not use a large pretrained LLM in the first experiment. Pretraining is a later controlled factor.
- Do not copy the external dataset into this repository. Read it through a configurable path.

## Data

- Default ETTm1 root: `/public/chenjiahui/波数据时序基座大模型/UniTS-main/dataset/ETT-small`.
- Default ETTm1 target: `OT`.
- ETTm1 standard split: 12 months train, 4 months validation, 4 months test at 15-minute frequency.
- Fit normalization statistics on the training segment only.

## Engineering

- PyTorch is assumed to be installed by the runtime environment and must not be listed in `requirements.txt`.
- The code must run on CPU for smoke tests and use CUDA automatically when available.
- Keep output artifacts under `results/` and keep raw datasets external.
