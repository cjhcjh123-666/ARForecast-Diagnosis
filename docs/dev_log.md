# Development Log

## Progress

| Item | Status |
| --- | --- |
| Repository and implementation specification | Done |
| Dataset loader | Done |
| Shared causal Transformer | Done |
| Direct forecast model | Done |
| Continuous autoregressive model | Done |
| Text autoregressive model | Done |
| Training and evaluation entry point | Done |
| Synthetic smoke test | Done |
| ETTm1 data check | Done |

## Decisions

- The first milestone uses a small Transformer trained from scratch. Qwen3-8B/WaveTLM is reserved for a later pretraining-factor experiment.
- ETTm1 is accessed through the external UniTS dataset directory and is never copied into this repository.
- All evaluation is free-running; teacher forcing is used only as the training protocol for autoregressive models.
- The first CPU smoke tests show that text-token rollout can be much less stable under a tiny training budget. Its hard tokenization also makes a `1e-3` input perturbation invisible unless it crosses a `0.01` quantization boundary, so the instability index must be interpreted as representation sensitivity for this mode.

## Running Notes

The runnable command is `python scripts/run_experiment.py`. The smoke test runs on CPU without downloading dependencies or model weights. Example artifacts were written during validation under `results/smoke/` and `results/ettm1_smoke/`; they are ignored by Git.

## Validation Results

- Synthetic AR, synthetic sine, and ETTm1 all completed direct, continuous autoregressive, and text autoregressive smoke runs.
- The direct and continuous models produced finite `[B, H]` forecasts in every run.
- The text tokenizer round-trip test and all model contract tests passed when called directly. The environment lacks `pytest`, so the `pytest` command itself was not available during validation.
