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
- GPU validation used A800 devices successfully. The small baseline completed CUDA runs for all three modes on synthetic AR data.
- The local numeric tokenizer package was renamed from `tokenizers/` to `representations/` after a Qwen3TS load exposed a namespace collision with HuggingFace's `tokenizers` package.
- The local Wave-MoE Qwen3TS-8B checkpoint loads on an A800 and generates text, but its separate time-series encoder weights are not merged automatically by `from_pretrained`; it is not yet a valid pretrained time-series baseline.
- The Qwen3-8B LoRA smoke completed on A800 GPU 3 with 8.44B total parameters, 7.67M trainable parameters, and one-step masked LM loss `1.260159`. The generated numeric text parsed successfully; its all-zero output is not a trained forecast result.
- A formal small Qwen run on synthetic AR used 64 train windows, 32 test windows, `context=64`, `horizon=16`, and one epoch. All generated forecasts parsed successfully and achieved RMSE `1.1882`.
- A same-window one-epoch random-initialized character-token baseline achieved RMSE `7.1082`. This is a useful preliminary signal, not an isolated pretraining result, because the Qwen run uses Qwen BPE tokenization while the random baseline uses the custom fixed-width tokenizer.
- The next controlled ablation is a tokenizer-matched random baseline, followed by a random-initialized Qwen architecture if the compute budget is justified.
