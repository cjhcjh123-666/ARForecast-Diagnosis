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
- The tokenizer-matched random BPE baseline used the same Qwen tokenizer, text template, windows, and one epoch, but had parse coverage `0.0`; generations collapsed to repeated `4` or `-` tokens. Qwen3-8B reached parse coverage `1.0` under the same small protocol. This supports a preliminary language-prior effect, while larger matched runs are still required.
- A reproducible five-epoch sweep now fixes the seed, train/test window counts, and generation budget in both Qwen scripts. On synthetic AR with 64 train and 32 test windows, pretrained Qwen3-8B + LoRA reached parse coverage `1.0` and RMSE `1.2097`; the tokenizer-matched small random BPE model still had parse coverage `0.0` despite its training loss falling to `0.0035`.
- A capacity-matched random-initialized Qwen3-8B + LoRA run reached parse coverage `0.03125` (1/32 complete forecasts) after five epochs. Its reported RMSE `1.7884` is not a stable comparison because it is computed on one valid sample, but the large parse-coverage gap motivates repeating the pretrained/random initialization comparison on sine and ETTm1.
- The five-epoch sine repeat produced complete forecasts for both initializations: pretrained Qwen RMSE `0.2826` versus random Qwen RMSE `0.4494`; spectral amplitude MAE was `0.5326` versus `0.9636`. The ETTm1 pretrained run also parsed all 32 test windows with RMSE `0.1000`, and per-horizon RMSE increased from `0.0405` to `0.1284`.
- `scripts/summarize_qwen_results.py` now regenerates rollout and spectral summaries from ignored prediction artifacts. The tracked table is in `docs/qwen_phase1_results.md`; these numbers are preliminary because each run uses only 64 training windows and 32 test windows.
- `scripts/run_qwen_instability.py` loads a saved Qwen LoRA adapter and compares forecasts after perturbing the final history value. On 8 sine windows with `epsilon=0.01`, the index had mean `47.46`, median `6.12`, maximum `206.79`, and correlation `0.794` with sample rollout RMSE. The long tail is consistent with an instability mechanism, but the sample is too small for a final claim.
- The expanded synthetic-sine robustness sweep used 256 training and 64 test windows. Across seeds 7 and 17, pretrained Qwen RMSE was `0.09827 +/- 0.00080`, versus `0.15264 +/- 0.01453` for the random same-architecture control; all four runs had parse coverage `1.0`. The pretrained condition therefore reduced mean RMSE by `35.6%` and spectral amplitude MAE from `0.35701` to `0.25040`.
- The same two-seed, 256/64-window sweep on ETTm1 produced pretrained RMSE `0.11080 +/- 0.00652` versus random-initialized RMSE `0.35206 +/- 0.00574`, a `68.5%` relative reduction. Mean spectral amplitude MAE was `0.23757` versus `0.59526`; parse coverage was `1.0` for pretrained and `0.9922` for random because one random forecast was incomplete.
