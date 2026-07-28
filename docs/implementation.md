# Minimal Experiment Implementation

## Objective

Implement a controlled first experiment in which the Transformer depth, width, attention heads, optimizer, data windows, and training budget are held fixed as far as the representation allows. The only experimental factor is the output representation and decoding procedure.

## Repository modules

### `data/datasets.py`

- `generate_synthetic(kind, total_length, seed, ...) -> np.ndarray`: generate contiguous univariate AR or sine series.
- `load_ettm1(root, target) -> SeriesSplits`: read ETTm1, use the standard 12/4/4 month split, and standardize using train statistics.
- `WindowDataset(series, context_len, horizon, max_windows, seed)`: return `(context, future)` tensors with shape `[context_len]` and `[horizon]`.
- `build_dataset(name, ...) -> tuple[Dataset, Dataset, Dataset, NormalizationStats]`: create train/validation/test windows.

### `models/backbone.py`

- `CausalTransformerBackbone`: batch-first causal Transformer encoder with learned positional embeddings. It accepts `[B, T, d_model]` and returns the same shape.

### `models/forecast_models.py`

- `DirectForecastModel`: project scalar history to tokens, run the shared backbone, and map the final history state to all future values.
- `ContinuousARModel`: project scalar values to tokens, train with teacher forcing, and predict one scalar at a time during free rollout.
- `TextARModel`: encode each normalized scalar as five character-like tokens (`+d.dd`), train next-token prediction, and decode generated token groups back to values.
- Every model exposes `training_loss(context, future)` and `predict(context, horizon)`.

The local numeric tokenizer lives under `representations/` rather than a top-level
`tokenizers/` package, so it cannot shadow HuggingFace's `tokenizers` dependency
when the later Qwen3TS experiment imports Transformers.

### `analysis/metrics.py`

- `forecast_metrics(prediction, target)`: compute MSE, MAE, RMSE, and per-horizon RMSE.

### `analysis/rollout_error.py`

- `rollout_error_by_horizon(prediction, target)`: compute mean absolute and root mean squared error at every future step.

### `analysis/spectral_error.py`

- `spectral_metrics(prediction, target)`: compare average FFT amplitude and dominant frequency.

### `analysis/instability_index.py`

- `rollout_instability_index(model, context, epsilon)`: perturb the last observed value and measure the normalized divergence between forecast trajectories.
- `instability_error_correlation(instability, error)`: calculate Pearson correlation for the diagnostic validation.

### `experiments/train.py`

- `train_model(model, train_loader, val_loader, epochs, learning_rate, device, grad_clip)`: teacher-forced optimization with free-running validation monitoring.
- `evaluate_model(model, loader, device)`: collect free-running predictions and aggregate forecast metrics.

### `scripts/run_experiment.py`

Provide one command for synthetic and ETTm1 experiments. The command must accept `--mode {direct,continuous_ar,text_ar}` and save JSON metrics, NPZ predictions, and a compact training log under `results/`.

### `models/qwen_text_ar.py` and `scripts/run_qwen_smoke.py`

- `QwenTextARForecaster`: lazily load the local Qwen3TS-8B checkpoint, attach LoRA to Q/K/V/O projections, mask prompt tokens, and train only on future numeric text.
- `run_qwen_smoke.py`: run one GPU LoRA update and one greedy generation parse. This is an optional second-stage experiment and must use the `wavellm` environment.
- `run_qwen_experiment.py`: train for a small number of epochs, evaluate free generation on a fixed test window set, report parse coverage, and save adapter/prediction artifacts.
- `RandomBPETextARModel` and `run_random_bpe_experiment.py`: use the exact Qwen tokenizer and text template with a small random Transformer, isolating tokenizer effects from Qwen language pretraining.

## Tensor contracts

- Dataset context: `[B, L]`.
- Dataset future: `[B, H]`.
- Direct output: `[B, H]`.
- Continuous autoregressive teacher-forced output: `[B, H]`; free rollout repeatedly appends one scalar.
- Text autoregressive token sequence: `[B, 5 * (L + H)]`; logits used for the future span: `[B, 5 * H, V]`.

## Experiment defaults

- The command defaults to `context_len=32`, `horizon=8`, `d_model=32`, `n_heads=4`, `n_layers=1`, dropout `0.0`.
- AdamW, learning rate `1e-3`, batch size `32`, 2 epochs, and at most 256/128/128 train/validation/test windows.
- Larger formal runs should increase the window counts, horizon, and training budget explicitly while keeping those choices identical across modes.
- Synthetic datasets use 70/15/15 contiguous splits.
- ETTm1 uses the standard split and defaults to target `OT`.

## Validation checklist

- Synthetic data generation is deterministic under a seed.
- No normalization statistic uses validation or test values.
- All three models satisfy the same public training and prediction interface.
- Evaluation always uses free-running predictions.
- Text decoding handles malformed greedy token groups without crashing.
