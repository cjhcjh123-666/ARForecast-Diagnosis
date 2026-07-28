# Qwen Phase 1 Results

These are preliminary five-epoch results with `context=64`, `horizon=16`,
64 training windows, 32 test windows, batch size 2, and seed 7. All errors
are in train-normalized units. The Qwen runs use the local Qwen3-8B
checkpoint with LoRA; the random Qwen control uses the same architecture,
tokenizer, LoRA configuration, and training protocol but random base weights.

| Dataset | Initialization | Parse rate | Valid samples | RMSE | RMSE step 1 -> 16 | Spectral amplitude MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| synthetic AR | pretrained | 1.000 | 32/32 | 1.2097 | 0.8013 -> 1.3899 | 2.1346 |
| synthetic AR | random | 0.031 | 1/32 | not stable | 1.6463 -> 2.9935 | 2.3980 |
| synthetic sine | pretrained | 1.000 | 32/32 | 0.2826 | 0.0941 -> 0.5943 | 0.5326 |
| synthetic sine | random | 1.000 | 32/32 | 0.4494 | 0.4257 -> 0.5289 | 0.9636 |
| ETTm1 | pretrained | 1.000 | 32/32 | 0.1000 | 0.0405 -> 0.1284 | 0.2211 |

The main robust comparison is synthetic sine: both Qwen variants produce
complete numeric forecasts, while pretrained initialization reduces RMSE and
spectral amplitude error. The synthetic AR random run has only one complete
forecast, so its RMSE must not be treated as a reliable point estimate. The
small tokenizer-matched random BPE model is a separate capacity-mismatched
diagnostic and produced zero complete forecasts after five epochs.

Regenerate the table from saved artifacts with:

```bash
python scripts/summarize_qwen_results.py \
  results/sweep5/synthetic_ar/qwen3_8b_lora \
  results/sweep5/synthetic_ar/qwen3_8b_random_lora \
  results/sweep5/synthetic_sine/qwen3_8b_lora \
  results/sweep5/synthetic_sine/qwen3_8b_random_lora \
  results/sweep5/ettm1/qwen3_8b_lora
```
