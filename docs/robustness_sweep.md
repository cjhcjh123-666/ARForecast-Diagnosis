# Robustness Sweep

This follow-up uses synthetic sine with `context=64`, `horizon=16`, 256
training windows, 64 test windows, batch size 2, five epochs, and the same
LoRA protocol as phase one. The only controlled factor is initialization:
pretrained Qwen3-8B versus the same Qwen3-8B architecture with random base
weights. All errors are in train-normalized units.

| Seed | Initialization | Parse rate | RMSE | RMSE step 1 -> 16 | Spectral amplitude MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | pretrained | 1.000 | 0.09884 | 0.08612 -> 0.08961 | 0.25699 |
| 7 | random | 1.000 | 0.16291 | 0.14914 -> 0.14879 | 0.36871 |
| 17 | pretrained | 1.000 | 0.09770 | 0.09616 -> 0.08778 | 0.24380 |
| 17 | random | 1.000 | 0.14236 | 0.15337 -> 0.12933 | 0.34530 |

| Initialization | Mean RMSE | Sample std. | Mean spectral amplitude MAE |
| --- | ---: | ---: | ---: |
| pretrained | 0.09827 | 0.00080 | 0.25040 |
| random | 0.15264 | 0.01453 | 0.35701 |

The pretrained initialization reduces mean RMSE by `35.6%` relative to the
random initialization. Both conditions have complete numeric forecasts for
all 64 test windows, so this comparison is not explained by parse failure.
This is still a small two-seed result; the next robustness check should add a
third seed or increase the number of independent series before making a final
causal claim.

Regenerate the aggregate from ignored artifacts with:

```bash
python scripts/summarize_seed_sweep.py \
  results/robustness/synthetic_sine/seed7/qwen3_8b_lora \
  results/robustness/synthetic_sine/seed7/qwen3_8b_random_lora \
  results/robustness/synthetic_sine/seed17/qwen3_8b_lora \
  results/robustness/synthetic_sine/seed17/qwen3_8b_random_lora
```
