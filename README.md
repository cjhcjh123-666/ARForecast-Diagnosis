# ARForecast-Diagnosis

Controlled experiments for diagnosing autoregressive time-series forecasting.

The first milestone compares three mechanisms under a shared causal Transformer family:

- `direct`: one forward pass predicts the full horizon;
- `continuous_ar`: scalar values are fed back one step at a time;
- `text_ar`: each scalar is represented by fixed-width character-like numeric tokens and generated token by token.

## Quick start

PyTorch must already be available in the active environment. The remaining dependencies are listed in `requirements.txt`.

```bash
cd /9950backfile/chenjiahui/ARForecast-Diagnosis
python scripts/run_experiment.py --dataset synthetic_ar --mode direct --epochs 2 --max-train-windows 512
python scripts/run_experiment.py --dataset synthetic_ar --mode continuous_ar --epochs 2 --max-train-windows 512
python scripts/run_experiment.py --dataset synthetic_ar --mode text_ar --epochs 2 --max-train-windows 512
```

ETTm1 is already available in the supplied UniTS data tree:

```bash
python scripts/run_experiment.py --dataset ettm1 --mode continuous_ar --ett-root '/public/chenjiahui/波数据时序基座大模型/UniTS-main/dataset/ETT-small'
```

Results are written to `results/<dataset>/<mode>/`.

Saved Qwen artifacts can be summarized with rollout and spectral diagnostics:

```bash
python scripts/summarize_qwen_results.py results/sweep5/synthetic_sine/qwen3_8b_lora
```

## Qwen3-8B smoke

The language-pretraining factor is optional and uses the existing local
Wave-MoE Qwen3TS checkpoint. Run it in the `wavellm` environment:

```bash
cd /9950backfile/chenjiahui/ARForecast-Diagnosis
/public/duyinglong/miniconda3/envs/wavellm/bin/python scripts/run_qwen_smoke.py --device cuda:0
```

This performs one LoRA update on text-formatted numeric history and masks the
prompt tokens from the language-model loss. It is a loader/training smoke test,
not yet the formal comparison against the random-initialized baseline.

The small formal run uses the same data-window contract and reports numeric
parse coverage separately:

```bash
/public/duyinglong/miniconda3/envs/wavellm/bin/python scripts/run_qwen_experiment.py \
  --device cuda:0 --epochs 1 --max-train-windows 128
```

For the controlled language-pretraining ablation, use the same tokenizer,
windows, seed, and LoRA protocol while changing only the initialization:

```bash
/public/duyinglong/miniconda3/envs/wavellm/bin/python scripts/run_qwen_experiment.py \
  --device cuda:0 --epochs 5 --max-train-windows 64 --max-test-windows 32 \
  --seed 7 --output-dir results/sweep5/synthetic_ar/qwen3_8b_lora

/public/duyinglong/miniconda3/envs/wavellm/bin/python scripts/run_qwen_experiment.py \
  --device cuda:0 --epochs 5 --max-train-windows 64 --max-test-windows 32 \
  --seed 7 --random-init --output-dir results/sweep5/synthetic_ar/qwen3_8b_random_lora
```

The `--random-init` run instantiates the same Qwen3-8B architecture from its
local config, then applies the same LoRA adapters. It is the relevant control
for separating language pretraining from tokenizer and capacity effects.

## Design

See `docs/implementation.md` for tensor contracts and controlled variables. `analysis/` contains rollout, spectral, and instability diagnostics.
