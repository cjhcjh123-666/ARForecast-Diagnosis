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

## Design

See `docs/implementation.md` for tensor contracts and controlled variables. `analysis/` contains rollout, spectral, and instability diagnostics.
