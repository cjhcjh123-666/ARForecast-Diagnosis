# Rollout Instability Results

The text instability index perturbs the last history value by `epsilon=0.01`,
generates a second forecast, and divides the trajectory RMSE divergence by
the input perturbation. The two-decimal text representation makes smaller
perturbations potentially invisible.

| Dataset | Seed | Windows | Index mean | Median | Max | Baseline RMSE mean | Error correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic sine | 7 | 8 | 47.46 | 6.12 | 206.79 | 0.3622 | 0.794 |
| synthetic sine | 17 | 8 | 1.24 | 0.98 | 3.87 | 0.0799 | -0.380 |
| synthetic sine | 7 | 32 | 2.22 | 1.90 | 6.94 | 0.0957 | -0.463 |
| synthetic sine | 17 | 32 | 2.83 | 2.31 | 7.02 | 0.0869 | 0.183 |
| ETTm1 | 7 | 8 | 1.66 | 0.83 | 5.20 | 0.0728 | -0.412 |
| ETTm1 | 17 | 8 | 1.24 | 0.98 | 3.87 | 0.0799 | -0.380 |
| ETTm1 | 7 | 32 | 3.21 | 2.05 | 15.66 | 0.0858 | 0.119 |
| ETTm1 | 17 | 32 | 1.26 | 1.00 | 4.85 | 0.0843 | 0.051 |

The 32-window evaluations do not support a stable relationship between this
simple local perturbation index and forecast error. The positive correlation
from the initial eight-window sine check did not replicate, and ETTm1 remains
near zero. This is a useful negative result: raw text-rollout sensitivity is
not yet a reliable training weight or early-warning metric.

The next method should therefore use directly observed per-horizon errors and
frequency-domain degradation, rather than optimizing against this index.
