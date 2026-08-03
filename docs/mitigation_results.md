# Lightweight Mitigation Results

The first mitigation injects Gaussian noise into the normalized history only
during teacher-forcing updates. The forecast target and free-running
evaluation remain unchanged. The noise standard deviation is `0.02`, and all
other settings use the 256/64-window, five-epoch protocol.

| Dataset | Seed | Training history | Parse rate | RMSE | RMSE step 1 -> 16 | Spectral amplitude MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| synthetic sine | 7 | clean | 1.000 | 0.09884 | 0.08612 -> 0.08961 | 0.25699 |
| synthetic sine | 7 | noise 0.02 | 1.000 | 0.09635 | 0.09081 -> 0.09032 | 0.24817 |
| ETTm1 | 7 | clean | 1.000 | 0.11541 | 0.04132 -> 0.16780 | 0.22939 |
| ETTm1 | 7 | noise 0.02 | 1.000 | 0.10519 | 0.03162 -> 0.15250 | 0.24777 |
| ETTm1 | 17 | clean | 1.000 | 0.10620 | 0.03616 -> 0.15230 | 0.24574 |
| ETTm1 | 17 | noise 0.02 | 1.000 | 0.10623 | 0.03744 -> 0.15270 | 0.25361 |

On ETTm1, the mean RMSE changes from `0.11080` to `0.10571` across the two
seeds, a preliminary `4.6%` reduction. However, the improvement is driven by
seed 7; seed 17 is unchanged, and mean spectral amplitude MAE worsens from
`0.23757` to `0.25069`. The method is therefore a useful low-cost ablation,
not yet a robust contribution. More seeds and a noise scale sweep are needed
before using it in the main result.
