# Reviewer-follow-up CPU analyses (no model re-run)

All numbers computed from the v2 deliverable per-window predictions + stored
OOD ctx/fut (scripts/tsfm_expertbank_stability.py, tsfm_expertbank_ood.py).

## 1. Oracle stability: winner flips when the expert bank changes
Recomputing the OOD winner over a 5-expert bank
(trend:{Trend,TrendSeasonal}, periodic:{Periodic}, local:{AR5,AR10})
changes the **winner FAMILY** on 36.7-40.8% of the balanced-OOD windows
vs the original 3-expert oracle (per seed; both bal3 and bal3n). I.e. the
single-future winning expert is a highly implementation-sensitive target.

## 2. Margin (best/second-best) concentrates the instability
Clean270 family-vs-oracle agreement (3-expert oracle) vs margin:
  margin <1.05 (near tie): agreement 0.385 (n=26)
  margin 1.05-1.2 : 0.548 (n=73);  1.2-1.5 : 0.530 (n=83)
  margin >1.5 (robust): 0.689 (n=628)
Label flips concentrate on near-tie windows.

## 3. Qwen family-router gain grows with oracle margin (bal3)
Qwen balanced-accuracy (family-label router) by margin bin:
  margin<1.05 : P .796 R .648 (n small)
  1.05-1.2    : P .694 R .620 ;  1.2-1.5 : P .668 R .480
  margin>1.5  : P .882 R .508  -> pretrained-random gain +0.374 (largest)
So the 0.833-style advantage is carried by windows with a *stable* expert
decision, consistent with the "structural partition transfers" reading.

## 4. Headline is NOT robust to expert-bank definition (key caveat)
Router balanced accuracy when the target becomes the 5-expert winner FAMILY
(3-class; mean over seeds; pretrained | random | delta):
  qwen3_8b_base  0.466 | 0.439 | +0.027
  chronos_t5-small 0.418 | 0.349 | +0.069
  chronos_t5-base 0.559 | 0.341 | +0.218
  chronos_bolt-small 0.508 | 0.280 | +0.228
  timesfm_2.5-200m 0.559 | 0.409 | +0.150
  moment-1-large  0.369 | 0.453 | -0.085
  moirai-1.1-R-small 0.370 | 0.431 | -0.061
With the original 3-expert oracle Qwen is 0.833|0.522|+0.311; with the
5-expert-oracle family target it drops to +0.027 and several TS FMs look
"better". Conclusion: the +31pp headline is a property of the *specific
3-expert bank / single-future oracle*, not a bank-independent fact.

## 5. Framing consequence (advisory)
- Keep the headline as: **structural family partition transfers under the
  family-label router on the paper's balanced OOD sets** (a controlled,
  bank-specific diagnostic), NOT as "forecasting-relevant expert decision
  transfer".
- Recommended next experiments (GPU-bound): expected-risk / majority-future
  oracle (stabilize the target), dimension-matched Qwen routing (PCA/RP),
  Qwen 0.6B/1.7B/8B routing scale sweep, 10-seed Qwen headline, linear-vs-MLP
  router.

## 6. Qwen dimension matching (family-label routing; 8B features; fit on clean270 only)
Random projection to TSFM-scale dims does NOT remove the pretrained gain:
| dim | method | P_bal3 | R_bal3 | dP | dP_bal3n |
|---:|---|---:|---:|---:|---:|
| 384 | RP | 0.819 | 0.450 | +0.369 | +0.358 |
| 512 | RP | 0.831 | 0.442 | +0.389 | +0.372 |
| 768 | RP | 0.836 | 0.464 | +0.372 | +0.358 |
| 1024 | RP | 0.839 | 0.492 | +0.347 | +0.342 |
| 2048 | RP | 0.825 | 0.522 | +0.303 | +0.289 |
| 4096 | raw | 0.833 | 0.522 | +0.311 | +0.289 |
PCA fit on clean270 to 384-2048 collapses BOTH pretrained and random to ~chance
(0.32) on the OOD sets (clean train acc stays 1.0 at every dim): clean-fit PCA
discards the low-variance axes that carry the OOD composition signal, so PCA is
not an appropriate dimension-matched projector here. Conclusion: the Qwen gain is
not an artifact of the 4096-dim representation; it survives (indeed increases
slightly under) isotropic random projection to TSFM-scale dims.

## 7. Qwen scale sweep (native dims; family-label routing)
| size | dim | P_bal3 | R_bal3 | dP_bal3 | dP_bal3n |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 1024 | 0.653 | 0.478 | +0.175 | +0.167 |
| Qwen3-1.7B | 2048 | 0.683 | 0.525 | +0.158 | +0.142 |
| Qwen3-8B | 4096 | 0.833 | 0.522 | +0.311 | +0.289 |
Gain is not monotonic between 0.6B and 1.7B (+0.175 vs +0.158), then jumps at 8B
(+0.311). Combined with section 6 (dim not the cause), do NOT claim a clean
scaling law; claim: the large pretrained-routing gain appears at 8B and is not
explained by representation dimension.

## 8. Linear vs 2-layer MLP router (Qwen sizes; bal3; family-label protocol)
MLP: hidden 64, ReLU, Adam lr=1e-3 wd=1e-2, full-batch 600 steps, train on clean270 only.
| size | head | P | R | delta |
|---|---|---|---:|---:|
| Qwen3-0.6B | linear | 0.653 | 0.478 | +0.175 |
| Qwen3-0.6B | mlp64 | 0.606 | 0.439 | +0.167 |
| Qwen3-1.7B | linear | 0.683 | 0.525 | +0.158 |
| Qwen3-1.7B | mlp64 | 0.678 | 0.467 | +0.211 |
| Qwen3-8B | linear | 0.833 | 0.522 | +0.311 |
| Qwen3-8B | mlp64 | 0.839 | 0.475 | +0.364 |
A nonlinear router does NOT let the from-config random control close the gap:
the pretrained gain persists (8B: +0.311 -> +0.364 under MLP). So the Qwen
advantage is not merely linear accessibility under this protocol.

## 9. Qwen3-8B 10-seed headline routing (DONE; seeds 7,17,27,37,47,57,67,77,87,97)
bal3 & bal3n balanced windows (40/40/40) for the 7 extra seeds were generated by
replicating the e4 generators; 8B from-config random + pretrained features were
extracted per seed (gpu8). Family-label router, per-seed balanced-accuracy delta:
  seed  7 +0.325, 17 +0.367, 27 +0.242, 37 +0.283, 47 +0.250, 57 +0.233,
        67 +0.258, 77 +0.233, 87 +0.292, 97 +0.283   (bal3)
  bal3 : mean +0.277 +- 0.041 (sd across seeds) | seed-bootstrap 95% CI [+0.253,+0.304]
  bal3n: mean +0.267 +- 0.022 | CI [+0.253,+0.281]
Every one of 10 independent synthetic seeds (new windows + new from-config
random inits) shows a positive pretrained gain (+0.23 to +0.37), CI excludes 0.
The 3-seed mean was +0.311 (seeds 7/17/27); 10-seed mean +0.277. The headline
is therefore stable across synthetic seeds.
