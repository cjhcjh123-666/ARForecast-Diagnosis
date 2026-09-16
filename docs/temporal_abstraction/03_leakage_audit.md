# 03 — Leakage audit

_Machine-readable results: `results/temporal_abstraction/tables/leakage_checks.csv`
(48 checks = 16 checks × 3 seeds, all PASS after the two fixes below). Script:
`scripts/abstraction_leakage_audit.py`. Run **before** any probe was executed._

## Level 1 / 1b checks (synthetic probes)

| # | check | what it rules out | result |
|---|---|---|---|
| 1 | `duplicate_series_in_pool` | the same standardised series appearing twice (would inflate both train and test) | 0 duplicates (sha1 of ctx‖fut over 4380 windows × 3 seeds) |
| 2 | `train_test_index_overlap_targets` | a window being in train **and** test of the same target | 0 overlaps for all 13 targets × 3 seeds |
| 3 | `train_test_duplicate_series` | a test window with an identical twin in train | 0 |
| 4 | `param_train_test_KS` | train/test differing systematically in the continuous knobs (would make the probe look better/worse for a sampling reason) | KS ≤ 0.09 for all regression targets (threshold 0.15) |
| 5 | `prompt_template_only` | the serialized LM input containing anything but the fixed template + numbers | exact-match on the locked template |
| 6 | `prompt_label_text_leakage` | any label word (trend/period/anomaly/changepoint/family) leaking into the prompt | 0 |
| 7 | `unseen_signature_collision` | the composition probe being solvable without generalisation | 0 (see fix 2) |
| 8 | `metadata_outside_input` | probe labels reaching the representation | labels live only in `pool_s*.jsonl`; probes read frozen `h` arrays |
| 9 | `binary_label_balance` | degenerate majority-class probes | max class fraction 0.667 (P3, by design: 600 periodic vs 300 aperiodic controls) |

### Fix 1 — stratified splitting was wrong for regression targets (found by check 4)

`split_indices` treated every unique value of a continuous target as its own class, which sent all
500 windows to train and left the test split **empty** (silent `nan` metric, not an error).
Fix: continuous targets (`> 50 % unique values`) get a plain random 60/40 split; classification
targets keep the stratified split. Applied before any probe ran; `probes.py` now returns
`int` index arrays explicitly.

### Fix 2 — REGIME could not be used as an "unseen composition" for the signature probe

`REGIME` (trend → periodic, sequential) has the same 3-bit presence signature (1,1,0) as the
**seen** composition `TP` (trend + periodic, simultaneous). A probe trained on seen classes would
have been scored on a class whose signature it had already seen — free accuracy, no compositional
generalisation. Fix (pre-result):
* the 3-bit presence probe now trains on `{T, P, L, TP}` and is tested on `{TL, TPL}`, whose
  signatures `(1,0,1)` and `(1,1,1)` never occur in training;
* `REGIME` is retained for a separate **arrangement probe (C3)**: simultaneous `TP` vs sequential
  `REGIME`, i.e. *how* the primitives are arranged rather than which ones are present.

## Level 2 checks

* Relation probes (R1–R8) are built by *pairing windows of the same pool*, so no new latent process
  is introduced; each pair is assigned a label from the recorded latents only.
* The pair features are `concat(h_a, h_b)` in a fixed order for symmetric relations (R1–R7) and
  `concat(h_lag, h_lead)` for the asymmetric lead–lag relation (R8) — the order is protocol-locked.
* Composition train/test uses **class** disjointness (seen vs unseen combinations); parameter ranges
  are deliberately shared, so a probe cannot succeed by memorising a knob value.

## Level 3 checks (to be appended once the QA adapters are built)

These are mandatory before QA numbers are reported:

1. answer-frequency bias per benchmark and per task (IRTS gold: A 379 / B 387 / C 293 / D 316 /
   T 157 / F 168 → max 22.8 %, already checked on the raw file);
2. option-position balance after our locked shuffling seed (ARFBench shuffles options);
3. TSQA: verify the gold answer text is not contained in the question/serialized input;
4. question-only condition must be reported for every benchmark — any benchmark where
   Question-only ≈ Full is flagged as a **language-prior** benchmark, not a temporal one;
5. shuffled/reversed condition multiset check (values identical, order changed);
6. factorized protocol: verify the temporal branch is the *only* difference between pretrained and
   random runs (same question features, same fusion, same head, same split) — assert by hashing the
   question-feature array.
