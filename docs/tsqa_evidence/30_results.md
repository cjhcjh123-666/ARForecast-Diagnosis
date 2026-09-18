# Real-TSQA v1 results — locked matrix complete

Updated: 2026-09-18 (Asia/Shanghai). Protocol SHA-256:
`9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c`.

This document is updated from machine-readable predictions. It is not a paper
claim. The locked 120-run model matrix is complete.

## Completed formal runs

### Qwen3-8B-Base native QA

The source predictions are `native/qwen3_8b_base_free.jsonl` and
`native/qwen3_8b_base_constrained.jsonl`. Each file contains 1,008 predictions:
336 locked test items under full, question-only, and shuffled conditions.
The recomputed table is `tables/qwen3_8b_base_native.csv`.

For the **full-evidence** condition:

| interface | valid-output rate | Track R item accuracy (domain macro) | TUS | CIS | FCJS | QSJA |
|---|---:|---:|---:|---:|---:|---:|
| free generation, strict parser | 1.0000 | 0.3525 | 0.0476 | 0.4286 | 0.0238 | 0.2381 |
| constrained choice | 1.0000 | 0.4785 | 0.0238 | 0.4286 | 0.0238 | 0.2381 |

The full-evidence interface difference is material for Track R item accuracy,
but not full-evidence validity. Across all three conditions, free-generation
VOR is 0.8542 versus 1.0000 constrained because some question-only/shuffled
responses fail parsing. Constrained scoring does not improve the four-cell
joint score in this run. Free and constrained results remain separate, as
specified before model evaluation.

Question-only FCJS is 0 for both interfaces. This is expected from the paired
construction and is not treated as proof of temporal understanding. The small
full-versus-shuffled differences are descriptive; they do not by themselves
identify language priors or a causal mechanism.

### Gemma-2-2B question-conditioned supervised diagnostic

Seeds 7/17/27 are complete. The source files are
`diagnostic/predictions/gemma2_2b_s{7,17,27}.jsonl`; the recomputed and paired
tables are `tables/gemma2_2b_all_seeds.csv`, `tables/gemma2_2b_pr_per_seed.csv`,
and `tables/gemma2_2b_pr_paired_bootstrap.csv`. The question/option cache is
identical across P/R and has feature SHA-256
`cebef38a4ca956b5cacee3f4d66fb050286d883bcc8402ac2a4e95694217f6ed`.

For the **full-evidence** condition, the three-seed, domain-macro paired
differences are:

| metric | pretrained | random | P−R | 95% group-cluster CI for P−R |
|---|---:|---:|---:|---:|
| Track R accuracy | 0.5143 | 0.5069 | +0.0073 | [−0.0269, +0.0437] |
| Track I accuracy | 0.4731 | 0.4825 | −0.0095 | [−0.0676, +0.0486] |
| FCJS | 0.0249 | 0.0000 | +0.0249 | [0.0000, +0.0527] |
| TUS | 0.0406 | 0.0000 | +0.0406 | [+0.0128, +0.0719] |
| CIS | 0.4774 | 0.5241 | −0.0467 | [−0.1322, +0.0314] |

The bootstrap treats the three prespecified head seeds as fixed and resamples
independent source groups. TUS/FCJS gains are driven by seed 7; seeds 17 and 27
both have TUS=FCJS=0 for P and R. They are therefore not described as stable
across head seeds. The item-accuracy intervals cross zero.

### Llama-3.2-3B question-conditioned supervised diagnostic

Seeds 7/17/27 are complete. The paired tables are
`tables/llama32_3b_pr_per_seed.csv` and
`tables/llama32_3b_pr_paired_bootstrap.csv`.

| metric | pretrained | random | P−R | 95% group-cluster CI for P−R |
|---|---:|---:|---:|---:|
| Track R accuracy | 0.5014 | 0.4534 | +0.0480 | [+0.0187, +0.0813] |
| Track I accuracy | 0.4396 | 0.4492 | −0.0096 | [−0.0812, +0.0630] |
| all-item accuracy | 0.4862 | 0.4508 | +0.0354 | [+0.0062, +0.0686] |
| FCJS | 0.0313 | 0.0064 | +0.0249 | [−0.0036, +0.0563] |

Track R and all-item accuracy have positive P−R differences for each of the
three seeds. Track I and four-cell joint behavior do not show the same stable
pattern. This is evidence of task selectivity for this model, not yet a
cross-model conclusion.

### Llama-3.1-8B: same-sample interface separation

Seeds 7/17/27 are complete for both controlled readout (A) and the
question-conditioned supervised diagnostic (B). All comparisons use the same
locked test families, P/R caches, split, question vectors, head seeds, and
domain-macro aggregation.

| interface | metric | pretrained | random | P−R | 95% group-cluster CI |
|---|---|---:|---:|---:|---:|
| A: controlled readout | Track R accuracy | 0.4291 | 0.4943 | −0.0651 | [−0.1271, −0.0040] |
| A: controlled readout | Track I accuracy | 0.5922 | 0.6088 | −0.0166 | [−0.0927, +0.0612] |
| B: question-conditioned QA | Track R accuracy | 0.4947 | 0.4453 | +0.0494 | [+0.0093, +0.0907] |
| B: question-conditioned QA | Track I accuracy | 0.4293 | 0.4408 | −0.0115 | [−0.0844, +0.0515] |

The Track R P−R direction reverses between A and B. This is a measured
interface-specific separation; it is not evidence that either head captures
model understanding. B has FCJS=0 for both P and R, so its positive item-level
Track R difference does not imply successful four-cell evidence updating.

### Qwen3-8B-Base controlled and supervised diagnostics

Seeds 7/17/27 are complete for A and B. Neither interface shows a general
positive P−R item-accuracy difference. In B, Track R P−R is −0.0110 with 95%
CI [−0.0650, +0.0459]; in A it is −0.0249 with CI
[−0.0909, +0.0453]. Track I intervals also cross zero. These outcomes are
retained because the model matrix and tasks were fixed before scoring.

### Gemma-2-9B and Mistral-7B-v0.3 controlled readout

All three A seeds are complete for both models. Gemma-2-9B has a positive
Track I P−R difference (+0.1687, 95% CI [+0.0762, +0.2671]) and target-update
success difference (+0.1722, [+0.0919, +0.2622]); its Track R interval crosses
zero. Mistral has a positive target-update success difference (+0.1540,
[+0.0538, +0.2559]), while its Track R and Track I item-accuracy intervals both
cross zero. These task-specific A results are retained without projecting them
onto B or native QA. Gemma-2-9B B is now closed and shows the opposite Track I
direction: −0.0861 with 95% CI [−0.1613, −0.0174], versus +0.1687 under A.
Its B Track R difference is +0.0033 with an interval crossing zero. Mistral B
is also closed: its A target-update success difference is +0.1540 with an
all-positive interval, whereas B is −0.0128 with CI [−0.0321, 0]. Track R and
Track I item-accuracy intervals cross zero in both Mistral interfaces.

### Complete five-model native QA matrix

All five prespecified native models are complete in both free and constrained
modes. For the full-evidence condition:

| model | interface | VOR | Track R accuracy (domain macro) | TUS | CIS | FCJS |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B-Base | free | 1.0000 | 0.3525 | 0.0476 | 0.4286 | 0.0238 |
| Qwen3-8B-Base | constrained | 1.0000 | 0.4785 | 0.0238 | 0.4286 | 0.0238 |
| Llama-3.1-8B | free | 0.4762 | 0.1610 | 0.0476 | 0.2857 | 0.0476 |
| Llama-3.1-8B | constrained | 1.0000 | 0.3776 | 0.0714 | 0.4286 | 0.0714 |
| Gemma-2-9B | free | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Gemma-2-9B | constrained | 1.0000 | 0.4160 | 0.0000 | 0.5000 | 0.0000 |
| DeepSeek-V2-Lite | free | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DeepSeek-V2-Lite | constrained | 1.0000 | 0.3898 | 0.1429 | 0.3571 | 0.0238 |
| OLMo-2-7B | free | 1.0000 | 0.3610 | 0.0476 | 0.3571 | 0.0238 |
| OLMo-2-7B | constrained | 1.0000 | 0.4001 | 0.0476 | 0.3810 | 0.0476 |

Gemma free-generation zeros reflect zero strictly parseable outputs: the raw
responses continue or echo the prompt rather than returning exactly one allowed
letter. DeepSeek free generation has the same zero-validity status. Their
constrained results measure choice scoring and must not be merged with free
generation. The machine-readable cross-model table is
`tables/native_core_all_models.csv`.

## Validation status

- Benchmark validity gates: 29/29 PASS.
- Current benchmark unit tests: 20/20 PASS; deadline P0 regression tests:
  12/12 PASS.
- All five native models have 336/336 unique test items per condition and mode.
- Gemma and Llama-3.2 supervised coverage: 336/336 unique test items for every
  seed × P/R × full/question-only run.
- Paired 2,000-iteration group-cluster bootstrap is complete for all ten A/B
  models and all three seeds. `tables/interface_a_complete_models.csv`,
  `tables/interface_b_complete_models.csv`, and
  `tables/same_sample_interface_comparison.csv` retain all metrics regardless of
  sign. Native C is complete for all five prespecified models and both modes.

## Newly closed model families

DeepSeek-LLM-7B has no Track R/I A or B interval excluding zero. DeepSeek-V2-Lite
shows positive A TUS (+0.1667, CI [+0.1004,+0.2415]) and FCJS (+0.0506,
[+0.0192,+0.0855]), while its B Track R difference is negative (-0.0563,
[-0.1015,-0.0092]). OLMo-2-7B shows positive A TUS and FCJS, and a smaller
positive B Track R difference (+0.0298, [+0.0020,+0.0571]). OLMo-2-13B shows
positive A TUS (+0.3044, [+0.1895,+0.4215]) and positive B Track R (+0.0396,
[+0.0105,+0.0679]); its Track I intervals cross zero. These results strengthen
the conclusion of task/interface selectivity rather than a universal gain.
