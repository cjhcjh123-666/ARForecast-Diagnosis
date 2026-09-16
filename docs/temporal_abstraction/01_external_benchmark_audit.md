# 01 — External time-series QA benchmark audit

_Audited 2026-09-16. All paths verified to exist on this machine; counts are read from the files._

## Summary — what is actually available locally

| benchmark | local path | items | task taxonomy | answer type | code available |
|---|---|---|---|---|---|
| **IRTS-ToolBench** | `ChatTS-Training-main/WaveTLM/third_party/IRTS-ToolBench/benchmark/irts_cleaned_benchmark.parquet` | **1700** | 10 task types (see below) | multiple choice ABCD/ABC/AB + true/false | `WaveTLM/scripts/eval_irts_wavetlm.py` (prompt + parse + per-task accuracy + macro-task accuracy) |
| **ARFBench** | `ChatTS-Training-main/WaveTLM/third_party/ARFBench-dataset/arfbench-qa.csv` + `…-hf/arfbench-ts-data/*.parquet` | **750** | 8 anomaly categories, Tier 1-3 | option string (mostly binary) | `WaveTLM/scripts/eval_arfbench_wavetlm.py` (option shuffling, official-style acc) |
| **SciTS** | `ChatTS-Training-main/WaveTLM/data/scits_eval_13tasks/wavetlm_eval.jsonl` | **10103** | anomaly_detection 4266 / imputation 2014 / forecasting 2005 / classification 1818 (13 SciTS task IDs: ENG04, ENG05, MEU01, NEG03, NEG04, PHG02, PHG03, PHU04, PHU05, PHU06, URG02, URG03, URU04) | mixed: numeric targets (`numeric_target`, `numeric_target_mask`) for forecasting/imputation, labels elsewhere | `WaveTLM/scripts/prepare_scits_eval.py` (download+convert+split) |
| **TSQA** | `/9950backfile/chenjiahui/TSQA_wave/all.jsonl` (full) and `TSQA_mini.jsonl` | **6209** (local copy) | `subset=Classification` only; `application_domain` e.g. Human Activity Recognition | free-text answer + gold (`Based on the given information, the answer is …`) | generation config `run_config.json` (vLLM, temperature 0, 8 workers) — no official scorer in this directory |

`IRTS / ARFBench / SciTS` are **not** standalone repos here; they are vendored under
`ChatTS-Training-main/WaveTLM/third_party/` (ARFBench also has its own `.git`), with the
evaluation scripts already written for the WaveTLM model. No IRTS/ARFBench/SciTS evaluation code
exists in `ARForecast-Diagnosis` — the adapters must be written fresh (see §"What we must add").

## IRTS-ToolBench (the primary Level-3 benchmark for this study)

* 1700 rows, columns `question_id, task_type, question, answer_format, answer, golden_tool_set`.
* Task mix (exact counts): `regular_vs_irregular_discrimination` 400, `anomaly_detection` 250,
  `classification` 150, `regularity_recovery` 150, `temporal_relationship` 150,
  `irregularity_severity_estimation` 150, `forecasting` 150, `characterization` 100,
  `missingness_reason` 100, `irregularity_cause_attribution` 100.
* Answer formats: `multiple_choice_abcd` 1075, `true_false` 325, `multiple_choice_ab` 200,
  `multiple_choice_abc` 100. Gold distribution: A 379, B 387, C 293, D 316, F 168, T 157
  → **no degenerate majority class** (max 22.8%).
* The question string *contains the (irregular) series as text* plus a domain description;
  mean question length 1055 chars (min 176, max 3823) → fits every LM context we use.
* Official protocol available in `eval_irts_wavetlm.py`: per-task accuracy, macro-task accuracy,
  `ANSWER_FORMAT_TO_LABELS` parsing. **Zero-shot** in our setting (no training data needed).
* Adaptation required for us: strip the WaveTLM model plumbing and keep (a) the prompt renderer,
  (b) the label parser. We do **not** re-define the benchmark.

## ARFBench

* 750 QA rows, columns include `question, task_category, difficulty, options_str, correct_answer,
  query_group, options, interpolate_1/2`.
* Categories: Anomaly Correlation 170, Anomaly Indicator 163, Anomaly Presence 111,
  Anomaly Categorization 104, Anomaly Magnitude 76, Anomaly Start 56, Anomaly Identification 38,
  Anomaly End 32. Difficulty: Tier 3 333 / Tier 2 306 / Tier 1 111.
* The series live in a separate parquet tree keyed by `query_group` (`<id>_<index>_<interval>.parquet`,
  intervals 10/60/300/1800/3600 s) → the adapter must join QA rows to series.
* Metric: option accuracy; the reference script shuffles options with a seed, which we keep fixed
  (protocol lock) so that no run is advantaged by option order.

## SciTS

* 10103 converted records; `task_type` distribution above; each record carries `timeseries`,
  `question`, `answer`, and for numeric tasks `numeric_target` + `numeric_target_mask`.
* Because 2/3 of SciTS is numeric (forecasting/imputation/anomaly detection), it is **not** the best
  Level-3 reasoning set; use its ~1818 `classification` records for a reasoning probe and keep the
  rest as Level-4 numerical reference (already covered by our forecasting readout).
* Official metric: task-specific (MSE/MAE for numeric, accuracy/F1 for classification) — the
  prepare script does not implement scoring, so scoring must be taken from the SciTS papers/ToolBench
  convention. Flagged as a risk in `02_protocol_lock.md`.

## TSQA

* Local copy is **classification-only** (6209 items, `subset=Classification`), with
  `input` containing a `<ts><ts/>` placeholder, a separate `timeseries` field, a free-text
  `answer`, and `application_domain` (e.g. Human Activity Recognition).
* Upstream TSQA has more subsets (imputation/forecasting/anomaly) but they are **not** present here.
  We therefore treat TSQA as a classification-reasoning benchmark, with free-text answers scored by
  exact/contained match against the gold string (documented deviation from any official scorer,
  which we do not have).

## What we must add (and what we must not do)

Add, under `temporal_abstraction/qa_adapters/`:
`irts.py`, `arfbench.py`, `scits.py`, `tsqa.py` — each providing
`load_items(split)`, `render(question, series, condition)`, `parse(text)`, `is_correct(pred, gold)`,
plus a frozen manifest with SHA256 so every QA run is traceable.

Do **not**: rebuild these benchmarks, re-implement their task definitions, tune prompts per benchmark
beyond the single locked template, or aggregate them into one "overall QA score".

## Consequence for the study design

* Level 3 (reasoning) uses **IRTS** as the primary set (temporal_relationship, characterization,
  anomaly_detection, classification, regularity_recovery), **ARFBench** as an anomaly-reasoning
  replication, **TSQA-Classification** as a distribution-shift replication, and SciTS-classification
  as a scientific-domain replication.
* Level 4 keeps the existing numerical evidence (readout / native generation / real-world routing);
  SciTS forecasting+imputation is an additional numerical reference, not a reasoning target.
* Natural-language QA can *never* be run with a matched-random LM as the temporal encoder and
  interpreted as attribution (a random LM cannot read the language) → the **factorized** protocol
  (`qa_factorized.py`, pretrained question branch + pretrained-or-random temporal branch) is mandatory,
  together with the **symbolic task-code** control.
