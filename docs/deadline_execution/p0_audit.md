# P0 validity audit

All twelve regression tests in `tests/test_deadline_p0.py` pass under
`python -m unittest discover -v tests`. The fixes change measurement code and
output locations; they do not change the task labels to improve scores.

| P0 question | Finding before fix | Resolution and test | Impact on existing results |
|---|---|---|---|
| Fixed pretrained question branch? | No. Factorized QA read `h_q` from each init cache. | `factorized_inputs` always takes question features from the paired pretrained cache; question-only also holds the nuisance temporal branch fixed. Tested by `test_factorized_question_branch_is_fixed_pretrained`. | All protocol-v1 factorized QA CSVs are invalid. |
| Temporal branch contains only temporal evidence? | No. `h_x` encoded the complete rendered question. | IRTS adapter now has separate `render_question` and `render_temporal`; the latter contains labelled series/timestamps only. Tested by branch-isolation assertions. | All protocol-v1 QA representation caches are invalid for factorized claims. |
| Task filtering keeps item/features/gold aligned? | No. Labels were re-indexed after filtering. | `select_finite_items` applies one mask to IDs, features, gold, and task types. Tested with non-contiguous retained indices. | Protocol-v1 filtered factorized metrics are invalid. |
| Stable shared Level-2 pair/split manifest? | No. Built-in `hash()` varied by process. | SHA-256 seeds and immutable manifests for seeds 7/17/27. Tests recreate manifests in independent processes and compare hashes. | All files under legacy `relations/` are excluded. |
| Constant labels in C1 unseen combinations? | Yes. Trend and local axes are constant on the C1 held-out signatures. | Manifest records label support; constant axes are excluded from balanced-accuracy reporting. Periodic remains evaluable. | Legacy C1 trend/local BA must not be reported. |
| R4/R7 duplicate? R3 identifiable? | R4 and R7 were identical; context normalization removes volatility scale needed by R3. | R7 is now a three-series latent-parameter similarity ordering task. R3 is explicitly omitted with an invalidity reason. | Legacy R4/R7 and R3 claims are excluded. |
| All relevant IRTS sequences/times extracted? | No. Only the first bracketed array was captured. | Adapter extracts prefix values, prefix timestamps, and full timestamps as three labelled blocks. Unit test uses a real-format item. | Legacy temporal perturbation controls are invalid. |
| Full/shuffled precision identical? | No. Shuffling rounded values to two decimals. | Lexical token permutation/reversal preserves each numeric token exactly. Unit test compares numeric token multisets. | Legacy shuffle/reverse contrasts are invalid. |
| Parser matches evaluation protocol? | No. The prior parser took the first standalone option letter anywhere. | Parser mirrors the released IRTS evaluator's implemented answer extraction. Tests cover answer markers and refusal text. A contradiction between that evaluator implementation and one upstream test is documented; stored native responses are single letters, so current full-condition scores are unaffected. | Reparse before using free-form outputs; current one-letter full outputs remain usable. |
| Lower-is-better metrics oriented correctly? | No. NMAE used pretrained-minus-random. | Aggregator uses random-minus-pretrained for NMAE/MAE/MSE/RMSE, so positive always means pretrained is better. Tested directly. | Old aggregate NMAE `delta` signs are wrong; raw scores remain valid. |
| Model alias resolves to the claimed checkpoint? | A Qwen QA extension initially pointed at a local `qwen3ts` wrapper while using the `qwen3_8b_base` key. | Extraction now rejects that key unless `config.json` declares `Qwen3ForCausalLM` / `qwen3` and stores path/type/config hash in new caches. Tested with a mismatched temporary config. | The one detected cache is quarantined and excluded; the rerun uses `models/qwen3-8b-base`. |
| Decoder-only native QA uses valid batched padding? | No. The legacy runner right-padded decoder-only prompts, so shorter items generated from a pad position. | The corrected runner forces left padding and records `irts_corrected_v2_left_pad_exact_lexemes`; a tokenizer regression test locks this invariant. | All legacy native QA rows, including full prompts, are withheld pending corrected rerun. |

The corrected protocol writes only beneath
`results/temporal_abstraction/corrected_v2/`. The original forecasting,
Level-1 shards, QA outputs, and protocol-v1 Level-2 files are preserved.
