# Cache and result inventory

Snapshot taken 2026-09-18 11:58 CST.

| Asset | Count | Size | Status | Reuse decision |
|---|---:|---:|---|---|
| Frozen Level-1 representation NPZs | 60 | 9.62 GB | Complete: 10 models × 2 inits × 3 seeds | Reuse directly for corrected Level-1 and Level-2 |
| Level-1 per-run probe CSVs | 60 | included in 11.8 MB primitives tree | Complete | Valid; do not rerun |
| Legacy Level-2 CSVs | 60 | 70.4 KB | Complete after takeover; workers exited | Preserve, but invalid for paper claims |
| Legacy IRTS QA representation NPZs | 10 | 107.7 MB | 5 models × 2 inits × seed 7 | IDs/gold align, but branch construction is invalid |
| Legacy native IRTS CSVs | 5 | included in 113.6 MB QA tree | 600 rows/model; four conditions | Full prompt usable; perturbation contrasts excluded |
| Legacy factorized IRTS CSVs | 15 | included in QA tree | 5 models × 3 conditions | Invalid and retained only for audit |
| Corrected Level-2 manifests | 3 | 1.58 MB | Complete for seeds 7/17/27 | Shared by every v2 model/init run |
| Corrected Level-1 aggregate | 1,560 rows + merged raw table | 1.74 MB | Complete | Valid improvement-oriented metrics |
| Corrected IRTS representation NPZs | 4 | two models × two inits | Complete for DeepSeek-V2-Lite and official Qwen3-8B-Base | Valid for factorized diagnostics |
| Corrected IRTS factorized CSVs | 8 | four conditions × two models | Complete at seed 7 | Valid but limited/mixed evidence |
| Corrected IRTS native CSV | 1 | 600 rows | Complete for official Qwen3-8B-Base seed 7 | Valid left-padded run; condition parse rates differ |
| Quarantined Qwen3TS cache | 1 | audit only | Identity mismatch caught before matched branch | Never merge into corrected QA tables |
| Corrected Level-2 CSVs | growing | separate `corrected_v2/relations` tree | Qwen seed-7 pair closing | Aggregate only complete pretrained/random pairs |

Manifest SHA-256 identities:

- seed 7: `fd6e6312e5f2f9d42e1b71338d1de2f5b0ec8a2cc2d17aa48591cfb1bdc07b05`
- seed 17: `f20b9096e2c3b82f3e7c8008723be5b48583204bcefa215315def66d41feaa9b`
- seed 27: `1b985641158a144250a0470fb99021c79103bbd36390bbc2f9508c2cc8ca46ee`

Files that exist only as runnable code without valid v2 results at this
snapshot: corrected Level-2 coverage beyond the active Qwen3-8B seed-7 pair.
The coverage guard records `may_claim_complete_10_model_coverage=false` until
all 30 model/seed pairs have both initializations.
