# Real-TSQA formal job status — completed locked matrix

Updated: 2026-09-18, current live snapshot.

Machine-detected outputs: **120/120 completed**. No Real-TSQA extraction or
head-training job remains active. `results/tsqa_evidence/real_v1/job_status.csv`
contains all 120 locked rows, output states, and reproduction commands.

| task | model | init | seed | status | validity | ETA | next step |
|---|---|---|---:|---|---|---|---|
| locked formal matrix | all prespecified models | all prespecified | all prespecified | COMPLETED | output present; protocol checks pass | DONE | verify package checksums |

Current closed coverage:

| interface component | complete | locked total | note |
|---|---:|---:|---|
| A controlled readout | 30 | 30 | ten models × seeds 7/17/27 |
| B question cache | 10 | 10 | all fixed pretrained question branches complete |
| B temporal cache | 40 | 40 | pretrained plus three random seeds for ten models |
| B supervised head | 30 | 30 | ten models × seeds 7/17/27 |
| C native free | 5 | 5 | complete |
| C native constrained | 5 | 5 | complete |

All final head runs reused the locked caches. No other user's process was
modified or stopped. The review bundle excludes representation caches and model
weights while retaining manifests, per-item predictions, statistics, hashes,
and reproduction commands.
