# ICLR deadline execution status

Snapshot updated: 2026-09-18 12:32 CST (Asia/Shanghai). Repository HEAD at takeover:
`d3924bda46c5ee18846e7e9e11301082e19d454e` on `main`, equal to
`origin/main`. The tracked tree was clean before the P0 fixes in this run.

The requested `ARForecast_ICLR_deadline_execution.md` is not present in the
repository, `/tmp`, or the accessible home tree. This run therefore follows
the complete priority list supplied in the user message. If the missing file
contains additional constraints, it still needs to be recovered before final
submission lock.

## Live jobs

| Task | Model | init | seed | Status | Validity | ETA from observed throughput | Next step |
|---|---|---:|---:|---|---|---|---|
| Level-1 representations and probes | 10-model suite | pretrained + random | 7/17/27 | Complete: 60 caches + 60 probe shards | Valid raw results; corrected aggregation writes to `corrected_v2` | Complete | Use corrected, metric-oriented table in evidence matrix |
| Level-2 relation/composition, protocol v2 | 10-model suite | pretrained + random | 7/17/27 | 1/30 matched model/seed pairs complete (Qwen seed 7); all remaining runs queued with 4 total bounded workers | Valid protocol; shared SHA-256 manifests and explicit coverage guard | About 5–10 h for 30 paired model/seed runs at observed/concurrent throughput | Continue queue; refresh paired table as runs complete |
| Level-2 relation/composition, protocol v1 | 10 models | pretrained + random | 7/17/27 | Complete: 60 CSVs; legacy workers have exited | **Invalid for claims**: process-random hash, constant-label C1 axes, duplicated R4/R7, unidentifiable R3 | Complete but unusable | Preserve under `relations/`; never merge with v2 |
| IRTS representation extraction and factorized diagnostic, protocol v2 | DeepSeek-V2-Lite + official Qwen3-8B-Base | pretrained + random | 7 | Complete: paired caches and four conditions for both models | Valid protocol after unit tests: fixed pretrained question branch, temporal-only branch, checkpoint identity lock | Complete | Report mixed two-model evidence; do not extrapolate |
| IRTS native zero-shot, legacy | 5 models | pretrained | 7 | Complete, 150 temporal_relationship items/model | **Invalid**: decoder-only prompts were right padded in batched generation; controls also have extraction/precision faults | — | Withhold every legacy native score; corrected official-Qwen rerun uses left padding |
| IRTS native zero-shot, protocol v2 | official Qwen3-8B-Base | pretrained | 7 | Complete: 150 items × four conditions | Valid left-padded, exact-lexeme protocol; parse coverage differs sharply by condition | Complete | Report as one-model interface boundary, not evidence that language priors dominate |
| IRTS factorized QA, protocol v1 | 5 models | pretrained + random | 7 | Complete, 15 CSVs | **Invalid**: random question branch, branch leakage, and filtered-label misalignment | — | Preserve; replace only with v2 outputs |
| Paper evidence matrix and independent draft | all validated evidence | — | — | Main and anonymous supplement sources/PDFs compiled under `paper/iclr2027_deadline/` | Quantitative statements require a matrix row; invalid native/Level-2 claims removed | Draft complete; ongoing evidence updates | Keep incomplete Level-2 coverage explicit; recompile after accepted result updates |

## Host resources

- GPUs: 8 × NVIDIA A800 80GB. GPU 4 was the only safely available device and
  is assigned to the corrected IRTS extraction. GPUs 1–3 and 5–7 are occupied
  by unrelated FastVOS/starVLA work; they were not touched. GPU 0 contains a
  small root-owned allocation and was left alone.
- CPU: legacy Level-2 workers have exited. Corrected closure uses four bounded
  workers (8 BLAS threads each): three for the other nine models and one for
  Qwen seeds 17/27.
- RAM: 2.0 TiB total, 1.8 TiB available at snapshot; swap is fully used (8 GiB).
- Disk: workspace filesystem has 92 TiB free but is 96% used; `/tmp` has 77 GiB
  free. New outputs are small compared with existing caches.

## Deadline clock

- T0: 2026-09-18 11:31 CST.
- Official abstract deadline: 2026-09-18 23:59 AoE = 2026-09-19 19:59 CST.
- Official paper deadline: 2026-09-25 23:59 AoE = 2026-09-26 19:59 CST.
- T0+4h audit checkpoint: 2026-09-18 15:31 CST.
- T0+24h evidence/draft checkpoint: 2026-09-19 11:31 CST.
- T0+48h claim lock: 2026-09-20 11:31 CST.
- Scope freeze: 2026-09-21 11:31 CST (T0+72h is earlier than deadline−72h).
- Checked submission candidate: no later than 2026-09-25 19:59 CST.

Abstract submission, author list, paid compute, and external publication remain
user-controlled actions.

## Completed locally but not committed or pushed

- Five P0 code fixes, the P0 regression test module, and the versioned
  Level-2 manifests.
- Corrected Level-1 aggregate table with improvement-oriented NMAE signs.
- Corrected DeepSeek-V2-Lite and official Qwen3-8B-Base IRTS paired caches,
  eight shared-head diagnostic files, and one corrected native-QA result file.
- The claim--evidence matrix, audit notes, independent anonymous paper draft,
  compiled draft PDF, supplementary source, and compiled supplementary PDF.

The repository was synchronized at takeover, so every item above is new local
work. No commit or remote push has been performed during this run.
