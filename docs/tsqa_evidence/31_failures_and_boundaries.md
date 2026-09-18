# Failures and boundaries

1. `MBA_ECG801_data.out` remains `LINEAGE_PARTIAL`. Its exact upstream ECG
   record/channel, resampling, point-label mapping, and data redistribution terms
   were not established. It is excluded from confirmatory native QA and from the
   public benchmark payload; no seconds or 360 Hz claim is made for that file.
2. The first formal runner launch failed before model loading because the shared
   Python environment exposed a scikit-learn package without SciPy. A narrow,
   fail-closed compatibility guard now bypasses only Transformers' unused
   assisted-generation metric import. No prediction was produced by the failed
   launches and no protocol input changed.
3. The earliest Qwen prediction writer omitted the convenience fields `split`,
   `track`, and `cell`. All specified prediction fields, unique item identifiers,
   prompts, responses, gold, and parse results were preserved. Aggregation joins
   these three fields from the locked item manifest and verifies gold/options;
   later outputs write them directly. No model rerun is required for this
   metadata-only omission.
4. Electricity questions refer to the reported load values in the source file.
   Until upstream aggregation semantics are independently confirmed, results are
   not described as household-level physical energy consumption.
5. The locked formal matrix is complete (120/120), including ten-model A/B and
   five-model native C coverage. It supports the bounded conclusion that effects
   vary across the prespecified models, tasks, and interfaces; it does not support
   a universal pretraining benefit. No task or model was selected by score sign.
6. Gemma-2-9B free generation produced 0/1,008 strictly valid outputs because it
   continued or echoed prompt fragments instead of emitting one allowed letter.
   Raw responses are retained. The first constrained run used batch 8 and OOMed
   before writing any result; it was safely rerun with batch 1, which changes
   memory/throughput only and leaves per-candidate log-prob scoring unchanged.
7. DeepSeek-V2-Lite's pinned official remote code uses three legacy
   `DynamicCache` accessors removed by the installed Transformers version. A
   narrow compatibility layer restores `seen_tokens`, `get_max_length`, and
   `get_usable_length` with their old read-only semantics. Failed attempts wrote
   no predictions. The repaired free run completed all 1,008 items, but none
   passed the exact-one-letter parser because responses commonly continued into
   another question. This is reported as output-interface failure; constrained
   choice scoring remains a separate run.
8. Interface A's parent protocol fixed the model matrix and comparison but did
   not specify optimizer details. Before any A result existed, the addendum
   `configs/tsqa_evidence/real_v1_interface_a_addendum.yaml` locked the linear
   head, normalization, optimizer, seeds, and paired P/R ordering. Its SHA-256
   is `63af96cbd7bf167882f8fe9b39ea22a1abcc2e6e35c85f832ba7ef8b5cad6a57`.
   A results must cite both protocol hashes.
