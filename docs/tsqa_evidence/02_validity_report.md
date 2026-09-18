# TSQA Evidence v1 有效性检查

总状态：**PASS**；通过 12，失败 0。

| gate | 检查 | 状态 | 摘要 |
|---:|---|---|---|
| 6 | `unique_item_ids` | PASS | {"n": 43200, "unique": 43200} |
| 8 | `family_level_split_disjoint` | PASS | {"families": 900, "violations": 0} |
| 8 | `twelve_base_questions_per_family` | PASS | {"families": 900} |
| 10 | `serialization_roundtrip_and_hash` | PASS | {"failures": [], "n_checked": 43200} |
| 10 | `gold_recomputed_from_visible_input` | PASS | {"failures": [], "n_checked": 43200} |
| 4 | `temporal_question_branch_isolation` | PASS | {"failures": [], "n_checked": 43200} |
| 11 | `relevant_change_irrelevant_invariant` | PASS | {"failures": [], "pairs": 10800} |
| 8 | `protocol_split_counts` | PASS | {"composition_ood_test": 200, "iid_test": 200, "train": 400, "validation": 100} |
| 12 | `answer_position_distribution` | PASS | {"A": 2619, "B": 2789, "C": 2700, "D": 2692} |
| 13 | `nonconstant_reported_targets` | PASS | {"cells": 48, "constant": []} |
| 5 | `question_only_has_no_observed_values` | PASS | {"failures": [], "n_checked": 2000} |
| 20 | `archive_schema` | PASS | {"keys": ["channels", "family_ids", "timestamps", "values", "variant_ids"], "shape": [900, 25, 48, 4]} |
