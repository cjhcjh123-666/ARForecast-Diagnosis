# Dataset × Task Eligibility（Real TSQA v1）

> 机器可读原表：`results/tsqa_evidence/real_v1/dataset_task_eligibility.csv`。只有 `eligibility=PASS` 的组合可以进入正式 manifest。

| 数据集 | A | B | C | D | E | F | 关键限制 |
|---|---:|---:|---:|---:|---:|---:|---|
| PTB-XL 1.0.3 | PASS | PASS | PASS | PASS | PASS | PASS | Visible waveform facts only; no inferred diagnosis; patient split. Task operates on visible lead/interval statistics. |
| Monash/pedestrian_counts_dataset | PASS | PASS | PASS | PASS | PASS | PASS | Hourly counts and sensor IDs support deterministic visible-value QA. |
| Monash/electricity_hourly_dataset | PASS | PASS | PASS | PASS | PASS | PASS | Use relative/reported-value questions until hourly aggregation unit semantics are confirmed. |
| Monash/temperature_rain_dataset_without_missing_values | PASS | PASS | PASS | PASS | PASS | PASS | Official source is CC BY-SA 3.0 AU; station IDs and observation units exist; zero-imputed spans are excluded and forecast series are out of scope. |
| Monash/weather_dataset | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | TSF omits station mapping, so independent group split is not auditable. |
| Monash/traffic_hourly_dataset | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | Road-occupancy semantics are clear; source terms and sensor mapping remain pending. |
| Paderborn Bearing Data Center | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | Official source confirms CC BY-NC 4.0, 64 kHz main signals, bearing groups and operating conditions; local MAT channel/unit schema still needs extraction. |
| TSB-UAD/ECG | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | PENDING_OR_EXCLUDED | Derived .out-to-upstream record and label mapping unresolved; supervision cannot repair provenance. |

`PASS_RESTRICTED` 表示接口仅能问清单中已核验的可见事实；它不自动令整行正式通过。当前 candidate 仍需完成 source-level license、最终哈希和 family-level validity gates，之后才生成 lock。
