# Real TSQA v1：候选数据 lineage

本文件记录首批候选的来源链。详细机器字段见 `results/tsqa_evidence/real_v1/source_inventory.csv`。

| 数据集 | 本地对象 | 上游与版本 | group 键 | 采样/单位 | 许可 | 当前结论 |
|---|---|---|---|---|---|---|
| PTB-XL | PhysioNet 1.0.3 ZIP | PTB-XL 1.0.3；归档内含 member SHA256 清单 | `patient_id`，记录为 `ecg_id` | 100/500 Hz，12 导联，mV | CC BY 4.0，归档内 LICENSE | PASS；问题只依据可见波形，不从波形推导诊断 |
| Melbourne Pedestrian Counts | Monash `pedestrian_counts_dataset.tsf` | City of Melbourne，经 Monash archive | `series_name` 对应 66 个传感器 | hourly，pedestrians/hour | Monash record 许可需随发布归档；目前仅发布 reconstruction manifest | PASS candidate；最终 release 仍保存 attribution/record metadata |
| Electricity hourly | Monash `electricity_hourly_dataset.tsf` | UCI 370 个 15-min client series，经 Lai/Monash 形成 321 hourly series | client `series_name` | hourly；上游 kW，小时聚合后的精确单位语义待确认 | 上游 UCI CC BY 4.0；本地派生链需记录 | restricted PASS；只问相对量与“reported load value”，不擅自写 kWh |
| Australian temperature/rain | Monash `temperature_rain_dataset_without_missing_values.tsf` | data.gov.au 2015–2017 verification data，经 Monash 转为 TSF | `station_id`（422 站），另含 `obs_or_fcst` | daily；温度 °C、雨量 mm；文件保留变量语义 | CC BY-SA 3.0 Australia | PASS candidate；只选 observation series，并排除由缺失值填零的窗口 |
| Paderborn bearing | 30 个 per-bearing RAR | Paderborn University Bearing DataCenter；Lessmeier et al. 2016 | bearing code 与工况 | 官方论文称两条主信号 64 kHz；本地 MAT schema 未核 | CC BY-NC 4.0 | PENDING；许可已补齐，但本机 7z 不支持该 RAR 方法，通道/单位仍未本地复核 |
| TSB-UAD ECG | 53 个派生 `.out` | TSB-UAD；上游部分来自 MIT-BIH | 映射不完整 | `.out` 不含 Hz、通道、单位 | 代码 Apache-2.0；派生数据许可不能据此推出 | LINEAGE_PARTIAL；详见 `lineage/TSB_UAD_MBA_ECG801.md` |

## 划分规则

先按患者、传感器、client、station 或 bearing 划分，再从各 split 内切窗口。同一 group 派生的原始窗口、干预、全部问题和选项排列必须在同一 split。连续单序列且没有可信独立 group 的 Autoformer CSV 不作为 confirmatory group-bootstrap 数据源。

## 数据发布规则

benchmark 工程默认只保存源标识、group/record ID、边界、通道、处理配置、gold、问题及哈希。除非来源许可和 attribution 要求全部通过复核，否则不复制原始时序值到公开 release。
