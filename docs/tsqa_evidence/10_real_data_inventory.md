# 真实时序数据盘点（Real TSQA v1）

> 生成命令：`python scripts/real_tsqa_inventory.py`。这是 protocol lock 前的审计，不是 benchmark 发布清单。

## 审计口径

本表区分 `REAL`、`DERIVED_REAL`、`SYNTHETIC` 和 `UNKNOWN`。目录名、任务标签或集合级许可证不能替代数据源级 provenance。`UNKNOWN` 不进入正式 benchmark；无法确认再分发条件的数据只保存 reconstruction manifest，不复制原始值。

机器清单共 297 行。现实属性计数：`{'DERIVED_REAL': 28, 'REAL': 7, 'UNKNOWN': 261, 'SYNTHETIC': 1}`。审计状态计数：`{'AUDITED_LOCAL_METADATA': 56, 'INVENTORIED_NOT_INITIAL_CANDIDATE': 1, 'INVENTORIED_EXCLUDED_UNKNOWN': 218, 'SOURCE_LICENSE_VERIFIED_GROUP_NEEDS_ENCODING': 1, 'LINEAGE_PENDING_HIGH_VALUE_CANDIDATE': 1, 'EXCLUDED_SYNTHETIC': 1, 'SOURCE_LICENSE_GROUP_VERIFIED': 1, 'LINEAGE_PARTIAL': 18}`。

## 首批候选

| 数据集 | 域 | 当前判断 |
|---|---|---|
| PTB-XL 1.0.3 | biomedical | 可进入 candidate；只问可见波形事实，按 patient_id 分组 |
| Monash/pedestrian_counts_dataset | mobility | 可进入 candidate；按 66 个 sensor 分组 |
| Monash/electricity_hourly_dataset | energy | 限制进入 candidate；先只问相对量/报告值，单位聚合语义待补 |
| Monash/temperature_rain_dataset_without_missing_values | environment | 可进入 candidate；仅使用 observation series，排除含零填补的窗口 |

## 明确排除或隔离

- `PLAsTiCC 2018` 是合成挑战数据，不能充当真实世界主 benchmark。
- UCR/UEA 的 158 个本地子集不能按集合名一律判为真实；逐数据集的许可、采样、单位和 group 大多缺失，当前全部隔离。
- UniTS/WaveFormer 镜像是指向项目外 97.9 GB 数据树的符号链接；当前目录没有统一来源/许可清单，先逐条列为 `UNKNOWN`。
- Gravity Spy 本地对象主要是时频表示而非直接 1-D 波形，且许可仍待核验，不进入首版。
- Paderborn 官方已确认 CC BY-NC 4.0、bearing/工况结构以及两条主信号 64 kHz；本机解压工具不支持该 RAR 压缩方法，MAT 通道名与单位尚未本地复核，因此暂不进入 candidate。
- TSB-UAD 的 `.out` 是派生格式。代码的 Apache-2.0 不能替代上游数据许可；详见 lineage 专页。

## 已知重叠

Autoformer CSV、Monash TSF、UCR 与 TSB-UAD 在 Timeseries-PILE 中是不同打包视图，部分来源重叠。正式 benchmark 只保留一个经核验的 canonical source，并按原始 group 去重。

## 待完成后才能锁协议

1. 澳洲 weather verification 与 Paderborn 许可已核验；Caltrans traffic 的来源级再分发条件仍待核验。
2. 为候选源计算/保存最终文件哈希，并固定源版本。
3. 先固定 group split，再构造窗口；不得从窗口随机切 split。
4. 对 electricity_hourly 明确 15 分钟到小时的聚合算子及单位。
5. TSB-UAD 若无法补齐逐文件 lineage，则只保留内部 exploratory，排除 native headline 与公开原始数据。
