# Real TSQA v1.0：有效性报告

状态：**PASS（29/29 automated gates）**。正式 target model 结果在构造、验证和 lock 过程中均未运行或查看。

## 数据与结构

| 项目 | 数量 |
|---|---:|
| real domains | 4 |
| datasets | 4 |
| independent groups / families | 264 / 264 |
| train / validation / test families | 193 / 29 / 42 |
| QA items | 2,112 |
| Track R / Track I items | 1,584 / 528 |
| tasks A/B/C/D/E/F | 264 / 528 / 264 / 528 / 264 / 264 items |

每个域正好 66 个 family。test 的独立 group 为 biomedical 7、mobility 9、energy 13、environment 13，因此域内 CI 均应标注 `UNSTABLE_FEW_GROUPS`；总体 42 个 test groups 不能消除单域样本量限制。

## 核心 gate

以下均 PASS：source provenance 与 license 状态；selected dataset × task eligibility；group leakage=0；exact/near duplicate；gold 重算；serialized-input 一致性；target sets 不相交；control invariant；paired candidate set/order；答案位置平衡；question-only evidence 删除；隐藏标签与 semantic answer token 检查；干预坐标；channel/unit；数值精度；manifest referential integrity；item/gold keys；deterministic oracle；metric/parser unit tests。

机器结果：

- `results/tsqa_evidence/real_v1/validity/formal_candidate_validation.json`：23/23。
- `results/tsqa_evidence/real_v1/validity/overall_validity.json`：29/29。
- `python -m unittest tests.tsqa_evidence.test_real_metrics tests.tsqa_evidence.test_real_protocol -v`：12/12。

## 答案位置

四格全体计数：

| cell | A | B |
|---|---:|---:|
| t0 | 134 | 130 |
| c0 | 130 | 134 |
| t1 | 130 | 134 |
| c1 | 130 | 134 |

任务整体：A 为 A/B=131/133；B 为 264/264；C 为 A/B/C/D=65/66/69/64；D 为 260/268；E 为 A/B/C/D=66/67/65/66；F 为 133/131。正式约束在更细的 `domain × task × split × cell` 内验证 max-min≤1，完整分层计数见 `tables/answer_position_counts.csv`。

## 非模型 baseline

test 上 deterministic oracle 的四格 accuracy、FCJS、TUS、CIS、QSJA、VOR 全为 1.0。训练集答案位置 majority baseline 在 42 个 test family 上 FCJS=0、TUS=0、CIS=0.4762、QSJA=0.2381，全体 336 test items accuracy=0.4345。随机二选一的期望 FCJS=0.0625、TUS/CIS/QSJA=0.25。

这些 baseline 只验证构造和位置先验。majority 的 FCJS=0 来自同一 target 问题在 t0/t1 的正确位置相反，不能被写成模型使用 evidence 的结果。

## 边界

- TSB-UAD ECG 保持 `LINEAGE_PARTIAL`，未进入正式 manifest。
- Paderborn 的许可和官方 64 kHz 信息已确认，但本地 MAT schema 未成功解压复核，未进入 v1.0。
- electricity_hourly 只称 `reported load value`；不把派生小时值擅自写成 kWh。
- weather 只使用 observation T_MEAN/T_MAX，任何含零值窗口被排除，以避免 zero-imputation 混入观测。
- 当前没有 short-numeric item；numeric tolerance gate 记为预注册 N/A，而不是伪造数值测试。
