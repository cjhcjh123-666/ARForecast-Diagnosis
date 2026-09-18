# Real TSQA v1：干预卡（pre-lock）

## `local_constant_offset`

用途：改变指定通道的“哪一半均值更高”，同时保持另一通道事实或同一通道的半段峰峰值关系。

程序：

1. 在四位小数序列上计算目标通道前后半段均值。
2. 找到均值较低的半段。
3. 只对该半段、该通道加固定常数；常数至少为原均值差加一个预设 margin。
4. 重新序列化为相同精度。
5. 从序列化后的 `x/T(x)` 分别重算 target 和 control gold。
6. 只保留 target semantic answer 改变且 control semantic answer 不变的 family。

允许变化集合精确记录为 `(target_channel, selected_half)`。时间戳、其他通道、未选择半段均不能改变。每条 Track I 样本保存 `derived_from_real=true`、`source_record_id` 和 `intervention_type`。

## 解释边界

Track I 是由真实记录派生的受控行为测试。它不称为未经修改的 real-world observation。offset 的目的在于得到可验证的反事实四格，并不表示真实采集系统实际经历了该物理过程。

Track R 独立成立：即使某来源不适合安全干预，只要来源、group 与 gold 通过 gate，仍可在相同 evidence 的多问题测试中报告 single-cell accuracy 与 QSJA。
