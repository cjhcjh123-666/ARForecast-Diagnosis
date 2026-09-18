# Real TSQA v1：任务卡（pre-lock）

固定研究问题：预训练使哪些真实时序信息更容易提取，以及模型能否面对同一证据按问题选择相关事实。

## 评测单位

一个 evidence family 由一个真实 group 中的窗口 `x`、同一窗口上的多种问题和可选的受控变体 `T(x)` 构成。同一 family 当前提供六类问题：

| 代码 | 能力 | 当前可复算实例 | gold 来源 |
|---|---|---|---|
| B | Numeric Reading | 前/后半段哪段均值更高 | 序列化可见值的算术均值 |
| C | Temporal Localization | 四个等长区间中最大值位于哪里 | 序列化可见值的 argmax；固定并列规则 |
| D | Temporal Relation | 另一通道哪半段均值更高，或同通道哪半段峰峰值更大 | 可见值的均值或 max-min |
| F | Conditional Reasoning | 先找均值更高的半段，再在该半段定位最大值 | 两步确定性程序 |
| A | Global Structure | 整个窗口的线性趋势总体向上或向下 | 固定索引上的最小二乘斜率符号 |
| E | Multi-evidence Composition | 同时判断目标均值半段与控制统计量半段 | 两个独立可复算事实的四种组合 |

A–F 均在当前 16 个真实样例中出现。E 使用固定四选一 semantic set；它要求同时读出两个事实，而不是预测数据集类别。

## 四格核心

- `t0 = x + q_target -> a`
- `c0 = x + q_control -> b`
- `t1 = T(x) + q_target -> a'`
- `c1 = T(x) + q_control -> b'`

硬约束是 `G_t0 ∩ G_t1 = ∅` 和 `G_c0 = G_c1`。同一问题在两种 evidence 下共享完全相同的文字、semantic candidates 和 candidate order。判断基于模型实际看到的四位小数序列重新计算。

## 域限制

- PTB-XL：只回答可见导联、电压和区间统计；不把统计差异解释成诊断。
- Pedestrian：值是每小时行人计数；sensor 是独立 split group。
- Electricity：本地小时派生文件只称 `reported electricity load value`；在聚合算子核实前不把它写成 kWh。
- Australian weather：只用 observation 的 `T_MEAN/T_MAX`；含任何零值的候选窗口排除，以避免把 zero-imputation 当观测。

## 三种接口

A 受控属性读出、B question-conditioned supervised QA 和 C native QA 分表报告。只有同一 family、同一语义目标和同一 split 才能直接比较。A/B 的训练预算与 C 的 zero-shot/few-shot 生成预算分别记录，绝不把监督 head 与 native QA 当成同等监督条件。
