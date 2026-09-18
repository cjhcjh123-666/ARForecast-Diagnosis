# 真实时序 Evidence-Family Benchmark：指标草案

状态：**DRAFT / 在任何正式模型测试前锁定**。

## 1. 评测单位

family `i` 包含四个单元：

| 单元 | 时序证据 | 问题 | gold 约束 |
|---|---|---|---|
| `t0` | `x` | `q_target` | `a` |
| `c0` | `x` | `q_control` | `b` |
| `t1` | `T(x)` | `q_target` | `a' != a` |
| `c1` | `T(x)` | `q_control` | `b' = b` |

令 `C_i,s` 表示模型在 family `i` 的单元 `s` 上是否给出正确的语义答案。无效输出、无法解析输出和拒答均计为错误，同时另报有效输出率。

## 2. 主指标

### Four-cell Joint Success（主要指标）

`FCJS = mean_i(C_i,t0 * C_i,c0 * C_i,t1 * C_i,c1)`

一个 family 的四题必须全部正确才记 1。它同时要求模型读对原始事实、随相关证据更新，并在无关变化下保持答案。该指标不被单一高频答案轻易抬高。

## 3. 必须同时报告的拆解指标

### Cell Accuracy

分别报告 `Acc_t0`、`Acc_c0`、`Acc_t1`、`Acc_c1`。另外可以给四者的 macro average，但不能只报告这个平均值。

### Target Update Success

`TUS = mean_i(C_i,t0 * C_i,t1)`

因为构造 gate 已保证 `a' != a`，两题都答对就意味着模型正确更新了目标答案。单独统计“预测是否改变”不能替代 TUS。

### Control Invariance Success

`CIS = mean_i(C_i,c0 * C_i,c1)`

因为构造 gate 已保证 `b' = b`，两题都答对才表示正确保持。始终输出同一错误答案不算成功。

### Question-Switch Joint Accuracy

`QSJA = (1 / (2N)) * sum_i [C_i,t0*C_i,c0 + C_i,t1*C_i,c1]`

它测量同一份证据下切换问题后，模型能否同时回答两个不同事实。两个问题的答案字母可以碰巧相同；评价依据是语义 gold，不要求输出字母必须不同。

### Valid Output Rate

`VOR = 可解析且落在允许答案集合中的输出数 / 总输出数`。

VOR 不替代准确率；无效输出在准确率中计错。

## 4. 诊断指标

- `TUS | t0 correct`：原始目标题答对的 family 中，变换后仍答对的比例。
- `CIS | c0 correct`：原始控制题答对的 family 中，变换后仍答对的比例。
- `Target prediction-change rate` 与 `Control spurious-change rate`：只作行为诊断，必须与 gold-correct 指标并列，不能作为成功率。
- 错误矩阵：按问题类型、数据域、证据操作、答案位置和变换类型分层统计。

## 5. 连续数值题

选择题采用 exact semantic accuracy。连续数值题同时报告：

- 容差内准确率：`abs(pred - gold) <= max(abs_tol, rel_tol*abs(gold))`；容差在测试前按单位和测量精度锁定。
- MAE。
- NMAE：`mean(abs(pred-gold) / scale_task)`，其中 `scale_task` 在训练数据上预先固定。

NMAE 是 lower-is-better，统计、表格排序和显著性方向不得反转。若为了展示定义 `1-NMAE`，必须另命名，不能继续叫 NMAE。

## 6. 聚合、置信区间和随机种子

- 先在 family 级计算，再按问题类型和数据域 macro-average，避免题目较多的数据源占主导。
- bootstrap 的抽样单位是患者、设备、站点或原始记录 group；同一 family 的四题与所有变体不可拆开抽样。
- A/B 监督接口按每个 seed 分别报告，并给 seed 均值、标准差和跨 group 的 95% bootstrap CI。
- C native QA 固定 decoding；若采用随机 decoding，固定并公开生成 seed，不能把多次采样当独立 family。
- 不发布一个任意加权的总 leaderboard 分数。主结果是 FCJS，TUS、CIS、QSJA 解释其组成。

## 7. 三种接口与预训练贡献

A（受控属性读出）、B（question-conditioned supervised QA）和 C（native QA）分别报告上述指标。只有使用同一 family、同一语义目标和同一 split 时才直接比较。

对 pretrained/random-init 配对，报告：

`Delta_M = M_pretrained - M_random_init`

其中模型架构、样本、split、问题向量、读出协议和 seed 必须匹配。Delta 按 family/group 做配对置信区间；完整报告所有预注册任务，不按 Delta 正负筛选。

监督接口与 native QA 的训练预算不同，因此二者的绝对分数是接口能力对照，不能解释为同等监督下的模型优劣。

## 8. 答案位置平衡

- 二选一任务在每个 `domain × task_type × split` 内使 A/B 数量尽量各 50%。
- 四选一任务在同一分层内使 A/B/C/D 尽量各 25%。
- 先确定语义 gold，再由 `hash(family_id, question_id, protocol_version)` 生成固定选项排列；禁止根据模型结果改排列。
- 同一 family 的 `x`、`T(x)` 和所有问题共享 split。选项重排不能产生跨 split 的近重复样本。
- 同时报随机选择基线和该分层的多数答案基线，并检查模型预测字母分布。

## 9. 异常标注任务的额外要求

每个异常数据源必须记录：原始数据集与版本、原始记录 ID、输出文件哈希、采样率与是否重采样、通道/单位、异常的领域语义、点级标签生成规则、数据许可、处理脚本版本、group/split 键。

这些信息不完整时：

- 可以按点号构造内部设计 demo；
- 不得把点号换算成秒；
- 不得把异常扩大解释为临床诊断；
- 不纳入需要领域定义的 native QA 主结果；
- 可以保留在 supervised diagnostic，明确模型获得了何种异常定义和训练监督。
