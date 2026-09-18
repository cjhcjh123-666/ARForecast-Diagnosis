# Real-World Evidence-Family TSQA v1.0（LOCKED）

状态：**在任何正式 target-LM test 之前锁定**。锁文件为 `configs/tsqa_evidence/real_v1_locked.yaml`，具体文件哈希见 `results/tsqa_evidence/real_v1/manifest_sha256.txt`。

## 研究问题与证据边界

本 benchmark 回答两个问题：语言模型预训练使哪些真实时序信息更容易被提取；面对同一份时序的不同问题，模型能否选择并使用相关证据。probe/readout 可读性不写成模型“理解”，native QA 与 supervised diagnostic 分开报告，相关性不写成共享因果机制。

Track R 是未经本项目修改的真实观测窗口。Track I 是在真实窗口上执行最小、可复算变换后的 `DERIVED_REAL` 行为测试，始终保存 `derived_from_real=true`、源记录和变换坐标，不能称为未经修改的真实观测。

## 数据与规模

使用四个来源和四个域：PTB-XL 1.0.3（biomedical）、Melbourne Pedestrian Counts（mobility）、UCI/Monash electricity hourly（energy）和 Australian temperature/rain observations（environment）。正式 manifest 各取 66 个独立 group，共 264 groups/families；split 为 193 train、29 validation、42 test。

每个 family 有 A–F 六类问题。核心 target/control 在 `x` 与 `T(x)` 上形成四格，故总计 2,112 QA items，其中 Track R 1,584，Track I 528。具体 group 与不均衡 split 数量透明保留，主聚合按 domain macro，不能用 pooled 数量让大来源支配结论。

PTB-XL 按官方 patient-respecting stratified fold 划分；其他来源以 SHA-256 对 source group 固定分配。先划 group 再切窗口。同一 group 的窗口、问题、干预和选项全部继承同一 split。

## 问题与 gold

- A Global Structure：固定索引的最小二乘趋势方向。
- B Numeric Reading：前后半段均值比较；它是当前值/统计量读取，不是未来值预测。
- C Temporal Localization：四个连续等长区间中的最大值位置。
- D Temporal Relation：另一通道均值关系，或同通道峰峰值关系。
- E Multi-evidence Composition：目标均值关系和控制统计关系的四种组合。
- F Conditional Reasoning：先选均值更高的半段，再定位该半段内最大值。

gold 只由四位小数的实际可见序列通过确定性程序计算。比较使用对应整数刻度，避免浮点末位改变语义关系。LLM 不生成 gold。

## 四格与干预

`t0=x+q_target`、`c0=x+q_control`、`t1=T(x)+q_target`、`c1=T(x)+q_control`。每条 family 必须满足 target acceptable singleton sets 不相交且 control semantic gold 完全相等。

唯一锁定干预是 `local_constant_offset`：只在目标通道均值较低的半段增加固定 offset。对多通道数据，control 读取未修改通道；对单通道数据，control 读取 offset 不改变的峰峰值。所有关系从序列化后的 `x/T(x)` 重算，未通过则丢弃 family。

## 语言、序列化与答案位置

正式问题和选项为 English。evidence 使用 canonical compact JSON，保留 timestamp、channel、unit 和四位小数 values。任何条件均不得截断；超长输入应在 preflight 中报错。Full 与 shuffled 复用完全相同的数值词元精度，shuffled 只重排完整观测行，timestamp grid 保持固定。

semantic set 先确定，再以 SHA-256 和分层贪心约束在 `domain × task × split × cell` 内平衡 A/B 或 A/B/C/D。t0/t1 共享同一个问题、semantic set 和 option order；c0/c1 同理。正式评测后禁止修改 option manifest。

## 接口 A/B/C

A 是 frozen final-layer 的受控属性线性读出，只回答 representation 中信息是否可读。

B 是 question-conditioned supervised diagnostic。question/option branch 固定为该模型 official pretrained checkpoint；P/R 只改变 temporal representation 来源。两个分支共享同一 family、split、字节相同的 question/option cache、head architecture、优化预算、minibatch order 和 seed。temporal branch 只含时间、通道、单位和值。`QuestionEvidenceScorer` 通过非线性 MLP 联合 projected question-option 与 temporal vectors，不能退化为与问题无关的 temporal classifier。head seeds 固定为 7/17/27。

C 是 official base checkpoint 的 native zero-shot QA。主结果采用 deterministic free generation 与 strict parser；constrained-choice log-prob 单独作为补充，不与主结果择优合并。条件为 full、question-only 和 shuffled。parser 仅接受单个允许字母或严格的 `Answer: X`，解释文本和越界字母均为 invalid/incorrect。

## 模型矩阵

B 固定十个模型：Qwen3-8B、Llama-3.1-8B、Llama-3.2-3B、Gemma-2-9B、Gemma-2-2B、Mistral-7B-v0.3、DeepSeek-LLM-7B、DeepSeek-V2-Lite、OLMo-2-7B、OLMo-2-13B。P/R 与 seeds 7/17/27 全部报告，不按增益筛选。

C 固定五个 official base checkpoint：Qwen3-8B、Llama-3.1-8B、Gemma-2-9B、DeepSeek-V2-Lite、OLMo-2-7B。先完成整个 locked core test，不按分数决定扩展。

## 指标与聚合

主指标是 four-cell behavioral test 的 FCJS；必须同时报告四格 accuracy、TUS、CIS、QSJA、VOR、条件成功率以及 prediction-change diagnostics。Question-only 用于检查模板/位置先验，不能仅用 `FCJS_full > FCJS_question-only` 声称理解 evidence。

聚合顺序固定为 family → group → task → domain → domain macro。CI 以 patient/sensor/client/station 为 cluster，2,000 次 bootstrap，seed 20260918；少于 20 个独立 group 的分层标记不稳定。连续数值扩展遵守预注册 tolerance、valid-only MAE/NMAE 与 full-denominator success；NMAE 方向固定为 lower-is-better。

## 变更策略

模型测试后不得因结果修改数据、问题、干预、选项、指标、模型矩阵、prompt 或 parser。真正实现 bug 升级为 v1.0.1，写 CHANGELOG，并重跑全部受影响模型。
