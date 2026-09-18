# Codex 执行指令：以受控时序 QA 诊断预训练贡献与证据使用

## 0. 任务性质、来源和优先级

工作目录：`/9950backfile/chenjiahui/ARForecast-Diagnosis`。

**本轮先做研究问题、benchmark 和有效实验，不以写论文或赶交一份 PDF 为完成标准。**
本文件替代之前方案中相冲突的研究扩展要求；已有有效实验、修正队列和结果保护要求继续有效。

固定研究问题：

> What time-series information does language-model pretraining make accessible,
> and can models use the relevant evidence to answer different questions about the same series?

中文：语言模型预训练使哪些时序信息更容易被提取？面对同一份时序的不同问题，模型能否选择并使用相关证据？

工作定位：**受控时序 QA 诊断与归因研究**。QA 是主要评测接口；探针是解释工具；原 forecasting/routing 实验是下游参考。既不是重新包装 forecasting classification，也不是要求提出新的 SOTA QA 模型。

三个问题分别回答，不能预先合成结论：
1. **可读性**：哪些已观测时序属性能从冻结表示中恢复？预训练改变了多少？
2. **按问题使用证据**：相关证据改变时答案是否正确更新，无关变化时答案是否正确保持？
3. **接口差异**：同一批样本、同一语义目标下，受控读出与原生回答表现如何？

不预设“所有 QA 都有正增益”“深层一定更好”“结构能力一定沿一条阶梯增长”。周期估计等数值属性恢复与未来值预测分别分析，禁止“结构好、数值都不好”的笼统结论。

本文件中的样本量、任务和模型队列是**拟议 v1 设计**，不是已完成实验。现有进度以下方用户回报为起点，但以本地审计核对为准。

---

## 1. 先接上当前工作，不再从零执行旧方案

用户最近回报：
- Level-1 已完成 10 模型 × 2 init × 3 seeds，NMAE 方向已修正。
- 旧 Level-2 协议无效，完整保留；修正结果位于 `corrected_v2`，其余作业可能仍在排队。
- Factorized IRTS 已有 official Qwen、DeepSeek 的 seed 7 结果，方向不一致。
- 修正后的 official Qwen native QA 四条件已完成，但 Full 与 Question-only 解析率相差较大。
- Qwen checkpoint 身份混用、Qwen3TS 误缓存、decoder-only 右侧 padding 已被发现并修复。
- 相关修复当时尚未提交；后续状态需要查本地，不假定仍然如此。

第一步只读审计：
1. `git status`、HEAD、未提交修改、最近日志；不要 reset、checkout 覆盖已有改动。
2. 当前 PID、命令、配置、结果目录、队列进度；共享 Linux 用户下不能仅凭用户名判断进程归属。
3. 检查 `docs/deadline_execution/`、`results/temporal_abstraction/corrected_v2/` 和现有测试。
4. 建立结果状态：VALID / INVALID / PARTIAL / RUNNING / NOT_RUN，附理由与路径。
5. 有效 corrected_v2 队列继续运行；不要复制启动，不重新跑全部 Level-1。
6. 修改前保留补丁快照和版本记录；本轮独立目录，不混入 invalid 缓存。

输出：`docs/tsqa_evidence/00_current_state.md`、`results/tsqa_evidence/v1/inventory.csv`。

不要因为本文件未出现在服务器 `/mnt/data` 中而搜索整个共享文件系统：它是用户提供的指令附件。若附件不可读，明确说明只获得了启动摘要，不能声称已读取完整版。

---

## 2. 最小研究单元：evidence family，而不是一条孤立 QA

定义：
- `family_id`：一个独立采样的证据族，包含一到两条基准序列、其所有扰动版本、全部问题及复述。
- `series_id`：具体可观察序列版本。
- `question_id`：自然语言问题，含查询通道/区间/对象等参数。
- `semantic_answer`：与选项位置无关的答案值或结构。
- `variant_id`：base / relevant_edit / irrelevant_edit 等版本。
- `target_program`：只用于标注和评分的确定性规则，不作为模型隐藏输入。

同一份基准证据至少支持 6 个问题，覆盖至少 3 类能力；问题适用性由事前规则决定。至少包含一对**相同回答类型但不同查询对象或区间**的问题，防止模型只识别“题型”即可通过。

每个 family 至少构造一组交叉测试：

```
                 q_target                 q_control
原始 x            a                        b
变换 T(x)         a' != a                   b' = b
```

例如移动一个局部事件：事件位置问题的正确答案改变，周期问题的正确答案保持。
这是设计示例，不是预设每个移动操作必然保持周期：必须对最终可见输入重新计算两道题的 gold，检查性质后才纳入。

同时构造 question contrast：
- 输入完全相同；
- 问题查询不同区间/通道/关系；
- 回答类型一致；
- 至少一个预先定义的子集 gold 不同；
- 模型必须分别答对，不能以“输出发生变化”代替正确性。

QA 按单题独立调用。多问题共享序列是数据关系，不给模型上一题答案，也不以连续对话引入额外信息。

**成对比较固定问题与候选集。**Relevant/irrelevant evidence edits 中，自然语言问题、查询区间及候选内容原则上保持相同；预先让候选集同时包含原答案与变换后的答案，避免为每个 gold 重新造选项而改变任务。选项置换是另一个独立条件。数值干扰项须去重并避开容差重叠，正确项不得总是最长、最精确或最像模板的一项。

---

## 3. Benchmark 的 v1 范围：覆盖不同操作，不追求题量

内部目录名 `tsqa_evidence/v1`，先不宣称公共 benchmark 名称或首次提出。

以下六类、每类两个核心目标为默认设计。允许在**未评测目标 LM 的工程验证阶段**发现定义不可测并提出等价修订；修订须书面记录。协议锁定后不能根据模型分数改题。

| 类别 | 两个核心目标 | 主要答案类型 | 标注要求 |
|---|---|---|---|
| A 全局结构 | 指定区间趋势方向；主要重复间隔 | 类别、整数 | 趋势阈值和周期估计约定固定，说明是观测统计还是潜在参数恢复 |
| B 数值信息 | 指定区间均值；两个区间的均值差 | 标量 | 直接来自可见数值；保留单位与尺度，不能先除掉所问信息 |
| C 时间定位 | 最强局部偏离所在区间；主要变化点所在区间 | 位置、区间 | “局部偏离”“变化点”给定具体评分规则和有效区间，不混同普适异常真值 |
| D 时序关系 | 两条序列的重复间隔比较；指定通道间领先/滞后 | 类别、延迟 | 包含正、负、近零且有事前 tie 规则；明确符号约定，不由 pair 顺序泄漏答案 |
| E 多信息组合 | 重复结构与局部事件的联合描述；前后区间结构变化描述 | 组合选择、结构记录 | 每个被问属性均可单独验证，不能仅用生成 family 名称充当组合推理 |
| F 条件式推理 | 根据可观测条件选择区间后定位极值；指定事件间的先后关系 | 位置、类别 | 条件写进自然语言问题，规则可独立执行；不是预测哪个未来专家会赢 |

这些是**能力维度**，不是已经验证的认知层级。

B 类必须保留：它区分“当前数值读取/计算”与原有“未来外推”。新 QA 主体不围绕预测或插值；本轮不强制重新训练原 forecasting/imputation 管线。

### 3.1 为每个任务输出 task card

在 `01_task_spec.md` 明确：
- 自然语言示例、模板与完整查询参数。
- 模型实际能看到哪些通道、时间戳、数值与元信息。
- ground truth 类型：观测确定函数 / 有噪声的潜在属性 / 未来实现。
- 精确定义、候选空间、tie/不确定性处理、数值容差、有效区间。
- 正负样本、答案位置与任务适用性分布。
- 相关/无关变换及其应有答案变化。
- 简单基线和独立参考实现。

主 QA v1 优先使用**可观察、可复算的属性定义**。例如“按所规定的估计规则得到的主要周期”，不能悄悄等同于“无误差恢复噪声中的真实物理机制”。

已有 latent-parameter probes 另列为潜在属性恢复实验，不抹去其价值，也不与新 QA gold 偷换概念。

### 3.2 可观测性是硬约束

- 所有 gold 最后都针对**最终序列化再反序列化的可见输入**校验，检查裁剪、舍入、归一化是否改变答案。
- 问绝对均值或波动大小时必须保留相关尺度；给数值单位，不能把标签含义暗中改为标准化尺度。
- 时间索引约定统一，默认从 0 开始；区间端点规则明确并在需要时写入问题。
- 不根据隐藏注入位置要求模型猜一个从观测无法判断的位置。
- 对噪声潜在标签若不可唯一判定，记录为 latent recovery / ambiguity 分层；不可包装为确定事实问答。
- 非唯一最大值、周期混叠、相关峰并列、近阈值分支：用预先规定的 tie 类或独立的 ambiguous 层；不得看 LM 分数后剔除。
- 高/低难度由 SNR、周期数、事件幅度、关系 margin 等数据参数事前定义，完整报告各层。

---

## 4. 数据生成、划分和规模

### 4.1 复用与新增

复用现有 trend / periodic / local / mixture / event / change-point 生成器和元数据结构，但不要更改原结果对应的数据版本。

新的 evidence families 必须新增采样，与已看过结果的旧池区分。旧样本可用于工程开发和兼容性检查，不冒充新 benchmark 的未见测试集。

每个 family 独立采样一到两条基准序列、背景噪声和变换。不得跨 train/test 借同一条 companion series 组成不同 pair。

### 4.2 默认规模（协议锁定前核算成本）

拟议主 manifest：
- 400 个 train families；
- 100 个 validation families；
- 200 个 IID test families；
- 200 个 composition-OOD test families；
- 每个 family 至少 6 个主问题；
- 测试 family 至少一个经过验证的“相关改变/无关保持”交叉组。

这是以**独立证据族**计数，不把同一序列的六道题说成六个独立时间序列。

样本构建用一个固定 dataset seed，保存最终 manifest；新 benchmark 的 probe/head seeds 为 7/17/27。三个训练或随机初始化 seed 不是三份独立测试集。旧 Level-1 的数据种子定义按原协议保留。

若成本需要调整，必须在获取新测试模型分数前，根据 token 数、运行吞吐和覆盖要求整体调整并锁定，所有模型使用同一 manifest。不要根据预训练增益增删题。

### 4.3 严格的 group split

同一 family 的：
- 原始序列与所有变换；
- 同一潜在过程的重采样后代；
- 所有问题、选项置换、复述和不同答案格式；
- 全部配对 companion series；

必须处于同一个 split。保存 family-level manifest；用确定性稳定哈希派生子 seed，不使用 Python 进程随机化 `hash()`。

先划分再配对；若复用旧池导致共享原序列，按共享血缘构造 group，检查 connected components，防止 pair-level 随机拆分泄漏。

### 4.4 Composition-OOD

“未见”指相对受控 head 的训练集/指定示例，不声称知道基础 LM 预训练时没见过这些模式。

默认候选：训练以 T/P/L 单成分为结构主体，OOD 为 TP/TL/PL/TPL；事件、尺度、噪声等 nuisance 的边际分布尽量匹配。具体组合如需调整，先出覆盖矩阵再锁定。

每个所报告的二分类属性必须检查 OOD 正负支持量。若某标签恒为 1：
- 不把该列算作完整 balanced accuracy；
- 可报告正类 recall、覆盖量；
- exact combination match 与每个属性指标分开；
- 不用恒定列抬高组合宏平均。

IID/OOD 本身可能难度不同。报告原始 P/R 分数、配对差值、SNR/margin 分层；不把较高 OOD 增益自动称作更强抽象能力。

### 4.5 Question generalization

训练、验证、测试分别使用事前审核的自然语言模板族。主测试保留规范模板；额外建立有限的 paraphrase-OOD 视图，复用相同证据，不增加无限新数据。

问题复述可以由语言工具提议，但由任务语法和人工抽查确认语义等价。不得让语言工具生成或裁定实验 gold。

所有题包含真实自然语言问题；task code 仅用于单独控制，不能取代主 QA。

---

## 5. 干预设计：不能再把所有 shuffle 当作“应下降”

每一项干预在 manifest 指定：
- `intervention_type`；
- 被改变的可观察变量；
- `expected_relation` = CHANGE / INVARIANT / RECOMPUTE / EVIDENCE_REMOVED；
- 原答案、新答案、重新标注程序与容差；
- 适用任务及排除规则。

### 主条件

1. **Full**：完整问题 + 原始证据。
2. **Relevant edit**：改变当前问题所需事实，gold 必须随之改变。
3. **Irrelevant edit**：其他信息变化，但当前问题的 gold 保持。
4. **Question-only**：保留原问题、选项和允许元信息，移除全部时序数值及能替代数值的摘要。

Question-only 是缺失证据的诊断，不是一个仍然拥有完整输入的合法任务。不要自动让“无法判断”成为通用标准答案，再与 Full 的正确率混比。

### 次要条件

- 合法选项置换：语义答案不变、标签字母可能变；比较必须先还原语义答案。
- 等价问题复述：保持全部参数与语义。
- 时间反转/通道交换：重新生成时间坐标或重新标注，视任务可能 CHANGE 或 INVARIANT。
- Shuffle：作为任务适用的结构干预；若 gold 已改变，必须重算；不能拿原 gold 给变换后合法 QA 评分。

### 不规则时序特别规则

- 将 `(timestamp, value)` 成对排序/置乱通常只改变展示顺序，不一定改变实际时间关系。
- 保持 timestamp 顺序、重新分配 values 才是另一种证据改变，但 gold 也可能变化。
- 两者命名和结论必须分开。禁止声称“打乱了输入文本顺序，所以一定破坏了时序”。

所有干预在最终可见数值上校验；相关/无关样本是否通过只取决于 ground truth 约束，绝不取决于 LM 是否答对。

---

## 6. 三种测量接口，必须使用可比较的语义目标

### I. 受控属性读出：信息是否可读

- 继续保留旧 Level-1/修正 Level-2 的有效结果。
- 对新 QA 核心属性建立与 QA **同样本、同目标**的小规模读出对照。
- 冻结 temporal backbone；主结果仍可使用已锁定线性读出。
- 明确这是 supervised accessibility diagnostic，不叫完整 QA。
- 不重新开启所有层、所有维度、所有 router 的笛卡尔积扫描。

### II. Question-conditioned supervised diagnostic：能否按问题读出

主输入必须实际依赖问题和候选答案语义，不能每个 task 配一个独立分类器后称为 QA。

#### 6.1 固定语言分支

对每个 architecture：
- 问题/选项向量由该 architecture 的 official pretrained checkpoint 生成；
- P-temporal 与 R-temporal 两个条件引用同一份 `h_question_option` 缓存；
- 用 item ID、option ID、张量 SHA-256 证明两边问题向量完全相同；
- 问题向量包含原问题、查询参数和该候选答案，不含序列实际值/隐藏标签/生成 family。

这控制的是**问题分支不随 P/R 改变**，不保证该分支已经理解了问题，也不意味着 language confound 全部被消除。跨 architecture 的问题编码器不同；跨模型对照仍是各自条件下的效应。

#### 6.2 真正隔离时序分支

`h_series` 只从时序证据及允许的通道、时间、单位信息提取，不包含自然语言问题、候选答案、gold 或 task ID。

分别提取 P/R：同 tokenizer、同序列化、同 attention 实现、同可见数值、同层与池化规则。多通道/双序列的顺序及通道 ID 显式保留。

禁止 `h_x_full = encode(full_question_and_series_prompt)` 冒充独立 temporal representation。

#### 6.3 读出头必须能表达 question × evidence 交互

**禁止把纯线性加性候选打分作为主 QA 头：**
若 `s_k = w_x^T h_x + w_q^T h_(q,k)`，同一题所有候选共享的时序项会在比较中抵消。

拟议统一候选 scorer：

```
u_x  = normalized_fixed_projection(h_series, d=128)
u_qk = normalized_fixed_projection(h_question_option_k, d=128)
score_k = Linear( GELU( Linear( concat(u_x, u_qk), 64 ) ), 1 )
```

- scorer 在所有任务、候选间共享；不是 task-specific heads。
- projection matrix 由稳定 seed 生成，不从测试集 fit；P/R 输入维度相同的 temporal branches 使用同一 projection matrix。
- feature mean/std 仅用相应训练支路 fit 并保存；问题分支统计两边共享。
- 禁止直接对未对齐的随机/预训练空间做 `|h_q-h_x|` 并将它解释为同一语义距离。
- P/R 的 scorer 分别训练，使用同结构、相同初始参数 seed、相同数据顺序、优化和训练预算；不能说训练后的权重也相同。
- 将此解释为“固定语言分支及训练算法条件下，temporal representation 初始化的配对效应”，不是唯一、完全的自然语言预训练因果效应。

默认优化候选：AdamW，lr=1e-3，weight_decay=1e-2，batch=256，100 fixed epochs，head seeds 7/17/27；若已有 corrected_v2 的可靠协议更适合，先选一套在 protocol 中锁定，不能两套取较优。

对参数规模、表示压缩造成的影响需记录。本轮只在一个预先选定模型上允许一项容量/不压缩 sanity check，不能靠无限增加 head 找到正结果。

#### 6.4 QA head 的测试

- 用人工设计的简单输入证明改变问题可以改变对 evidence 的选择。
- 验证不同候选 score 差确实依赖时序输入，不是共享项抵消。
- synthetic known-features QA sanity：使用非答案本身的基础统计特征验证统一 scorer 在简单例子上可学习。
- 同一 dataset 下运行 question-only、raw-context、handcrafted-feature 对照，训练预算同等并报告输入维度。
- gold 完全打乱只作为泄漏检查，不把这一检查直接等同于文献中的完整 selectivity 评价。

### III. Native QA：模型自身能否作答

优先五个预先固定的 base checkpoints：
- Qwen3-8B **official base**；
- Llama-3.1-8B；
- Gemma-2-9B；
- DeepSeek-V2-Lite；
- OLMo-2-7B。

保持 base-model 身份，不悄悄替换为 instruction-tuned 版本。若后来加入 instruct，只能是独立 post-training 对照，不并入 base pretraining 归因。

#### 两个分开报告的回答接口

A. **自由生成、严格评分**：统一要求一个合法选项标签；固定提示、生成长度和停止规则，保留原始输出与完整解析状态。

B. **候选标签概率评分**：对完全相同的题目与候选标签计算条件 log probability，选择最高者。它绕开部分输出格式失败，是独立的 constrained-choice diagnostic，不是对 A 的回答修复。

- 核查候选 tokenization，正确计分整个候选字符串在共同前缀后的条件概率。
- 选项 A/B/C/D 位置在语义层平衡，候选内容不能依靠长度或文风泄漏。
- 不能在两个接口中择优作为同一个分数。
- 固定贪心推理不重复三个 decoding seed 伪装三个独立实验。
- decoder-only 批量生成使用已经验证的 padding / mask / prompt slicing；测试 batch size 1 与混长 batch 一致性。
- 不截断问题、序列、候选后悄悄算错；超长样本记录真实失败原因。

主 native 表使用真实自然语言问题，绝不是固定任务 ID 分类。

本轮主输入是数值/时间戳的文本序列化。结论限于已测输入表示、base checkpoints 和回答协议；不能推广为所有多模态 TLM、图像时序输入或经过专用 TS encoder 对齐后的能力。

### 6.5 数值/区间答案视图

主统计先用选择题保持可比较的 QA 评价；另外在**同一语义目标**的预定子集上增加短标量/时间区间回答，覆盖 B/C/F 类。

- 各模型用同一子集，选择方式与成绩无关。
- 同时报告选择题与短答案，不称完整开放式解释能力。
- 不用 LLM-as-judge 给主分。
- 数值 parse rate、原始误差和预先定义的容差内正确率分别给出。

---

## 7. 受控模型矩阵与预算，不无限扩展

### 已有任务

corrected_v2 Level-2 队列继续；结果按有效性和完整覆盖自然收完。它的最终正负不决定是否保留 QA 研究问题。

### 新 QA 的正式目标

Question-conditioned diagnostic：现有 10 个 base LM，P/R temporal branch，3 seeds；每个模型固定 pretrained question/option branch。

Native：先固定五模型；不默认扩所有模型，不默认加入付费 API、新 checkpoint、外部四大 benchmark。

### 默认 native 子集

从两个测试 split 预先分层选 160 个 families（IID 80、composition OOD 80），每族 4 个主问题及相应 cross-contrast：
- 选择按 task/category/gold/SNR 覆盖规则和固定 seed，不按模型分数；
- Full、Relevant、Irrelevant、Question-only 使用相同基础 item IDs；
- 可解析 generation 与 constrained-choice 在同一 subset；
- 数值/区间短答案视图选其中事前规定的一部分，写清实际 item 数；
- 任务覆盖不足时只在锁定前按覆盖规则调整 family 数并记录，不能评完后补容易题。

### 执行顺序

1. 无目标 LM 的 gold/干预/泄漏/可观测性测试。
2. 两个轻量 smoke models：仅检查加载、prompt、缓存、batch 和 scorer，使用独立 dev families。
3. 锁定 v1 数据、协议、native subset、预算和统计表。
4. 正式先运行固定的五模型；有效后按**预先计划及资源可用性**扩十模型 diagnostic，无论结果正负。
5. ETA 超预算时留下覆盖表及可恢复任务，不谎报完成，不用换模型/删任务制造整齐结论。

使用已有 8×A800，但先识别空闲资源。每个 checkpoint 的 token/s、内存、预计调用量先记录；CPU 线程限制避免 BLAS 超额并发。

不同时全开全部大模型；可复用的 pretrained features 不因 head seed 重算。随机模型 features 的 cache 必须区分 random weight seed/fingerprint。

---

## 8. 数据与模型有效性测试：本轮必须过的 gates

使用明确 assert 和机器可读结果，至少覆盖：

1. official checkpoint 身份、revision、config、权重 fingerprint 与 tokenizer；拒绝隔离中的 Qwen3TS 路径。
2. 同一随机初始化实例用于同一支路 train/val/test、所有问题与所有干预；不能每个 split 重建一套不同随机权重。
3. P/R 问题和候选向量 byte/hash 相同。
4. temporal input 不含 question、options、gold 或 latent metadata。
5. question-only 确实移除了所有直接/派生数值证据。
6. item ID、option ID、gold、筛选后索引一一对应；task subset 后重建索引。
7. 候选分数差对 temporal evidence 有依赖；非加性退化。
8. family-level split 无交集；pair、变换、复述血缘无泄漏。
9. 跨进程/分片读取同一 manifest，无 Python `hash()` 随机 split。
10. 输入序列化 round-trip 不改变 gold；不会把目标尺度归一化掉。
11. 所有相关干预确实 CHANGE；所有无关干预确实 INVARIANT。
12. answer-position、template-answer、length-answer、question-only baseline 分布审核；只查数据结构不能保证无全部偏置，结论不要夸大。
13. OOD 每个报告标签的正负支持量，恒定列不算完整 BA。
14. native left-padding / mask / completion slicing；混长 batch vs singleton 一致。
15. 三个 head seeds 的结果来自真实不同训练 seed，不复制同一文件。
16. error metrics 方向正确：保留 raw delta，并另算 improvement；R² 可为负，不做裁剪美化。
17. lower-is-better 与 higher-is-better 正增益计数正确；单测包含人工已知答案。
18. 失败、超时、未解析均在全分母内，状态区分，不默默丢样本。
19. native scoring 与 generation 的模板差异、分母、候选处理明确记录。
20. 打包数据、predictions、tables 和 figure-data 数值重算一致。

已有修复通过时展示真实测试，不机械重写已工作的代码。有效性错误修复后升协议版本、保留 invalid 输出并重跑所有受影响条件，不仅重跑 P 分支或表现差的模型。

---

## 9. 评价指标：正确使用证据，不是“答案发生变化就算好”

### 单题指标

- 全分母 QA accuracy：无效输出为错误。
- parse rate，valid-only accuracy（辅助，不能与全分母混用）。
- 每任务/每类别 accuracy、class support、适用时 balanced accuracy。
- 数值 MAE/NMAE、区间/位置误差、容差内准确率；不能与分类百分点直接求平均。
- 数值未解析时不填 0、均值或凭空设定的误差。数值误差明确其 valid-only 分母，同时报告全分母的“成功解析且在容差内”比例和解析率。

### 核心配对指标

设 `correct(x,q)` 使用对应版本的 gold。

1. **Relevant-edit pair accuracy**：原始题与相关变换题均正确的比例。
2. **Irrelevant-edit pair accuracy**：原始题与无关变换题均正确的比例。
3. **Question-switch pair accuracy**：同一序列上两道 query-different 问题均正确的比例。
4. **Cross-contrast accuracy**：同一 family 的四格交叉组全部正确的比例。

这些都是全配对分母指标；完整报告每组样本量。

另外可报告：
- semantic prediction change rate；
- semantic invariance consistency；
- 原题答对条件下的更新/保持率。

但这些只能辅助：始终输出同一个错误答案也可能有高 invariance；简单改变标签也可能是假响应。不能以一致性代替正确性。

### 预训练差值

同样本、同训练协议的 P/R：
- 分类 gain_pp = 100 × (score_P − score_R)；
- error improvement = error_R − error_P；
- R² improvement = R²_P − R²_R。

保留 metric direction、两边原值、原始差和对齐方向的改善值。不要再出现 NMAE 降低却统计为“0/10 改善”。

Full − Question-only 只能描述输入证据条件的性能差，不是语言预训练效应；不会自动证明或排除语言先验。解析分母不同、候选位置偏置等要同时查看。

### Interface comparison

仅在同 dataset version、family/item、target semantics、option set 的交集上配对比较。

- supervised readout vs native zero-shot：是不同训练/接口预算下的可用性对照，不是净粹因果分解。
- 新 QA 与旧 forecasting 的相关性仅作探索性描述。
- 少数模型的相关性不能证明共享机制，也不能用“没显著”证明独立。

---

## 10. 统计、预注册和开发隔离

### Protocol lock

开发 gold 和渲染时可以看独立 dev pool 的数据与标签，但不要看正式目标模型的测试分数再设规则。

记录：
- 数据/生成器/模板/容差/干预/划分/native subset；
- 模型/revision/random seed/表示/头/训练预算；
- prompt/decoding/parser/候选 scoring；
- 指标方向、bootstrap unit、检验家族；
- 正式评测前代码 commit、各 manifest SHA-256、时间戳。

这是内部锁定与审计，不冒充第三方盲测。已经见过旧结果，需明确新 v1 是基于已有发现提出的后续验证，不回溯宣称全研究事先预注册。

### 不确定性

- bootstrap 以 `family_id` 为聚类单位，整组问题和变换一起抽样。
- P/R 同步采样相同 family、保持 seed 配对。
- 主要 CI 描述固定 checkpoint 与已运行 seeds 条件下的测试族不确定性；另列三个 seed 的原值/标准差。
- 不把同一序列的多道题、多个干预版本当独立样本。
- 检验家族在锁定时声明；若报告大量显著性，按事先定义的模型×任务 family 做 BH-FDR，同时保留 effect size 和 CI。
- 10 个模型不是随机抽取的全体模型总体，不宣称所有 LLM 普遍如此。

结果支持、不支持、覆盖不够、测量无效分别标注。有效负结果与正结果在研究交付中同等保留。

---

## 11. 如何复用原实验，不能偷换结论

1. Level-1：给出具体属性可读性；周期估计回归是数值属性恢复，不能再说“数值能力无贡献”。
2. corrected Level-2：反映特定表示与合适读出下的关系/组合能力；测量不可表达时先标仪器问题。
3. forecasting family routing：下游结构决策参考，不把它等同于所有 temporal QA。
4. oracle-supervision reversal：反映所测试目标及训练接口的不同；不是单凭反号就证明唯一机制。
5. 旧 expected-risk 实验必须核实实现：`argmin mean(MSE)` 与 `argmax frequency(win)` 不是同一标签，报告实际做了哪一种，不能互换。
6. native numerical generation：按具体 base checkpoint、输入与输出协议解释；不能由一个接口推广到所有语言模型或所有数值任务。
7. 老模型结果必须经 checkpoint/cache 影响范围核验再引用。旧图中的手填近似范围和占位误差线不进入新统计。
8. 新旧不同数据集间的 probe vs QA 分离是跨任务观察；本轮同样本接口对照才用于更精确诊断。

---

## 12. 外部 QA 和 symbolic control 的位置

已有修正 IRTS 继续保留，不删除低解析结果；不要重启旧五模型无效表。

本轮外部验证只选择一个已有、授权、协议明确的 QA slice：优先 corrected IRTS。记录官方任务含义、实际输入变换和原评估分母；不要对没有可靠重新标注程序的公开题做“label-changing intervention”后仍当官方分数。

相同公开题不能训练 head 又作为 zero-shot test；若官方只有 test，不拿它做 supervised 训练。新 QA 上训练后迁移只在输入/答案协议支持时执行，否则报告该比较不适用。

Symbolic question 可作一个有限对照，但必须保留 query channel、time span、operator、threshold 等完整语义参数，不能只剩 `TASK=PEAK_TIME`。

它检验措辞影响，不保证消除了语言知识或自动使 P/R 比较“完全干净”。不把它升级成另一个完整 benchmark。

---

## 13. 工程组织与运行安全

优先复用已验证的通用模块，不强制拆出几十个空脚本。

建议：

```
tsqa_evidence/
  task_specs.py
  family_builder.py
  visible_evidence.py
  gold_programs.py
  interventions.py
  manifests.py
  representations.py
  candidate_scorer.py
  native_qa.py
  evaluation.py
  reporting.py
configs/tsqa_evidence/v1.yaml
scripts/tsqa_evidence.py
scripts/export_tsqa_review_bundle.py
tests/tsqa_evidence/
docs/tsqa_evidence/
results/tsqa_evidence/v1/
```

推荐统一 CLI 子命令：`audit / build / validate / lock / extract / train / native / evaluate / report / export`。

cache key 至少包含：dataset/protocol version、visible input hash、tokenizer、checkpoint revision/weight fingerprint、random seed、dtype、layer/pooling、模板版本、question-option版本。

- atomic write + shard files，merge 去重并核对预期 item IDs；不能以输出文件存在判断完整。
- 准备 item-level completeness manifest，错误可恢复且不得混合协议。
- 保存原始 JSONL/CSV/NPZ，不只存 markdown。
- 不碰其他人的进程、共享 shell 配置、凭据和 Codex 登录。
- 不申请付费 API、不外发测试集、不修改投稿、不自动推送公共仓库。
- 本地阶段 commit 可以做，但只包含本轮许可文件；不擅自提交其他未完成修改或受限数据。
- 人工数据验证若未实际完成，标 `HUMAN_REVIEW_PENDING`，不能由模型签字称“人工审核通过”。

---

## 14. 阶段交付，而非等全部跑完才报告

### Gate 1：设计和有效性

交付：任务卡、真实渲染样例、observable-gold测试、干预四格、答案分布、分组划分、成本估算和拟锁定协议。

默认继续推进无阻塞工程；如果核心任务 gold 不可观察、数据权限不明或预算需新增资源，只暂停相关部分，具体说明需要的决定。不要用“不确定”搁置已可执行模块。

### Gate 2：固定五模型的完整核心结果

完整报告同样本上的：
- QA Full；
- relevant / irrelevant / question-switch paired accuracy；
- P/R question-conditioned diagnostic；
- native generation 与 constrained-choice；
- parse coverage 和匹配分母。

与结果正负无关，按预定资源计划继续十模型 diagnostic。不要因为五模型不一致就删掉该任务。

### Gate 3：十模型诊断与新旧结果联合核验

产出模型×任务×接口×条件的覆盖矩阵、效应表、代表性样例、可支持主张及尚无证据的主张。

本轮不交论文 PDF，不自动改 title/abstract/introduction。可以给研究结论与后续实验优先级，但不能把预想结论写成完成状态。

---

## 15. 最后交付给 ChatGPT 的审阅包（必须可离线审阅）

生成：

`deliverables/TSQA_Evidence_Review_<UTC_TIMESTAMP>.zip`

压缩包解压后顶层只用一个目录。主包目标不超过 50 MB；如超出，把完整逐题预测分片压缩成附加包，主包保留覆盖清单、真实样例和所有汇总，不以采样隐藏差结果。

```
TSQA_Evidence_Review/
  START_HERE.md
  CURRENT_STATE.md
  RESEARCH_QUESTIONS.md
  protocol/
    protocol_v1.yaml
    task_spec.md
    data_card.md
    protocol_lock.json
    amendments.md
    split_manifest.csv
    native_subset_manifest.csv
    model_inventory.csv
  validation/
    validity_report.md
    test_results.json
    checkpoint_cache_audit.csv
    distribution_audit.csv
  data/
    review_examples.jsonl
    review_examples.md
    example_arrays.npz
    item_manifest.csv
  predictions/
    prediction_schema.json
    per_item_*.jsonl.gz
    coverage_matrix.csv
  tables/
    qa_by_task.csv
    pretrained_random_effects.csv
    contrast_pair_metrics.csv
    native_format_and_accuracy.csv
    same_item_interface_comparison.csv
    corrected_level2_summary.csv
    existing_results_provenance.csv
  analysis/
    claim_evidence_matrix.csv
    supported_and_unresolved.md
    failure_cases.md
    next_decisions.md
  figures/
    [结果图.pdf + 同名预览.png + 绘图数据.csv]
  reproduce/
    COMMANDS.md
    environment.txt
    relevant_source/ 或 source.patch
    export_manifest.json
  SHA256SUMS
```

### 15.1 START_HERE.md（读它就能知道这次完成了什么）

用中文，建议不超过 1500 汉字，直接回答：
- 新 QA 实际长什么样？不是计划中的题，而是已生成样本。
- 哪些任务、模型、条件已完成，分母分别多少？
- 哪些属性存在稳定 P/R 差异？
- 模型是否对相关证据正确更新、对无关变化正确保持？
- question-only 和格式失败各影响多少？
- 哪些结论仅适用于 supervised diagnostic，哪些来自 native QA？
- 目前最强证据、反例、未完成环节是什么？
- 下一步最多三项，写明每项要消除哪个具体不确定性。

不要只给 STRONG-GO / NO-GO，也不要按正号数量决定是否“成功”。

### 15.2 可审阅的真实样例

默认 12 个 families，每类能力至少 2 个覆盖例，按事先固定的 task strata 和 hash/ID 规则选取，不按模型成绩挑。

每个样例保留：
- 模型确实看到的完整时序/时间戳/单位；
- 自然语言问题、选项和语义答案；
- relevant/irrelevant edit 的原始与变换后输入；
- 四格 gold、gold 计算解释；
- 实际模型原始输出、解析结果、得分；
- protocol/model/item IDs。

另外补充按明确失败类型自动选取的少量案例，**与代表性样本分开命名**，标注选择规则。

关键不是只给一张曲线截图，而是让审阅者能够重算“为什么这道题的答案该变/不该变”。

### 15.3 逐题预测字段

至少包含：

```
run_id, code_commit, protocol_version, dataset_version,
family_id, series_id, item_id, question_id, task_id, split,
variant_id, contrast_group_id, expected_relation,
model, checkpoint_revision, checkpoint_fingerprint,
temporal_init, random_weight_seed, head_seed,
interface, condition, visible_input_sha256,
question_text, options, semantic_gold, gold_option,
raw_response, semantic_prediction, predicted_option,
parse_status, correct, metric_name, metric_value,
input_tokens, output_tokens, truncated, runtime_error,
question_feature_hash, temporal_feature_hash
```

不适用字段用 null 并说明，不填假的 seed 或缓存 hash。为避免 JSONL 中重复存大数组，可用精确 series_id 指向归档数组，但必须能离线重建对应输入。

公开第三方数据若许可不允许重分发，用原始 dataset ID、精确 item ID、版本哈希和本地可运行恢复程序替代，记录此限制；不伪装成已完整附带。

### 15.4 Claim-evidence matrix

每行包含：
- candidate claim；
- SUPPORTED / DESCRIPTIVE / UNRESOLVED / CONTRADICTED / INVALID；
- exact task/model/seed/interface 范围；
- 原始数据路径及筛选键；
- effect size/分母/CI；
- 仍可能的替代解释；
- 什么新增证据会改变判断。

至少单独审核：
- 属性可读性是否来自预训练；
- 问答是否依赖当前序列；
- question switch 是否有效；
- relevant change 与 irrelevant invariance 是否都成立；
- 有无同样本 probe/native 接口差异；
- 是否支持跨模型泛化；
- 是否支持组合泛化；
- 是否把数值属性估计误写成未来预测能力。

### 15.5 原始结果和旧实验复核

所有图表从机器结果生成。不提交 placeholder 图；没有精确区间就不画误差线。

图沿用用户已认可的论文风格：矢量 PDF、嵌入字体、无栅格对象、简洁坐标轴。Times New Roman 不可用时明确字体替代；不能把字体文件打包。

`existing_results_provenance.csv` 必须列出旧 forecasting/Level-1/Level-2/IRTS 哪些有效、哪些作废、哪些覆盖不完整。不能靠把不同版本数值拼起来制造完整表。

### 15.6 不要打包

- 模型 checkpoint、大型全部 hidden-state caches；
- 密钥、token、登录状态、订阅地址、个人邮件；
- 他人文件或进程日志；
- 无授权的第三方数据全集；
- 失效结果冒充有效结果；
- 未实际完成的人工验证证明。

INVALID 历史结果保留在服务器；主包提供其索引与作废原因，需要审查时可单独导出。

---

## 16. 最终回复格式

最终回复用户时必须给出：
1. 审阅 ZIP 的真实服务器绝对路径与大小，以及可用下载/附件方式；服务器路径不是 ChatGPT 的 sandbox 链接。
2. 当前 git commit 或未提交 patch 哈希。
3. 一张简洁实际覆盖表。
4. 最有证据的三条观察，明确范围；不得把试验计划写成结果。
5. 最重要的两项未决问题。
6. 原始结果 → 汇总 → 图表的复现命令。
7. 哪个文件适合先给 ChatGPT 阅读。

用户应当将 ZIP 下载后上传到本对话。不要让 ChatGPT 猜测服务器本地路径，也不要只让用户转发一段 Codex 总结。

现在开始：先审计实际进度，保护 corrected_v2 队列，再建立 task cards 和可验证的 evidence families；通过工程 gates 后锁定 v1 并执行。研究方向固定，结论由有效证据决定。

---

## 方法论参考（帮助理解设计，不是本项目已完成结果）

- Ribeiro et al., ACL 2020, *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*：能力×测试类型的行为测试思路。https://aclanthology.org/2020.acl-main.442/
- Gardner et al., Findings of EMNLP 2020, *Evaluating Models' Local Decision Boundaries via Contrast Sets*：有意义的小幅输入变化及对应 gold 变化。https://aclanthology.org/2020.findings-emnlp.117/
- Hewitt and Liang, EMNLP-IJCNLP 2019, *Designing and Interpreting Probes with Control Tasks*：区分表示中的可读信息与 probe 本身的学习能力。https://aclanthology.org/D19-1275/

本文档提出的是本项目的候选实施方案；不把这些通用思想声称为新的研究贡献，也不凭文献为本项目尚未跑出的机制结论背书。
