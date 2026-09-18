# TSQA Evidence v1 任务卡

## 共用可见输入与评分约定

每个 family 有 48 个等间隔时间点（`t=0..47`），四个显式命名通道 `sensor_A..D`，单位为
`arbitrary_units`。数值以带符号三位小数呈现；gold 必须从这份文本反序列化后的值重算。
区间均为闭区间，时间从 0 开始。主 QA 是四选一，语义答案先生成，再用稳定哈希置换到 A/B/C/D。

每个 family 有 12 道 base 问题，覆盖 A–F 六类。每个 target 问题有：

- `RELEVANT_EDIT`：问题和候选完全不变，目标 gold 必须改变；
- `IRRELEVANT_EDIT`：可见证据发生变化，目标 gold 必须保持；
- `CROSS_CONTROL`：把同一 relevant 变换交给使用互不相同证据的 control 问题，control gold 必须保持；
- `QUESTION_ONLY`：仅保留问题、候选、通道名、长度、时间范围和单位，不含数值或派生摘要。

### A1 指定区间趋势方向

- 问题：`sensor_A, t=0..23` 的 OLS slope；`>+0.025` rising，`<-0.025` falling，否则 stable。
- gold：可观察确定函数；候选为三类及 `not uniquely determined`。
- relevant：用相反方向强斜率替换目标区间；irrelevant：只平移 `sensor_C`。
- 基线：直接 OLS；难点分层按背景 AR 成分。

### A2 主要重复间隔

- 问题：对 `sensor_A` 全 48 点去线性趋势，比较 lag 6/8/12 的 Pearson autocorrelation。
- gold：规定估计量的观测统计，不声称恢复真实物理周期；并列按固定候选顺序裁决，margin 单独审计。
- relevant：用另一候选 lag 的正弦序列替换 A；irrelevant：只平移 C。
- 基线：同一去趋势自相关实现。

### B1 指定区间均值

- 问题：`mean(sensor_A[0:24])`，四舍五入到 0.5 unit。
- gold：直接可见数值函数；短答案容差预定为 0.25 unit。
- relevant：目标区间整体加减 3 units；irrelevant：只平移 C。
- 基线：算术平均。

### B2 两区间均值差

- 问题：`mean(A[0:24])-mean(A[24:48])`，四舍五入到 0.5 unit。
- gold：直接可见数值函数；短答案容差 0.25 unit。
- relevant：仅改变前半段 A；irrelevant：只平移 C。
- 基线：两次平均再相减。

### C1 最强局部偏离区间

- 问题：四个 12 点区间内，哪个区间含有相对本区间中位数绝对偏离最大的单点。
- gold：规定的观测事件统计，不等同于普适 anomaly ground truth；相同分数按较早区间裁决。
- relevant：在另一 12 点区间注入可见强偏离；irrelevant：只平移 C。
- 基线：逐区间 median 与最大绝对偏差。

### C2 主要变化边界

- 问题：候选边界 12/24/36，各比较边界前后 4 点均值的绝对差，取最大者。
- gold：规定窗口下的观测变化点，不等同于隐藏注入位置；并列按较早候选裁决。
- relevant：在另一候选边界构造可见 level shift；irrelevant：只平移 C。
- 基线：三个局部均值差。

### D1 两通道重复间隔比较

- 问题：分别按 A2 规则估计 A/B，回答 A shorter、B shorter 或 same。
- gold：两个观测估计量的关系；pair 顺序固定且问题显式命名通道。
- relevant：重构 B 的周期或令 B 与 A 相同；irrelevant：只平移 C。
- 基线：两次 A2 后比较整数 lag。

### D2 通道领先/滞后

- 问题：C/D 分别中心化并裁剪到各自 10–90% 分位，搜索 lag `-4..+4` 的最大相关；正值表示 D lag C。
- gold：规定估计量及符号约定；相关峰 margin 单独保存/审计。
- relevant：把 D 重构为 C 的另一个 lag；irrelevant：只平移 B。
- 基线：穷举九个 lag。

### E1 重复结构与局部事件联合

- 问题：同时给出 A2 的 lag 与 C1 的 event interval。
- gold：两个分量都从可见 A 独立重算；主分报告 exact combination，并分开审计各分量。
- relevant：移动强事件；irrelevant：只平移 C。
- 基线：A2 与 C1 结果的笛卡尔组合。

### E2 前后结构变化

- 问题：分别在 `t=0..23` 和 `t=24..47` 估计 lag 6/8/12，回答 after shorter/longer/same。
- gold：两个区间的观测估计量比较。
- relevant：以预定周期对重构两个半段；irrelevant：只平移 C。
- 基线：两个半段分别自相关。

### F1 条件选择后定位极值

- 问题：先按 B 的均值选择较高的半段（并列选前半），再定位该半段内 A 最大值所在 12 点区间。
- gold：可执行的两步观测程序；条件与查询通道都写入自然语言问题。
- relevant：翻转 B 的半段选择；irrelevant：只平移 C。
- 基线：半段均值选择 + argmax。

### F2 两事件先后

- 问题：C/D 各按 C1 找 event interval，比较先后或同区间。
- gold：区间级顺序关系，不使用隐藏事件时间。
- relevant：重构 C/D 的强事件区间关系；irrelevant：只平移 B。
- 基线：两次 C1 后比较区间索引。

## Question contrast 与四格

每个 A/B 类 target 使用 D2 作为 control；D2 和 F2 使用 A2 作为 control。相关变换只改变 target
所需通道，因此四格为 `base-target=a`、`edited-target=a'!=a`、`base-control=b`、
`edited-control=b'=b`。此外，同一 base 序列上的 12 道题构成 question-switch pool；直接比较只在
相同答案类型的预定 pair 内进行，不能用“输出变了”替代两题都答对。

## 人工检查状态

程序 gold、可见性和干预由独立实现与 asserts 检查。自然语言模板的人工审核尚未由项目负责人签字，
因此当前标记为 **HUMAN_REVIEW_PENDING**。
