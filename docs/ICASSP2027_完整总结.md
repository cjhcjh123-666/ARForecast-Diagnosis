# ICASSP 2027 论文——设计 + 实验完整总结

> 本文件是全部设计、实验与结论的唯一权威汇总。更新时间：2026-08-13。
> 细节数据在各实验的 JSON / 表文件里，见文末「产物清单」。

---

## 1. 论文命题（一句话）

**"Recognize, Don't Generate"** —— 在 LLM 时序预测里，语言预训练的本质是
**时序结构先验**（会"认识"动力学），而不是数值生成能力（"生成"很脆弱）。

- 论文：`paper/icassp2027/main.tex`（已按官方 **spconf** 模板写成 4 页两栏，主图留半页占位）
- 目标：ICASSP 2027（截稿 **2026-09-16**，Toronto）
- 与批判派直接对话：Tan et al., NeurIPS 2024《Are LMs Actually Useful for TS Forecasting?》

## 2. 核心设计（可控性保证）

| 维度 | 设定 |
|---|---|
| 模型 | 本地 Wave-MoE Qwen3-8B（Qwen3TS，只用语言主干） |
| 文本格式 | `History: +0.12 -0.04 ...`（两位小数，clamp [-9.99,9.99]）|
| 训练 | LoRA r=8 于 q/k/v/o，teacher forcing + prompt 掩码，free-running 贪心评测 |
| 窗口 | context=64, horizon=16；256 train / 64 test；5 epochs；仅训练段归一化 |
| 唯一变量 | 初始化：预训练权重 vs 随机权重（同架构/同 tokenizer/同窗口/同 LoRA） |
| 数据 | 合成 sine；ETTm1/ETTh1 的 OT 通道 |

Router/probe 组件：
- 冻结 LLM probe：最后 token 的 last-layer hidden state 到线性 softmax probe
- 特征 floor：9 个手算统计量（斜率、线性 R2、波动率、lag-1 自相关、主周期/强度、谱熵、谱峰集中度、极差比）
- 动力学种类：trend / periodic / local(AR1) / mixture(趋势+正弦) / regime(窗口内 trend 到 periodic)
- 轻量专家：trend=线性外推、periodic=自相关周期检测+季节naive、local=稳定阻尼 AR(5)、cross=跨通道 ridge

## 3. 实验总览

| 实验 | 内容 | 一句话结论 |
|---|---|---|
| E1 | 受控预报消融（3 数据集 x 2 初始化 x 3 seeds） | 预训练比随机 MSE 降 2.2-25 倍 |
| E2 | 冻结结构识别（5 类动力学） | pretrained 96.3% > random 90.6%；特征 floor 99.0% |
| E3 | 同窗口"识别 vs 生成"配对 | 识别 96.7% vs 生成 parse 仅 26.7%、MSE 5.5x oracle |
| E4 | 零样本路由（3 干净类 到 未见 mixture/regime） | pretrained 67.8% vs feature 12.8% vs random 28.9% |
| E5 | 真实数据零样本路由（ETTm1/ETTh1） | LLM 路由避开 local 陷阱；特征路由塌陷 |
| E6 | patch 级动态路由（4x4 段） | regime 窗口 -33%；平稳窗口不获益（诚实边界） |

## 4. 各实验详细结果

### E1 受控预报消融（3 seeds: 7/17/27）

| 数据集 | 预训练 MSE | 随机 MSE | 相对降幅 |
|---|---|---|---|
| synthetic_sine | 0.0094 +- 0.0006 | 0.0212 +- 0.0011 | 125% |
| ETTm1 | 0.0099 +- 0.0017 | 0.2453 +- 0.1366 | 2380% |
| ETTh1 | 0.0487 +- 0.0097 | 0.2896 +- 0.2059 | 495% |

预训练 parse 率全 1.0、方差极小；随机初始化方差大、ETTh1 有 parse 失败。
结论：正面反驳 Tan 2024——在 PEFT（LoRA）微调现代 LLM 的场景下，语言预训练明确有用。

### E2 冻结结构识别（5 类）

| 方法 | 总准确率 | trend | periodic | local | mixture | regime |
|---|---|---|---|---|---|---|
| 特征 + logistic | 0.990 | 1.00 | 0.97 | 0.98 | 1.00 | 1.00 |
| 冻结 pretrained Qwen | 0.963 | 1.00 | 0.94 | 0.94 | 0.94 | 1.00 |
| 冻结 random Qwen | 0.906 | 1.00 | 0.96 | 0.78 | 0.88 | 1.00 |

结论：干净合成数据大家都高；预训练优势体现在分布外（E4），而非分布内分类。

### E3 同窗口"识别 vs 生成"（40 窗口分层抽样）

| 指标 | 数值 |
|---|---|
| 冻结 probe 识别准确率 | 96.7% |
| 冻结贪心生成 parse 率 | 26.7% |
| 生成完整者的 MSE | 1.854 |
| oracle 专家 MSE（同窗口） | 0.334（约 5.5x 差距） |

结论：识别鲁棒、生成脆弱的直接证据。

### E4 零样本路由（train 3 干净类 到 未见 mixture/regime，3 seeds）

| 方法 | 对 oracle 命中率 | MSE | Soft MSE |
|---|---|---|---|
| Oracle（上界） | — | 0.446 | — |
| Best single（trend） | — | 0.500 | — |
| Uniform | — | 1.287 | — |
| 特征路由 | 0.128 | 1.278 | 1.235 |
| 冻结 random probe | 0.289 | 1.079 | 1.056 |
| 冻结 pretrained probe | 0.678 | 0.739 | 0.723 |

结论：语言预训练的表示能零样本迁移到未见动力学；特征 floor 和随机同架构都不行。
诚实备注：本组 trend 主导 oracle，best-single 仍有竞争力；路由的价值在"识别迁移"。

### E5 真实数据零样本路由（300 窗口，无适配）

ETTm1（专家 MSE：trend 0.032 / periodic 0.014 / local 0.720 / cross 0.016；oracle 0.010）：

| 方法 | 对 oracle 命中 | MSE | 路由分布 (T/P/L) |
|---|---|---|---|
| 特征路由 | 0.060 | 0.657 | 11/18/271（塌进 local） |
| 冻结 random probe | 0.677 | 0.014 | 0/300/0（退化恒选 periodic） |
| 冻结 pretrained probe | 0.517 | 0.020 | 82/218/0 |

ETTh1：特征路由 5%/0.748（292/300 选 local）；pretrained probe 54%/0.041；
random probe 54%/0.039（同样退化恒选 periodic）。

诚实结论：两个 LLM 路由（任意初始化）都避开 local 陷阱，但均质真实序列上
"恒选 periodic"的退化策略已接近最优，语言预训练的优势只在多样/未见动力学（E4）。

### E6 patch 级动态路由（horizon 切 4x4 段，冻结 LLM probe 每段独立路由）

| 动力学 | 整体-LLM | Patch-LLM | Patch oracle |
|---|---|---|---|
| regime | 0.817 | 0.546（-33%） | 0.280 |
| local | 1.635 | 1.305（-20%） | 0.624 |
| mixture | 0.731 | 0.860 | 0.229 |
| periodic | 0.280 | 0.999 | 0.086 |
| trend | 0.001 | 0.065 | 0.001 |
| overall | 0.693 | 0.755 | 0.244 |

结论：动态路由在窗口内动力学变化时真正有用（regime/local）；平稳窗口因 32 步 patch
结构证据弱而变差（诚实边界）。LLM patch 路由整体（0.755）比特征 patch 路由（1.144）好 34%。

## 5. 关键结论（论文叙事）

1. 严格对照下，语言预训练大幅帮助预测（MSE 降 2.2-25 倍）。
2. 其本质是时序结构先验：冻结 LLM 的表示零样本迁移识别/路由未见动力学，
   手算特征与随机同架构都不行。
3. 识别鲁棒、生成脆弱（96.7% vs 26.7% parse / 5.5x 误差，同一批窗口）。
4. 结构先验在异质动力学上最值钱（跨窗口：E4 零样本；窗口内：E6 patch 路由）。

## 6. 诚实局限（论文已写明）

- 合成 probe 数据仅 3 个种子；窗口规模小（256/64）。
- 干净平稳合成动力学上，手算特征就够强（99.0%），LLM 优势集中在分布外。
- 均质真实序列上"恒选周期"退化策略已接近最优，语言优势不在那里。
- AR 是干净平稳窗口上的"万能模型"（正弦满足低阶递推），specialist 专家集合
  在分布内不总能赢 AR——这本身是"路由价值在 OOD"的佐证。
- patch 路由仅对窗口内动力学变化有效。

## 7. 产物清单

结果文件（均在 results/icassp/）：
- E1_table.md / E1_summary.json —— E1 汇总表
- probe/summary.json、probe_seed17/、probe_seed27/ —— E2/E3/E4 各 seed
- ettm1_router/probe_summary.json（v1）、probe_summary_v2.json（含随机对照）、
  etth1_router/probe_summary.json —— E5
- patch_routing/summary.json（特征版）、patch_routing/llm_summary.json（LLM 版）—— E6

脚本：
- scripts/run_icassp_forecast_sweep.sh —— E1（18 个 Qwen LoRA 训练）
- scripts/run_frozen_probe.py —— E2/E3/E4（带缓存与增量保存）
- scripts/compute_probe_results.py —— 从缓存算 E2/E4（纯 CPU）
- scripts/aggregate_probe.py —— 多 seed 汇总
- scripts/run_probe_ettm1.py（支持 ettm1/etth1/etth2）、scripts/run_ettm1_router.py —— E5
- scripts/run_patch_routing.py（CPU 特征版）、scripts/run_patch_routing_llm.py（LLM 版）—— E6
- scripts/summarize_icassp.py —— E1 汇总

代码模块：
- data/dynamics.py —— 5 类带标签合成动力学
- analysis/features.py —— 9 维手算时序统计量
- analysis/linear_probe.py —— softmax 线性 probe
- models/experts.py —— trend/periodic/local/cross 专家（local 带稳定性阻尼）

文档：
- docs/ICASSP2027_完整总结.md —— 本文件（唯一权威汇总）
- docs/icassp2027_results.md —— 实验数字速查
- docs/icassp2027_outline.md —— 论文规划与格式说明
- paper/icassp2027/main.tex —— 论文（spconf 模板，4 页，主图占位）

## 8. 下一步

1. 你本地编译 main.tex 通读版面（需 spconf.sty + IEEEbib.bst，来自官方模板）。
2. 画主图（fig:main 占位框已留好，建议：(a) 识别准确率对比 (b) 零样本路由 MSE）。
3. 引言/相关工作按 ICASSP 风格润色。
4. （可选）补第 4 个 probe 种子 / 混合"平稳则整体、非平稳则 patch"策略。

---

## 9. 投稿前补充实验（2026-08-23，按审稿意见补）

### E1b. 小模型公平性对照（Qwen3-0.6B / 1.7B，ETTm1，2x2：预训练/随机 x LoRA/全参）

| 模型 | 方式 | 预训练 MSE | 随机 MSE | 随机/预训练 |
|---|---|---|---|---|
| Qwen3-0.6B | LoRA | 0.0113 | 0.1490 | 13.2x |
| Qwen3-0.6B | Full FT | 0.0118 | 0.1413 | 12.0x |
| Qwen3-1.7B | LoRA | 0.0112 | 0.4881 | 43.6x |
| Qwen3-1.7B | Full FT | 0.0147 | 0.1415 | 9.6x |

结论：random+LoRA 确实吃亏（1.7B 尤甚，0.488）；但 **random+全参微调依然追不上预训练**（仍差 10-13x）——预训练增益不是 LoRA 伪影。回答了 reviewer 的 E1 公平性质疑。

### E4b. 非 LLM 学习型路由基线（同 3 类训练 → OOD mixture/regime）

| 路由 | 对 oracle 命中 | MSE |
|---|---|---|
| MLP on raw 窗口 | 37.5% | 0.802 |
| MLP on 9 特征 | 28.1% | 1.088 |
| 线性 on raw | 34.4% | 0.817 |
| 特征 floor（logistic） | 12.8% | 1.278 |
| 冻结 random Qwen | 28.9% | 1.079 |
| **冻结 pretrained Qwen** | **67.8%** | **0.739** |

结论：学习型非线性分类器（37.5%）远低于冻结预训练 LLM（67.8%）——OOD 迁移优势被进一步隔离，不是"随便一个监督分类器都行"。

### E5b. 多域真实数据零样本路由（300 窗口，无适配）

| 数据集 | 特征路由 MSE（对oracle命中） | 预训练 probe MSE（命中） | 最优单一专家 | 备注 |
|---|---|---|---|---|
| ETTm1 | 0.657（6%） | 0.020（52%） | periodic 0.014 | 特征塌进 local |
| ETTh1 | 0.748（5%） | 0.041（54%） | periodic 0.039 | 特征塌进 local |
| electricity | 0.342（19%） | 0.374（42%） | local 0.343 | 唯一 local 最优，LLM 略差 |
| weather | 0.482（32%） | 0.082（44%） | periodic 0.025 | LLM 避开 local 灾难 |
| exchange_rate | 1.591（5%） | 0.028（54%） | periodic 0.027 | 特征塌进 local（1.59!） |

结论（诚实）：LLM probe 路由在 4/5 数据集上显著优于特征路由（后者在 3/5 上塌进 local 灾难）；LLM probe 系统性避开 local 专家；但 electricity 上 local 是真最优，LLM 的 local-avoidance 略伤。随机 probe 常退化成"恒选周期"（均质数据上近最优）。

### E3b. 受控生成（语法约束解码，Qwen3-0.6B，120 窗口 24/类）
- 方法：语法受控解码（每个位置强制 空格/符号/数字/. /数字/数字），parse 率=100%，彻底排除"只是格式失败"
- 结果：
  - 识别准确率：**98.3%**
  - **约束生成 MSE：2.473（95% CI 2.17-2.78），parse 100%**
  - oracle 专家 MSE：0.401 → **约束生成误差仍为 oracle 的 6.2 倍**
  - 非约束生成（40 窗口子集）：MSE 4.40
- 结论：格式正常化后误差依然巨大——"识别强、生成弱"不是输出格式的假象
