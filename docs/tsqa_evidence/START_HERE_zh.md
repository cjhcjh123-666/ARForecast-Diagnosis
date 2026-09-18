# Real-TSQA Evidence v1 审查包 — START HERE

状态：**锁定实验矩阵完成（120/120）**。本包用于研究设计、benchmark 与有效实验审查；不修改论文，
不投稿，不申请付费资源，也不推送公共仓库。

## 研究问题

1. 语言模型预训练使哪些真实时序信息更容易被提取？
2. 面对同一份时序的不同问题，模型能否选择并使用相关证据？

主 benchmark 使用经过来源核验的真实记录窗口，覆盖 biomedical、mobility、
energy、environment 四个域。每个 evidence family 包含同一证据上的多个问题，
并在可验证的四格结构中测量相关事实变化后的正确更新、无关事实变化后的正确
保持，以及同一时序切换问题后的答案区分。

## 当前规模与协议

- 264 个 evidence families，2,112 个 items。
- train/val/test families：193/29/42。
- Track R（未修改的真实窗口）：1,584 items。
- Track I（derived-from-real 干预）：528 items，单独报告。
- 主协议 SHA-256：
  `9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c`。
- Interface A 补充协议 SHA-256：
  `63af96cbd7bf167882f8fe9b39ea22a1abcc2e6e35c85f832ba7ef8b5cad6a57`。
- benchmark gates：29/29 PASS；benchmark 单测 20/20 PASS；deadline P0
  回归测试 12/12 PASS。
- 锁定实验矩阵：120/120 完成；A/B 覆盖预注册 10 个模型、seeds 7/17/27，
  C 覆盖预注册 5 个模型的 free 与 constrained 两种接口。

## 三个接口

- A：受控属性读出。固定 final-layer 表示，任务特定线性 head；P/R 使用相同
  样本、head 初始化和顺序。
- B：question-conditioned supervised QA diagnostic。question/options 只进入固定的
  pretrained question branch；temporal branch 只含时序与时间信息；head 显式建模
  二者交互。
- C：native QA。free generation 与 constrained choice scoring 分开报告。

A/B 的监督训练预算与 C 的零样本生成预算不同，不能把绝对分数直接解释成同一
能力尺度。数值属性读取也不等于未来数值预测。

## 如何阅读

1. 先看 `docs/20_REAL_BENCHMARK_SPEC_V1_LOCKED.md`、
   `docs/22_INTERFACE_A_ADDENDUM_LOCKED.md` 和 `protocol_lock.json`。
2. `manifests/families_candidate.jsonl` 提供真实窗口及 family 结构；
   `manifests/item_manifest.jsonl`、`gold/`、`options/`、`interventions/` 提供可复算
   的问题、答案和干预关系。
3. `validity/` 是 gates、数据来源和 checkpoint 审计；真实样例也见
   `review/candidate_examples.jsonl` 与 `docs/21_HUMAN_REVIEW_EXAMPLES.md`。
4. `predictions/` 是逐题输出；`tables/` 是从逐题输出重算的统计。
5. `claim_evidence_matrix.csv` 把每条当前主张绑定到原始文件、metric、模型、seed
   与协议版本。`docs/31_failures_and_boundaries.md` 记录失败和不能外推的边界。

## 当前最稳妥的读法

已完成模型不支持“预训练普遍提高所有真实时序 QA”的统一结论。已观察到的
P−R 差异依赖模型、任务与接口；Llama-3.1-8B 在同样本 Track R 上甚至表现为
A 负、B 正的方向分离。native free 与 constrained 的有效输出率和准确率也存在
明显差异。完整十模型 A/B 矩阵仍不支持普遍正增益；以上是可复算的选择性现象，
不是“模型理解”或共享因果机制。

TSB-UAD 的 `MBA_ECG801_data.out` 因上游记录/通道、重采样、点标签映射和再分发
条款未完全闭合，保持 `LINEAGE_PARTIAL`，未进入 confirmatory native benchmark。

## 复现

完整命令见 `reproduction_commands.sh`。每个 runner 会先核验协议哈希，并在目标
文件已存在时跳过，避免覆盖有效结果。GPU 设备必须在运行前按服务器实时占用
选择；不要重启整套矩阵。
