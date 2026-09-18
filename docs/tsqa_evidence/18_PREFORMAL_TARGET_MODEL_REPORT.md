# Real TSQA v1.0：正式模型评测前汇报

## A. 数据源与排除

正式选择 PTB-XL 1.0.3、Melbourne Pedestrian Counts、UCI/Monash Electricity Hourly 和 Australian Temperature/Rain Observations，覆盖 biomedical、mobility、energy、environment 四个域。

未选：TSB-UAD ECG 因 derived `.out` 的 upstream record/channel/resampling/label mapping 不完整；Paderborn 因本地 MAT schema 与单位尚未复核；Monash weather 因 TSF 缺 station group mapping；traffic 因 source terms/sensor mapping 未完成；UCR/UniTS 大部分条目仍为 dataset-specific lineage `UNKNOWN`；PLAsTiCC 是 synthetic。

## B. Benchmark 规模

264 个独立 group 对应 264 个 family，193/29/42 train/validation/test。每个 family 有 A–F 六题；target/control 在 Track I 追加两个对应 cell，共 2,112 items：Track R 1,584，Track I 528。每域 66 family，不以窗口数冒充独立样本数。

## C. 真实 QA 样例

固定抽取的 16 个真实 family 位于：

- `results/tsqa_evidence/real_v1/review/candidate_examples.jsonl`（完整逐点值）；
- `docs/tsqa_evidence/21_HUMAN_REVIEW_EXAMPLES.md`（人工可读问题、gold、选项、干预和四格关系）。

每域四条且每条覆盖 A–F，因此每个 `domain × task` 有四个审阅实例。这 16 条是 locked manifest 的子集。

## D. Validity gates

29/29 PASS。核心内容包括：group leakage=0、duplicate audit、序列化后 gold 重算、target disjoint、control invariant、P/R 问题候选共享、干预坐标、隐藏标签、答案位置与 manifests referential integrity。详见 `16_validity_report.md`。

## E. 答案位置

t0 A/B=134/130，t1=130/134；c0=c1 均为 130/134。C 的 A/B/C/D=65/66/69/64，E=66/67/65/66。所有正式分层通过 max-min≤1。

## F. 指标测试

12/12 unit tests PASS，覆盖 FCJS/TUS/CIS/QSJA 公式及不等式、parse failure denominator、NMAE lower-is-better、分层 macro、group bootstrap、P/R branch renderer 隔离、Full/Shuffled 精度与 multiset、question-only evidence 删除和 strict parser。

## G. TSB-UAD lineage

最终状态 `LINEAGE_PARTIAL`。`MBA_ECG801_data.out` SHA-256 为 `82e82bd8f9e9c37ef9a080fd9b115c3c1480ad963ba2028bf5f4cba1db14af9f`。它不进入正式 native QA headline 或公开原始数据 release。完整记录见 `lineage/TSB_UAD_MBA_ECG801.md`。

## H. Protocol SHA-256

`9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c`

该 SHA 冻结 18 个数据、manifests、gold/options/interventions、validity、模型清单、renderer/parser、metrics 与协议文件。Git HEAD 为 `d3924bda46c5ee18846e7e9e11301082e19d454e`；工作区新增文件的逐文件 SHA 在 `manifest_sha256.txt`，因此复现不依赖它们已提交。

正式 target model 尚未启动。后续只能按 locked B 十模型 P/R×seeds 7/17/27 与 C 五模型矩阵执行；任何真正 bug 升级 v1.0.1 并重跑受影响结果，不能依分数修改任务或模型。
