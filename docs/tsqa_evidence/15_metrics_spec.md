# Real TSQA v1：指标规格（pre-lock）

实现：`tsqa_evidence/real_metrics.py`。正式锁定前仍可修正实现 bug，但不能依据目标模型分数改指标。

## 四格指标

令 `C_s` 为 cell `s ∈ {t0,c0,t1,c1}` 是否给出正确语义答案。无法解析、拒答和越界答案均记错。

- `FCJS = mean(C_t0 C_c0 C_t1 C_c1)`，四格行为测试 joint success，是主指标。
- `TUS = mean(C_t0 C_t1)`。
- `CIS = mean(C_c0 C_c1)`。
- `QSJA = mean((C_t0 C_c0 + C_t1 C_c1)/2)`。
- `Acc_t0/Acc_c0/Acc_t1/Acc_c1` 分别报告。
- `VOR` 是可解析且属于允许集合的输出占比，不替代准确率。
- `TUS | t0 correct` 与 `CIS | c0 correct` 保存 numerator、denominator 和 value。
- target prediction-change 与 control spurious-change 只在两个输出均有效时作为行为诊断，并显式报告 denominator。

必须满足 `FCJS ≤ TUS`、`FCJS ≤ CIS`、`FCJS ≤ QSJA` 以及 `FCJS ≥ max(0,TUS+CIS-1)`。

## 连续数值题

容差成功条件为 `abs(pred-gold) ≤ max(abs_tol, rel_tol*abs(gold))`。`abs_tol`、`rel_tol` 和只由 train data 得到的 `scale_task` 在模型测试前固定。

必须同时报告 full-denominator tolerance success、VOR、valid N、valid-only MAE 与 valid-only NMAE。parse failure 在 tolerance accuracy 中计错，但不得填 0 或 gold 进入 MAE/NMAE。NMAE 是 lower-is-better。

预训练对照同时保存原始差 `P-R` 与方向对齐 improvement：accuracy 类为 `P-R`，error 类为 `R-P`。

## 聚合与不确定性

顺序固定为 family → group 内 family 平均 → task 内 group macro → domain 内 task macro → domain macro。raw pooled 只作描述。每个 domain/task 同时报 records、groups、families 和 QA items 数。

95% CI 在 patient/device/station/client/original-record 这一最高合理独立 group 上 cluster bootstrap。同一 group 的窗口、四格、问题、干预和 paraphrase 整体抽样；P/R 配对使用相同 bootstrap indices。独立 group 少于 20 时标注 CI 不稳定。

## Question-only

question-only 主要用于答案分布、位置偏差和模板先验。由于 t0/t1 问题相同而 gold 不同，`FCJS_full > FCJS_question-only` 不单独作为 evidence understanding 的证据。
