# TSQA Evidence 接管审计（2026-09-18 05:54 UTC）

## 版本与未提交状态

- 当前分支：`main`
- HEAD：`d3924bda46c5ee18846e7e9e11301082e19d454e`
- 本轮修改前 tracked patch SHA-256：`a1a7d0c46fdaa44ce13bd1105a2d84ae0a98331d3e19eee62a116b846878e159`
- 本轮修改前 porcelain 状态 SHA-256：`364fa70dc262eeb8c658c7ba035e67ffa31d7e661e71058df3de492457de77c8`
- 已存在但未提交的修复文件共 6 个：`aggregate_abstraction.py`、`extract_qa_reps.py`、
  `run_factorized_qa.py`、`run_level2_probes.py`、`run_native_qa.py`、`qa_adapters/irts.py`。
  本轮不 reset、不 checkout，也不把它们混入新 benchmark 目录。

## 作业与资源状态

| 任务 | 模型 | init | seed | 状态 | 有效性 | ETA/依据 | 下一步 |
|---|---|---|---:|---|---|---|---|
| corrected Level-2 closure | Qwen3-8B official base | P/R | 7/17/27 | COMPLETE | VALID；共享 manifest | 已完成 | 仅汇总 |
| corrected Level-2 closure | Llama-3.1-8B / Llama-3.2-3B / Gemma-2-9B / Gemma-2-2B | P/R | 7/17/27 | COMPLETE | VALID；共享 manifest | 已完成 | 仅汇总 |
| corrected Level-2 closure | Mistral-7B-v0.3 | P/R | 7；17 部分 | RUNNING | 当前输出协议有效；须成对后纳入 | 单个 CPU run 约 15–25 分钟，依实际日志更新 | 保留 PID 1711080 队列，不复制启动 |
| corrected Level-2 closure | DeepSeek-LLM-7B / OLMo-2-7B / OLMo-2-13B / DeepSeek-V2-Lite | P/R | 7/17/27 | QUEUED | 尚无 corrected 完整覆盖 | 取决于前序 CPU 队列 | 原队列自然执行 |
| corrected IRTS factorized | Qwen3-8B / DeepSeek-V2-Lite | P/R | 7 | COMPLETE | VALID_LIMITED；训练型 diagnostic | 已完成 | 作为外部 QA slice，不能写成 native |
| corrected IRTS native | Qwen3-8B official base | pretrained | 7 | COMPLETE | VALID_LIMITED；四条件 parse coverage 不同 | 已完成 | 保留逐题结果，不扩旧无效表 |
| TSQA Evidence v1 数据构建 | deterministic generator | n/a | dataset 20260918 | RUNNING | 锁定前；目标模型成绩尚未查看 | 由本机实测更新 | 全量重算 gold 与干预 gates |

共享用户 `zhangyafei` 下存在其他训练。按命令和父 PID 识别后，本项目只接管
`scripts/run_level2_closure.py`（父 PID 1711080）及其子进程；未终止、降优先级或修改任何其他进程。

审计时资源：64 CPU 核，2.0 TiB RAM（约 1.8 TiB available），8×A800 80GB。GPU 0、4 空闲；
GPU 1–3、5–7 正被其他训练占用。项目共享盘 96% 已用、约 92 TiB 可用；`/tmp` 91% 已用、约
77 GiB 可用。新数据采用压缩数组与 CSV，避免复制 9 GiB 旧表示缓存。

## 结果与缓存判定

| 资产 | 状态 | 判定与路径 |
|---|---|---|
| Level-1 10模型×P/R×3 seeds | VALID | `results/temporal_abstraction/reps/`；修正聚合在 `corrected_v2/level1/` |
| legacy Level-2 | INVALID | `results/temporal_abstraction/relations/`；保留但不得与 corrected 数值拼接 |
| corrected Level-2 | PARTIAL/RUNNING | `results/temporal_abstraction/corrected_v2/relations/`；审计时 33/60 run 文件，完整性按 P/R 对检查 |
| corrected QA reps | VALID_LIMITED | Qwen/DeepSeek seed 7；身份 JSON 与 P/R cache 分离 |
| Qwen3TS 冒名缓存 | INVALID_QUARANTINED | `corrected_v2/quarantine/model_identity_mismatch/` |
| corrected factorized IRTS | VALID_LIMITED | 两模型、seed 7、四条件；语言分支固定为 pretrained cache |
| corrected native IRTS | VALID_LIMITED | official Qwen seed 7；left padding 修正，仍须按全分母和 parse rate 并报 |
| 新 TSQA Evidence | BUILDING | 独立写入 `results/tsqa_evidence/v1/`，不复用或覆盖旧 QA 输出 |

## 边界

本轮不改论文、不投稿、不申请付费资源、不推送公共仓库。`CODEX_TSQA_EVIDENCE_PLAN.md` 已按
748 行完整读取，附件 SHA-256 为
`b144a66389365bdaebca8c5cee80d51bdcc441e2becbc2c36cf2cff4d3f756af`。
