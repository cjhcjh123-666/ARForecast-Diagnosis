# Open-LLM 跨家族归因实验 — 阶段状态（2026-09-11）

## 已完成
- 仓库/环境/GPU 审计：`docs/open_llm_exp00_repo_audit.md`（8×A800-80GB，driver 535.261.03，CUDA 12.2）。
- 下载通路打通：hf-mirror 直连可用；模型 revision 已 pin；5 个模型并行下载中
  （DeepSeek-LLM-7B、DeepSeek-V2-Lite、Mistral-7B-v0.3、OLMo-2-7B、OLMo-2-13B）。
- API discovery 完成：`/v1/models` 返回 1235 个模型，已按 family 冻结 ≤15 个（`configs/api_models_frozen.yaml`）；未产生付费调用。
- 统一 Open-LM runner 实现：`scripts/openllm_suite.py`（clean/shuffle、linear+MLP readout、family-label/oracle-label routing、MLP router、pooling、native）。
- **Qwen3-8B 协议对账已完成且完全一致**（见 `docs/open_llm_validation_qwen.md`）：clean 0.981、shuffle 0.640、readout 0.588、family-label bal3 0.833 / bal3n 0.831、routed MSE 0.710；新增 MLP64 readout 0.554。

## 阻塞
- **Llama-3.1-8B / Llama-3.2-3B / Gemma-2-9B 为 gated repo，服务器未设置 HF_TOKEN** → 需提供 token 才能下载；
  不使用非官方镜像替代。其余 P0 模型不受影响。
- 镜像单流约 1.2–1.9 MB/s，5 路并行聚合 ~7.5 MB/s；~90GB 预计还需数小时。
  hf_transfer / aria2c 与镜像不兼容（已记录）。

## 下一步（按执行顺序）
1. 等待下载完成 → 对每个 P0 模型跑 smoke + matched random + A/B/C/C2/R + 10-seed headline。
2. 真实数据 15 数据集 P/R 归因 + BH-FDR。
3. 本地 native generation（base LMs）+ API 直接生成 zoo（需 RUN_PAID_API=1 与预算上限）。
4. 生成 Table A–E、Figure A–E、`docs/open_llm_final_report.md`，并据实决定 claim（Case A–D）。
