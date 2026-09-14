# Open-LLM 跨家族归因实验 — 最终状态（2026-09-14）

## 1. 一页结论

- **预训练相对同架构随机初始化的"结构可读性"优势真实且跨家族**：family-label OOD 组合路由
  10/10 模型为正 —— Qwen3-8B +31.1pp、DeepSeek-V2-Lite(MoE) +31.4pp、Llama-3.1-8B +22.8pp、
  Gemma-2-9B +20.0pp、Gemma-2-2B +15.0pp、Llama-3.2-3B +14.7pp、Mistral-7B-v0.3 +10.6pp、
  OLMo-2-7B +9.4pp、OLMo-2-13B +8.3pp、DeepSeek-LLM-7B +7.5pp。
- **换成 oracle-label 监督，10/10 全部转负**（−5.6 ~ −21.4pp）：预训练表征组织的是稳定的结构划分，
  而非"单次未来实现出来的专家选择"；family↔oracle 标签在 clean 窗口上只有约 64% 一致。
- **真实数据零样本迁移：预训练同样占优**。9 个已抽特征的 LM 在 15 个数据集上 12–15/15 胜同架构随机
  控制，中位相对 routed-MSE 改善 12–28%（含配对 bootstrap 与 BH-FDR q 值）。
- **数值生成这条路径两边都不行**：本地 base LM 预训练版能解析出数字（parse 0.87–1.00），
  但 MSE 是 oracle 的 4–15 倍、且比 best-fixed 专家更差；同架构随机模型 parse rate 全为 0。
  API 侧 13 个模型严格解析率最高 33%，多数 0–6%。

## 2. 完成度

| 项目 | 状态 |
|---|---|
| 10 个 open base LM × P/R × 3 seeds（clean / shuffle / readout / family-label / oracle-label / MLP-router） | 全部完成 |
| 真实数据（15 数据集 × P/R × 9 模型，零样本路由 + BH-FDR） | 完成（Qwen 走早期 legacy 路径，未并入本表） |
| Native generation（本地 base LM 直接生成数值） | 9/10 本轮完成（Qwen/GPT-2 用上一轮结果） |
| API 直接生成 zoo（13 模型 × 3 seeds × 40 分层窗口，1560/1560 成功） | 完成（只用于 direct generation，不进 attribution 表） |
| 图表 | Figure A/B/C/D + `paper_tables.tex`，全部由 CSV 自动生成 |
| 5-expert 敏感性 | 仅 Gemma-2-2B/9B、Llama-3.2-3B（Qwen 侧另有 5-expert/local-rich） |
| dimension matching（PCA/随机投影）、pooling 敏感性 | 本套未跑 |
| 10-seed headline 复现 | 仅 Qwen（+0.277 [+0.253,+0.304]） |

## 3. 复现与数据一致性修复（本轮）

- Gemma 随机分支 native 崩溃（`torch._dynamo`：gemma2 在 sdpa 下启用 static cache 触发 torch.compile）：
  统一 pretrained/random 的 attention 实现（gemma 用 eager），重跑两个 gemma 模型。
- DeepSeek-V2-Lite native：官方 remote code 调用的 `DynamicCache.get_max_length()` 在 transformers ≥4.49
  已删除 → 加兼容 shim；随机分支 sdpa 不支持 → 回退 eager（与 pretrained 分支一致）。重跑后 pretrained
  结果（parse 0.975 / MSE 3.924）与崩溃前完全一致。
- Table D 分析原先每个数据集都重训一遍 router（每模型 30 次冗余）：改为按 (model, init) 缓存，
  数值完全一致、约 15× 加速。
- 早期"随机初始化特征与 OOD 特征来自不同随机权重"的问题已在本轮之前修好（同进程重导 clean+OOD 特征）。

## 4. 后续（camera-ready 级别，非结论阻塞项）

1. 对 Qwen3-8B 补齐 MLP-router / pooling / dimension-matching，使 10 个模型口径一致。
2. 把 5-expert（family5）敏感性扩到全部 10 个模型。
3. headline 条件扩到 10 seeds（现仅 Qwen 为 10 seeds）。
4. 图表一律由 `scripts/openllm_{make_report,figures,export}.py` 生成，论文数字不得手工改动。
