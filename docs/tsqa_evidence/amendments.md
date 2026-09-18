# Protocol Amendments

## v1 锁定前工程记录

- 2026-09-18：候选方案落地为 48×4 可见序列和 12 个任务；采用三位小数 round-trip 后重算 gold。
- 2026-09-18：A2/E2 明确使用候选 lag 6/8/12 和固定 tie rule；margin 作为 ambiguity 审计字段，
  不在看到目标模型结果后删题。
- 2026-09-18：D1 relevant edit 增加 `sensor_B := sensor_A` 候选，以保证 base 已为最短候选时仍能
  构造可验证的 gold change。该改动发生在任何目标模型测试前。
- 2026-09-18：首次全量验证的 2 个失败来自 gate 实现而非数据：float32 内存值不应与三位小数
  parse 值做二进制 `array_equal`；问题中的合法阈值也不应因碰巧等于单个观测值而算泄漏。验证器改为
  检查序列化文本 round-trip 完全一致，并检查完整 CSV header/观测行。43,200 条可见输入 gold 重算
  在修正前已全部通过；数据 manifest 未改，修正后全量重跑。

正式锁定后若需修改，在此追加版本、原因、受影响输出和完整重跑范围。
