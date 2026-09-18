# TSQA Evidence v1 Data Card

- 用途：诊断冻结语言模型表示中可读的已观测时序属性，以及模型能否按自然语言问题选择相关证据。
- 来源：本项目确定性合成生成器；无第三方个人数据，无外部 LLM 生成 gold。
- 规模：400 train、100 validation、200 IID test、200 composition-OOD test families；每族 12 个
  base questions，并有 relevant、irrelevant 和 cross-control 条件。
- 独立单位：`family_id`。同一族的序列变换、问题、选项、复述和 companion channels 永不跨 split。
- OOD 含义：相对受控 head 训练划分的背景成分组合；不声称基础 LM 预训练未见过这些模式。
- 可见性：48×4 数值、时间戳、通道名和单位均显式呈现；隐藏生成参数仅在 audit metadata 中，
  不进入 temporal 或 question branch。
- gold：所有 gold 从三位小数的最终可见序列重算。周期、异常和变化点均是明确估计规则，不冒充
  唯一物理真值。
- 限制：合成数据、固定候选算子、文本数值输入；结论不能外推到所有真实时序、所有多模态接口或
  未来值预测。模板尚为 `HUMAN_REVIEW_PENDING`。
- 版本：任何影响 gold、可见输入或有效性的修改都必须升协议版本，旧输出保留并标 INVALID。
