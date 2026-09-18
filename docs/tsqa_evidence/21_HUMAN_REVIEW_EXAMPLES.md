# Real TSQA v1：真实 family 人工审阅样例（pre-lock）

> 这些是 16 个真实数据构造样例，不是正式 test manifest，也没有任何目标模型分数。完整逐点数值在 `candidate_examples.jsonl`。

有效性状态：**PASS**。来源覆盖：biomedical, energy, environment, mobility。

## `ptbxl_ecg16703_0720`

- 来源：PTB-XL 1.0.3；group `patient_id=16865`；series `ecg_id=16703`；split `train`。
- 通道/单位：[('lead II voltage', 'mV'), ('lead V1 voltage', 'mV')]。源边界：`{'sample_start_zero_based': 720, 'sample_stop_exclusive': 816}`。
- 可见数值预览（前 6 点）：sample_720@100Hz: [0.112, 0.048]; sample_721@100Hz: [0.123, 0.043]; sample_722@100Hz: [0.111, 0.003]; sample_723@100Hz: [0.173, -0.003]; sample_724@100Hz: [0.159, 0.036]; sample_725@100Hz: [0.166, 0.021]
- q_target：lead V1 voltage 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_control：lead II voltage 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `c0=c1=first_half`；固定选项 `{'A': 'first_half', 'B': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，lead V1 voltage 的最大值位于哪个区间？ gold `quarter_3`。
- q_conditional：先找出lead V1 voltage平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `earlier_quarter`。
- q_global：用整个窗口的线性趋势概括，lead V1 voltage总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① lead V1 voltage哪半段均值更高；② lead II voltage在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 1, 'region': 'first_half', 'offset': 0.1134, 'allowed_change': 'only the target channel in the named half'}`；改变 48 个序列化坐标。target changed=YES；control invariant=YES。

## `ptbxl_ecg15470_0336`

- 来源：PTB-XL 1.0.3；group `patient_id=18962`；series `ecg_id=15470`；split `train`。
- 通道/单位：[('lead II voltage', 'mV'), ('lead V1 voltage', 'mV')]。源边界：`{'sample_start_zero_based': 336, 'sample_stop_exclusive': 432}`。
- 可见数值预览（前 6 点）：sample_336@100Hz: [0.052, -0.058]; sample_337@100Hz: [0.078, -0.072]; sample_338@100Hz: [0.138, -0.093]; sample_339@100Hz: [0.186, -0.121]; sample_340@100Hz: [0.253, -0.144]; sample_341@100Hz: [0.315, -0.155]
- q_target：lead V1 voltage 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'B': 'second_half', 'A': 'first_half'}`。
- q_control：lead II voltage 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `c0=c1=first_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，lead V1 voltage 的最大值位于哪个区间？ gold `quarter_4`。
- q_conditional：先找出lead V1 voltage平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，lead V1 voltage总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① lead V1 voltage哪半段均值更高；② lead II voltage在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 1, 'region': 'first_half', 'offset': 0.1355, 'allowed_change': 'only the target channel in the named half'}`；改变 48 个序列化坐标。target changed=YES；control invariant=YES。

## `ptbxl_ecg01502_0288`

- 来源：PTB-XL 1.0.3；group `patient_id=984`；series `ecg_id=1502`；split `validation`。
- 通道/单位：[('lead II voltage', 'mV'), ('lead V1 voltage', 'mV')]。源边界：`{'sample_start_zero_based': 288, 'sample_stop_exclusive': 384}`。
- 可见数值预览（前 6 点）：sample_288@100Hz: [-0.02, 0.117]; sample_289@100Hz: [0.015, 0.081]; sample_290@100Hz: [0.063, 0.044]; sample_291@100Hz: [0.095, 0.013]; sample_292@100Hz: [0.078, 0.01]; sample_293@100Hz: [0.089, 0.025]
- q_target：lead V1 voltage 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=first_half`, `t1=second_half`；固定选项 `{'A': 'first_half', 'B': 'second_half'}`。
- q_control：lead II voltage 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `c0=c1=first_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，lead V1 voltage 的最大值位于哪个区间？ gold `quarter_4`。
- q_conditional：先找出lead V1 voltage平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `earlier_quarter`。
- q_global：用整个窗口的线性趋势概括，lead V1 voltage总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① lead V1 voltage哪半段均值更高；② lead II voltage在控制问题所用统计量下哪半段更大？ gold `target_first_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 1, 'region': 'second_half', 'offset': 0.1359, 'allowed_change': 'only the target channel in the named half'}`；改变 48 个序列化坐标。target changed=YES；control invariant=YES。

## `ptbxl_ecg12328_0144`

- 来源：PTB-XL 1.0.3；group `patient_id=6109`；series `ecg_id=12328`；split `test`。
- 通道/单位：[('lead II voltage', 'mV'), ('lead V1 voltage', 'mV')]。源边界：`{'sample_start_zero_based': 144, 'sample_stop_exclusive': 240}`。
- 可见数值预览（前 6 点）：sample_144@100Hz: [-0.068, 0.128]; sample_145@100Hz: [-0.035, 0.123]; sample_146@100Hz: [-0.08, 0.161]; sample_147@100Hz: [-0.075, 0.165]; sample_148@100Hz: [-0.057, 0.141]; sample_149@100Hz: [-0.071, 0.14]
- q_target：lead V1 voltage 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=first_half`, `t1=second_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_control：lead II voltage 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `c0=c1=first_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，lead V1 voltage 的最大值位于哪个区间？ gold `quarter_1`。
- q_conditional：先找出lead V1 voltage平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `earlier_quarter`。
- q_global：用整个窗口的线性趋势概括，lead V1 voltage总体向上还是总体向下？ gold `overall_downward`。
- q_composition：哪项同时正确描述：① lead V1 voltage哪半段均值更高；② lead II voltage在控制问题所用统计量下哪半段更大？ gold `target_first_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 1, 'region': 'second_half', 'offset': 0.3136, 'allowed_change': 'only the target channel in the named half'}`；改变 48 个序列化坐标。target changed=YES；control invariant=YES。

## `mobility_t36_009792`

- 来源：Monash/pedestrian_counts_dataset；group `T36`；series `T36`；split `train`。
- 通道/单位：[('pedestrian count', 'pedestrians/hour')]。源边界：`{'start_index_zero_based': 9792, 'stop_index_exclusive': 9840}`。
- 可见数值预览（前 6 点）：2015-01-21 00-00-01+9792*hour: [23.0]; 2015-01-21 00-00-01+9793*hour: [7.0]; 2015-01-21 00-00-01+9794*hour: [19.0]; 2015-01-21 00-00-01+9795*hour: [65.0]; 2015-01-21 00-00-01+9796*hour: [206.0]; 2015-01-21 00-00-01+9797*hour: [676.0]
- q_target：pedestrian count 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=first_half`, `t1=second_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_control：pedestrian count 在前半段和后半段中，哪一段的峰峰值更大？ 语义 gold `c0=c1=first_half`；固定选项 `{'A': 'first_half', 'B': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，pedestrian count 的最大值位于哪个区间？ gold `quarter_1`。
- q_conditional：先找出pedestrian count平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `earlier_quarter`。
- q_global：用整个窗口的线性趋势概括，pedestrian count总体向上还是总体向下？ gold `overall_downward`。
- q_composition：哪项同时正确描述：① pedestrian count哪半段均值更高；② pedestrian count在控制问题所用统计量下哪半段更大？ gold `target_first_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'second_half', 'offset': 222.1089, 'allowed_change': 'only the target channel in the named half'}`；改变 24 个序列化坐标。target changed=YES；control invariant=YES。

## `mobility_t49_010344`

- 来源：Monash/pedestrian_counts_dataset；group `T49`；series `T49`；split `train`。
- 通道/单位：[('pedestrian count', 'pedestrians/hour')]。源边界：`{'start_index_zero_based': 10344, 'stop_index_exclusive': 10392}`。
- 可见数值预览（前 6 点）：2017-11-30 00-00-01+10344*hour: [67.0]; 2017-11-30 00-00-01+10345*hour: [36.0]; 2017-11-30 00-00-01+10346*hour: [48.0]; 2017-11-30 00-00-01+10347*hour: [25.0]; 2017-11-30 00-00-01+10348*hour: [30.0]; 2017-11-30 00-00-01+10349*hour: [17.0]
- q_target：pedestrian count 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_control：pedestrian count 在前半段和后半段中，哪一段的峰峰值更大？ 语义 gold `c0=c1=second_half`；固定选项 `{'B': 'second_half', 'A': 'first_half'}`。
- q_localize：把窗口依次分成四个等长区间，pedestrian count 的最大值位于哪个区间？ gold `quarter_4`。
- q_conditional：先找出pedestrian count平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，pedestrian count总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① pedestrian count哪半段均值更高；② pedestrian count在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_second_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'first_half', 'offset': 57.5713, 'allowed_change': 'only the target channel in the named half'}`；改变 24 个序列化坐标。target changed=YES；control invariant=YES。

## `mobility_t42_003912`

- 来源：Monash/pedestrian_counts_dataset；group `T42`；series `T42`；split `validation`。
- 通道/单位：[('pedestrian count', 'pedestrians/hour')]。源边界：`{'start_index_zero_based': 3912, 'stop_index_exclusive': 3960}`。
- 可见数值预览（前 6 点）：2015-04-15 00-00-01+3912*hour: [52.0]; 2015-04-15 00-00-01+3913*hour: [33.0]; 2015-04-15 00-00-01+3914*hour: [17.0]; 2015-04-15 00-00-01+3915*hour: [12.0]; 2015-04-15 00-00-01+3916*hour: [13.0]; 2015-04-15 00-00-01+3917*hour: [51.0]
- q_target：pedestrian count 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_control：pedestrian count 在前半段和后半段中，哪一段的峰峰值更大？ 语义 gold `c0=c1=first_half`；固定选项 `{'A': 'first_half', 'B': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，pedestrian count 的最大值位于哪个区间？ gold `quarter_2`。
- q_conditional：先找出pedestrian count平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，pedestrian count总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① pedestrian count哪半段均值更高；② pedestrian count在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'first_half', 'offset': 20.8723, 'allowed_change': 'only the target channel in the named half'}`；改变 24 个序列化坐标。target changed=YES；control invariant=YES。

## `mobility_t25_005856`

- 来源：Monash/pedestrian_counts_dataset；group `T25`；series `T25`；split `test`。
- 通道/单位：[('pedestrian count', 'pedestrians/hour')]。源边界：`{'start_index_zero_based': 5856, 'stop_index_exclusive': 5904}`。
- 可见数值预览（前 6 点）：2013-09-01 00-00-01+5856*hour: [80.0]; 2013-09-01 00-00-01+5857*hour: [11.0]; 2013-09-01 00-00-01+5858*hour: [23.0]; 2013-09-01 00-00-01+5859*hour: [13.0]; 2013-09-01 00-00-01+5860*hour: [33.0]; 2013-09-01 00-00-01+5861*hour: [85.0]
- q_target：pedestrian count 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=first_half`, `t1=second_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_control：pedestrian count 在前半段和后半段中，哪一段的峰峰值更大？ 语义 gold `c0=c1=first_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，pedestrian count 的最大值位于哪个区间？ gold `quarter_2`。
- q_conditional：先找出pedestrian count平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，pedestrian count总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① pedestrian count哪半段均值更高；② pedestrian count在控制问题所用统计量下哪半段更大？ gold `target_first_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'second_half', 'offset': 43.1819, 'allowed_change': 'only the target channel in the named half'}`；改变 24 个序列化坐标。target changed=YES；control invariant=YES。

## `energy_t167_014784`

- 来源：Monash/electricity_hourly_dataset；group `T167`；series `T167`；split `train`。
- 通道/单位：[('reported electricity load', 'reported load value')]。源边界：`{'start_index_zero_based': 14784, 'stop_index_exclusive': 14832}`。
- 可见数值预览（前 6 点）：2012-01-01 00-00-01+14784*hour: [2265.0]; 2012-01-01 00-00-01+14785*hour: [1879.0]; 2012-01-01 00-00-01+14786*hour: [1766.0]; 2012-01-01 00-00-01+14787*hour: [1652.0]; 2012-01-01 00-00-01+14788*hour: [1728.0]; 2012-01-01 00-00-01+14789*hour: [2534.0]
- q_target：reported electricity load 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'B': 'second_half', 'A': 'first_half'}`。
- q_control：reported electricity load 在前半段和后半段中，哪一段的峰峰值更大？ 语义 gold `c0=c1=first_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，reported electricity load 的最大值位于哪个区间？ gold `quarter_4`。
- q_conditional：先找出reported electricity load平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，reported electricity load总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① reported electricity load哪半段均值更高；② reported electricity load在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'first_half', 'offset': 279.6869, 'allowed_change': 'only the target channel in the named half'}`；改变 24 个序列化坐标。target changed=YES；control invariant=YES。

## `energy_t111_020496`

- 来源：Monash/electricity_hourly_dataset；group `T111`；series `T111`；split `train`。
- 通道/单位：[('reported electricity load', 'reported load value')]。源边界：`{'start_index_zero_based': 20496, 'stop_index_exclusive': 20544}`。
- 可见数值预览（前 6 点）：2012-01-01 00-00-01+20496*hour: [152.0]; 2012-01-01 00-00-01+20497*hour: [143.0]; 2012-01-01 00-00-01+20498*hour: [156.0]; 2012-01-01 00-00-01+20499*hour: [146.0]; 2012-01-01 00-00-01+20500*hour: [153.0]; 2012-01-01 00-00-01+20501*hour: [139.0]
- q_target：reported electricity load 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_control：reported electricity load 在前半段和后半段中，哪一段的峰峰值更大？ 语义 gold `c0=c1=second_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_localize：把窗口依次分成四个等长区间，reported electricity load 的最大值位于哪个区间？ gold `quarter_4`。
- q_conditional：先找出reported electricity load平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，reported electricity load总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① reported electricity load哪半段均值更高；② reported electricity load在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_second_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'first_half', 'offset': 84.557, 'allowed_change': 'only the target channel in the named half'}`；改变 24 个序列化坐标。target changed=YES；control invariant=YES。

## `energy_t55_003456`

- 来源：Monash/electricity_hourly_dataset；group `T55`；series `T55`；split `validation`。
- 通道/单位：[('reported electricity load', 'reported load value')]。源边界：`{'start_index_zero_based': 3456, 'stop_index_exclusive': 3504}`。
- 可见数值预览（前 6 点）：2012-01-01 00-00-01+3456*hour: [97.0]; 2012-01-01 00-00-01+3457*hour: [85.0]; 2012-01-01 00-00-01+3458*hour: [80.0]; 2012-01-01 00-00-01+3459*hour: [75.0]; 2012-01-01 00-00-01+3460*hour: [76.0]; 2012-01-01 00-00-01+3461*hour: [73.0]
- q_target：reported electricity load 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_control：reported electricity load 在前半段和后半段中，哪一段的峰峰值更大？ 语义 gold `c0=c1=second_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_localize：把窗口依次分成四个等长区间，reported electricity load 的最大值位于哪个区间？ gold `quarter_2`。
- q_conditional：先找出reported electricity load平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，reported electricity load总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① reported electricity load哪半段均值更高；② reported electricity load在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_second_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'first_half', 'offset': 2.9015, 'allowed_change': 'only the target channel in the named half'}`；改变 24 个序列化坐标。target changed=YES；control invariant=YES。

## `energy_t300_014736`

- 来源：Monash/electricity_hourly_dataset；group `T300`；series `T300`；split `test`。
- 通道/单位：[('reported electricity load', 'reported load value')]。源边界：`{'start_index_zero_based': 14736, 'stop_index_exclusive': 14784}`。
- 可见数值预览（前 6 点）：2012-01-01 00-00-01+14736*hour: [123.0]; 2012-01-01 00-00-01+14737*hour: [123.0]; 2012-01-01 00-00-01+14738*hour: [82.0]; 2012-01-01 00-00-01+14739*hour: [82.0]; 2012-01-01 00-00-01+14740*hour: [82.0]; 2012-01-01 00-00-01+14741*hour: [41.0]
- q_target：reported electricity load 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'B': 'second_half', 'A': 'first_half'}`。
- q_control：reported electricity load 在前半段和后半段中，哪一段的峰峰值更大？ 语义 gold `c0=c1=second_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_localize：把窗口依次分成四个等长区间，reported electricity load 的最大值位于哪个区间？ gold `quarter_3`。
- q_conditional：先找出reported electricity load平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，reported electricity load总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① reported electricity load哪半段均值更高；② reported electricity load在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_second_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'first_half', 'offset': 143.2393, 'allowed_change': 'only the target channel in the named half'}`；改变 24 个序列化坐标。target changed=YES；control invariant=YES。

## `auweather_station11053_256`

- 来源：Monash/temperature_rain_dataset_without_missing_values；group `station_id=11053`；series `station=11053:T_MEAN+T_MAX`；split `train`。
- 通道/单位：[('daily mean temperature', 'degC'), ('daily maximum temperature', 'degC')]。源边界：`{'start_index_zero_based': 256, 'stop_index_exclusive': 288}`。
- 可见数值预览（前 6 点）：2015-05-02+256d: [20.125, 22.5]; 2015-05-02+257d: [20.3708, 22.9]; 2015-05-02+258d: [21.1708, 23.5]; 2015-05-02+259d: [21.8875, 25.4]; 2015-05-02+260d: [23.9083, 28.0]; 2015-05-02+261d: [25.275, 32.9]
- q_target：daily mean temperature 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=first_half`, `t1=second_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_control：daily maximum temperature 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `c0=c1=first_half`；固定选项 `{'B': 'first_half', 'A': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，daily mean temperature 的最大值位于哪个区间？ gold `quarter_1`。
- q_conditional：先找出daily mean temperature平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `earlier_quarter`。
- q_global：用整个窗口的线性趋势概括，daily mean temperature总体向上还是总体向下？ gold `overall_downward`。
- q_composition：哪项同时正确描述：① daily mean temperature哪半段均值更高；② daily maximum temperature在控制问题所用统计量下哪半段更大？ gold `target_first_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'second_half', 'offset': 1.2414, 'allowed_change': 'only the target channel in the named half'}`；改变 16 个序列化坐标。target changed=YES；control invariant=YES。

## `auweather_station58077_352`

- 来源：Monash/temperature_rain_dataset_without_missing_values；group `station_id=58077`；series `station=58077:T_MEAN+T_MAX`；split `train`。
- 通道/单位：[('daily mean temperature', 'degC'), ('daily maximum temperature', 'degC')]。源边界：`{'start_index_zero_based': 352, 'stop_index_exclusive': 384}`。
- 可见数值预览（前 6 点）：2015-05-02+352d: [19.9292, 27.3]; 2015-05-02+353d: [20.3042, 28.1]; 2015-05-02+354d: [21.0542, 26.7]; 2015-05-02+355d: [21.9708, 30.3]; 2015-05-02+356d: [20.9958, 29.3]; 2015-05-02+357d: [18.8167, 25.1]
- q_target：daily mean temperature 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=first_half`, `t1=second_half`；固定选项 `{'A': 'first_half', 'B': 'second_half'}`。
- q_control：daily maximum temperature 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `c0=c1=second_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_localize：把窗口依次分成四个等长区间，daily mean temperature 的最大值位于哪个区间？ gold `quarter_3`。
- q_conditional：先找出daily mean temperature平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `earlier_quarter`。
- q_global：用整个窗口的线性趋势概括，daily mean temperature总体向上还是总体向下？ gold `overall_downward`。
- q_composition：哪项同时正确描述：① daily mean temperature哪半段均值更高；② daily maximum temperature在控制问题所用统计量下哪半段更大？ gold `target_first_half__control_second_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'second_half', 'offset': 1.6135, 'allowed_change': 'only the target channel in the named half'}`；改变 16 个序列化坐标。target changed=YES；control invariant=YES。

## `auweather_station94029_448`

- 来源：Monash/temperature_rain_dataset_without_missing_values；group `station_id=94029`；series `station=94029:T_MEAN+T_MAX`；split `validation`。
- 通道/单位：[('daily mean temperature', 'degC'), ('daily maximum temperature', 'degC')]。源边界：`{'start_index_zero_based': 448, 'stop_index_exclusive': 480}`。
- 可见数值预览（前 6 点）：2015-05-02+448d: [5.1417, 9.6]; 2015-05-02+449d: [7.025, 12.1]; 2015-05-02+450d: [6.1875, 10.3]; 2015-05-02+451d: [7.9292, 11.6]; 2015-05-02+452d: [12.5167, 15.9]; 2015-05-02+453d: [10.3375, 14.5]
- q_target：daily mean temperature 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_control：daily maximum temperature 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `c0=c1=second_half`；固定选项 `{'A': 'second_half', 'B': 'first_half'}`。
- q_localize：把窗口依次分成四个等长区间，daily mean temperature 的最大值位于哪个区间？ gold `quarter_4`。
- q_conditional：先找出daily mean temperature平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `later_quarter`。
- q_global：用整个窗口的线性趋势概括，daily mean temperature总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① daily mean temperature哪半段均值更高；② daily maximum temperature在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_second_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'first_half', 'offset': 2.4274, 'allowed_change': 'only the target channel in the named half'}`；改变 16 个序列化坐标。target changed=YES；control invariant=YES。

## `auweather_station18120_592`

- 来源：Monash/temperature_rain_dataset_without_missing_values；group `station_id=18120`；series `station=18120:T_MEAN+T_MAX`；split `test`。
- 通道/单位：[('daily mean temperature', 'degC'), ('daily maximum temperature', 'degC')]。源边界：`{'start_index_zero_based': 592, 'stop_index_exclusive': 624}`。
- 可见数值预览（前 6 点）：2015-05-02+592d: [17.632, 21.8]; 2015-05-02+593d: [19.712, 26.5]; 2015-05-02+594d: [17.748, 22.1]; 2015-05-02+595d: [19.184, 27.3]; 2015-05-02+596d: [25.556, 38.3]; 2015-05-02+597d: [19.184, 23.7]
- q_target：daily mean temperature 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `t0=second_half`, `t1=first_half`；固定选项 `{'B': 'second_half', 'A': 'first_half'}`。
- q_control：daily maximum temperature 在前半段和后半段中，哪一段的平均值更高？ 语义 gold `c0=c1=first_half`；固定选项 `{'A': 'first_half', 'B': 'second_half'}`。
- q_localize：把窗口依次分成四个等长区间，daily mean temperature 的最大值位于哪个区间？ gold `quarter_3`。
- q_conditional：先找出daily mean temperature平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？ gold `earlier_quarter`。
- q_global：用整个窗口的线性趋势概括，daily mean temperature总体向上还是总体向下？ gold `overall_upward`。
- q_composition：哪项同时正确描述：① daily mean temperature哪半段均值更高；② daily maximum temperature在控制问题所用统计量下哪半段更大？ gold `target_second_half__control_first_half`。
- 干预：`{'type': 'local_constant_offset', 'channel_index': 0, 'region': 'first_half', 'offset': 0.7826, 'allowed_change': 'only the target channel in the named half'}`；改变 16 个序列化坐标。target changed=YES；control invariant=YES。

## 自动 gate

| gate | 状态 | 细节 |
|---|---|---|
| `family_count_12_to_20` | PASS | 16 |
| `four_real_domains` | PASS | ['biomedical', 'energy', 'environment', 'mobility'] |
| `four_examples_per_domain` | PASS | Counter({'biomedical': 4, 'mobility': 4, 'energy': 4, 'environment': 4}) |
| `unique_family_ids` | PASS | no duplicate IDs |
| `group_leakage_zero` | PASS | 16 independent groups |
| `duplicate_near_duplicate_audit` | PASS | no exact duplicate or |r|>0.999999 within dataset |
| `at_least_three_ability_types` | PASS | all families have B/C/D/F |
| `target_acceptable_sets_disjoint` | PASS | all target semantic answers change |
| `control_gold_invariant` | PASS | all control semantic answers remain |
| `gold_recomputation` | PASS | all semantic gold exactly regenerated from serialized evidence |
| `serialized_input_gold_consistency` | PASS | decimal_places=4 |
| `paired_candidate_set_and_order` | PASS | one fixed order per paired question |
| `track_i_marked_derived` | PASS | all interventions identify source record |
| `intervention_allowed_coordinates_only` | PASS | exactly target channel × selected half changed |
| `source_provenance_recorded` | PASS | path, hash/member hash, group and bounds present |
| `license_status_recorded` | PASS | every source records license or manifest-only release mode |
| `no_hidden_label_in_evidence` | PASS | visible channels contain no labels |
| `question_only_evidence_removed` | PASS | question objects carry text/options only |
| `no_answer_token_leakage` | PASS | semantic answer IDs absent from question text |
| `source_domain_units_preserved` | PASS | all visible channels have explicit non-UNKNOWN units |
| `same_numeric_precision` | PASS | fixed 4 decimals |
| `numeric_tolerance_non_overlap` | PASS | N/A for current categorical comparison/localization examples; required when short-numeric subset is added |
| `answer_position_balance` | PASS | 96 strata; max-min <= 1 |
