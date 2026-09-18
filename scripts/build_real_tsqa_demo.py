#!/usr/bin/env python3
"""Build a small, auditable TSQA evidence-family demo from local real data.

This is a design-review artifact, not a benchmark release.  It reads source
files in place and writes self-contained JSONL plus a human-readable review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_ROOT = Path("/public/chenjiahui/波数据时序基座大模型/data")
WEATHER = DATA_ROOT / "Timeseries-PILE/forecasting/autoformer/weather.csv"
TSB_ECG = (
    DATA_ROOT
    / "Timeseries-PILE/anomaly_detection/TSB-UAD-Public/ECG/MBA_ECG801_data.out"
)
PTB_ZIP = DATA_ROOT / "心电波/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
PTB_ROOT = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
PTB_HEADER = PTB_ROOT + "records100/00000/00001_lr.hea"
PTB_DATA = PTB_ROOT + "records100/00000/00001_lr.dat"
PTB_META = PTB_ROOT + "ptbxl_database.csv"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_float(value: float, digits: int = 6) -> float:
    result = round(float(value), digits)
    return 0.0 if result == -0.0 else result


def make_weather_family() -> dict[str, Any]:
    frame = pd.read_csv(WEATHER, nrows=12)
    columns = ["T (degC)", "p (mbar)", "max. wv (m/s)"]
    values = frame[columns].to_numpy(dtype=float)
    transformed = values.copy()
    transformed[6:, 0] += 0.60

    def oracle(array: np.ndarray) -> dict[str, Any]:
        temp_means = [array[:6, 0].mean(), array[6:, 0].mean()]
        wind_maxima = [array[:6, 2].max(), array[6:, 2].max()]
        return {
            "q_target": {
                "answer": "A" if temp_means[0] > temp_means[1] else "B",
                "calculation": {
                    "first_half_mean_degC": clean_float(temp_means[0]),
                    "second_half_mean_degC": clean_float(temp_means[1]),
                },
            },
            "q_control": {
                "answer": "A" if wind_maxima[0] > wind_maxima[1] else "B",
                "calculation": {
                    "first_half_max_m_s": clean_float(wind_maxima[0]),
                    "second_half_max_m_s": clean_float(wind_maxima[1]),
                },
            },
        }

    return {
        "family_id": "weather_20200101_0010_0200",
        "domain": "weather",
        "status": "REAL_ORIGINAL_PLUS_DERIVED_CONTROL",
        "source": {
            "dataset": "Autoformer weather local copy",
            "path": str(WEATHER),
            "sha256": sha256_file(WEATHER),
            "record_id": "weather.csv:rows[0:12]",
            "group_id": "weather_station_single_series",
            "split_note": "The local CSV has one continuous station series; split must be chronological with a guard gap before windowing.",
            "license_status": "PENDING_SOURCE_LEVEL_VERIFICATION",
        },
        "sampling": {"interval": "10 minutes", "window_points": 12},
        "evidence_schema": {
            "timestamps": frame["date"].tolist(),
            "channels": columns,
            "units": ["degC", "mbar", "m/s"],
        },
        "original_evidence": [[clean_float(v) for v in row] for row in values],
        "transformed_evidence": [[clean_float(v) for v in row] for row in transformed],
        "transformation": {
            "name": "temperature_second_half_offset",
            "description": "Add exactly 0.60 degC to T (degC) at points 7-12; all timestamps, pressure, and wind values are unchanged.",
            "changed_coordinates": {"channel": "T (degC)", "point_indices_1_based": [7, 8, 9, 10, 11, 12]},
        },
        "questions": {
            "q_target": {
                "text_zh": "这 12 个观测点按时间均分为前、后两段，哪一段的平均气温更高？",
                "options": {"A": "前 6 个点", "B": "后 6 个点"},
                "answer_source": "deterministic_visible_value_rule",
            },
            "q_control": {
                "text_zh": "哪一段观测到的最大风速（max. wv）更大？",
                "options": {"A": "前 6 个点", "B": "后 6 个点"},
                "answer_source": "deterministic_visible_value_rule",
            },
        },
        "gold": {"original": oracle(values), "transformed": oracle(transformed)},
    }


def make_ptb_family() -> dict[str, Any]:
    with zipfile.ZipFile(PTB_ZIP) as archive:
        raw = archive.read(PTB_DATA)
        header = archive.read(PTB_HEADER).decode("utf-8")
        metadata = pd.read_csv(archive.open(PTB_META))
    all_values = np.frombuffer(raw, dtype="<i2").reshape(-1, 12) / 1000.0
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    start, stop = 490, 590
    selected = all_values[start:stop][:, [lead_names.index("II"), lead_names.index("V1")]].copy()
    transformed = selected.copy()
    transformed[50:, 1] *= 1.15

    def oracle(array: np.ndarray) -> dict[str, Any]:
        v1_abs = [np.mean(np.abs(array[:50, 1])), np.mean(np.abs(array[50:, 1]))]
        ii_range = [np.ptp(array[:50, 0]), np.ptp(array[50:, 0])]
        return {
            "q_target": {
                "answer": "A" if v1_abs[0] > v1_abs[1] else "B",
                "calculation": {
                    "first_half_mean_abs_mV": clean_float(v1_abs[0]),
                    "second_half_mean_abs_mV": clean_float(v1_abs[1]),
                },
            },
            "q_control": {
                "answer": "A" if ii_range[0] > ii_range[1] else "B",
                "calculation": {
                    "first_half_peak_to_peak_mV": clean_float(ii_range[0]),
                    "second_half_peak_to_peak_mV": clean_float(ii_range[1]),
                },
            },
        }

    row = metadata.loc[metadata["ecg_id"] == 1].iloc[0]
    return {
        "family_id": "ptbxl_ecg00001_samples0490_0589",
        "domain": "clinical_ecg",
        "status": "REAL_ORIGINAL_PLUS_DERIVED_CONTROL",
        "source": {
            "dataset": "PTB-XL 1.0.3",
            "archive_path": str(PTB_ZIP),
            "member_path": PTB_DATA,
            "member_sha256": sha256_bytes(raw),
            "record_id": "ecg_id=1 / 00001_lr",
            "patient_group_id": f"patient_id={int(row['patient_id'])}",
            "official_strat_fold": int(row["strat_fold"]),
            "license": "CC BY 4.0 (verified from LICENSE.txt inside archive)",
            "header_excerpt": header.splitlines()[:3],
        },
        "sampling": {"frequency_hz": 100, "window_points": 100, "duration_seconds": 1.0, "source_sample_indices_zero_based": [start, stop - 1]},
        "evidence_schema": {"channels": ["II", "V1"], "units": ["mV", "mV"]},
        "original_evidence": [[clean_float(v) for v in row_] for row_ in selected],
        "transformed_evidence": [[clean_float(v) for v in row_] for row_ in transformed],
        "transformation": {
            "name": "v1_second_half_gain",
            "description": "Multiply only V1 samples 51-100 by 1.15; lead II and V1 samples 1-50 are bit-for-bit unchanged before JSON rounding.",
            "changed_coordinates": {"channel": "V1", "point_indices_1_based": list(range(51, 101))},
        },
        "questions": {
            "q_target": {
                "text_zh": "V1 导联在哪一半的平均绝对振幅更大？",
                "options": {"A": "前 50 个采样点", "B": "后 50 个采样点"},
                "answer_source": "deterministic_visible_value_rule",
            },
            "q_control": {
                "text_zh": "II 导联在哪一半的峰峰值更大？",
                "options": {"A": "前 50 个采样点", "B": "后 50 个采样点"},
                "answer_source": "deterministic_visible_value_rule",
            },
        },
        "gold": {"original": oracle(selected), "transformed": oracle(transformed)},
        "clinical_label_use": "No diagnosis is inferred or scored in this demo; questions use only visible waveform values.",
    }


def make_tsb_family() -> dict[str, Any]:
    array = np.loadtxt(TSB_ECG, delimiter=",", dtype=float)
    start, stop = 100, 260
    values = array[start:stop, 0].copy()
    labels = array[start:stop, 1].astype(int).copy()
    transformed_values = np.concatenate([values[80:], values[:80]])
    transformed_labels = np.concatenate([labels[80:], labels[:80]])

    def oracle(y: np.ndarray) -> dict[str, Any]:
        counts = [int(y[:80].sum()), int(y[80:].sum())]
        fraction = float(y.mean())
        if fraction < 0.25:
            bucket = "A"
        elif fraction <= 0.50:
            bucket = "B"
        elif fraction <= 0.75:
            bucket = "C"
        else:
            bucket = "D"
        return {
            "q_target": {
                "answer": "A" if counts[0] > counts[1] else "B",
                "calculation": {"first_half_anomaly_points": counts[0], "second_half_anomaly_points": counts[1]},
            },
            "q_control": {
                "answer": bucket,
                "calculation": {"anomaly_points": int(y.sum()), "total_points": int(len(y)), "fraction": clean_float(fraction)},
            },
        }

    return {
        "family_id": "tsbuad_mba_ecg801_points0100_0259",
        "domain": "ecg_anomaly_detection",
        "status": "REAL_ORIGINAL_PLUS_DERIVED_CONTROL",
        "source": {
            "dataset": "TSB-UAD-Public / ECG / MBA_ECG801",
            "path": str(TSB_ECG),
            "sha256": sha256_file(TSB_ECG),
            "record_id": "MBA_ECG801_data.out:rows[100:260]",
            "group_id": "MBA_ECG801",
            "split_note": "All windows and variants from this record must remain in one split.",
            "license_status": "PARTIAL: TSB-UAD code is Apache-2.0 and its paper traces ECG to MIT-BIH; MIT-BIH data are ODC-By-1.0. The exact redistribution lineage of this derived .out file still needs a manifest.",
            "anomaly_definition": "TSB-UAD describes ECG anomalies as ventricular premature contractions (PVCs); the second .out column is the benchmark's pointwise normal/abnormal annotation.",
        },
        "sampling": {
            "window_points": 160,
            "frequency_hz": None,
            "upstream_frequency_hz": 360,
            "frequency_status": "MIT-BIH is 360 Hz, but do not convert this derived file's indices to seconds until the TSB-UAD preprocessing lineage confirms that no resampling occurred.",
        },
        "evidence_schema": {"channels": ["signal"], "units": ["source_unit_unspecified"]},
        "original_evidence": [[clean_float(v)] for v in values],
        "transformed_evidence": [[clean_float(v)] for v in transformed_values],
        "oracle_annotations_not_model_input": {
            "original_anomaly_labels": labels.tolist(),
            "transformed_anomaly_labels": transformed_labels.tolist(),
            "warning": "The label column is used only to calculate gold answers and is never included in temporal evidence.",
        },
        "transformation": {
            "name": "swap_two_halves_with_annotations",
            "description": "Swap points 1-80 with points 81-160 and move their source anomaly annotations with them; values are not synthesized.",
        },
        "questions": {
            "q_target": {
                "text_zh": "按该数据集的原始异常标注，异常点主要集中在哪一半？",
                "options": {"A": "前 80 个点", "B": "后 80 个点"},
                "answer_source": "source_anomaly_annotation",
            },
            "q_control": {
                "text_zh": "按该数据集的原始异常标注，整个窗口中异常点占比属于哪一档？",
                "options": {"A": "小于 25%", "B": "25%–50%", "C": "大于 50%–75%", "D": "大于 75%"},
                "answer_source": "source_anomaly_annotation",
            },
        },
        "gold": {"original": oracle(labels), "transformed": oracle(transformed_labels)},
        "scope_warning": "This is an annotation-backed detection example. It must not be described as a clinical diagnosis task.",
    }


def validate(families: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("unique_family_ids", len({f["family_id"] for f in families}) == len(families), "family IDs must be unique")
    for family in families:
        fid = family["family_id"]
        original = np.asarray(family["original_evidence"], dtype=float)
        transformed = np.asarray(family["transformed_evidence"], dtype=float)
        check(f"{fid}:shape_stable", original.shape == transformed.shape, f"shape={original.shape}")
        check(f"{fid}:finite", np.isfinite(original).all() and np.isfinite(transformed).all(), "all serialized values finite")
        gold = family["gold"]
        check(
            f"{fid}:target_updates",
            gold["original"]["q_target"]["answer"] != gold["transformed"]["q_target"]["answer"],
            f"{gold['original']['q_target']['answer']} -> {gold['transformed']['q_target']['answer']}",
        )
        check(
            f"{fid}:control_invariant",
            gold["original"]["q_control"]["answer"] == gold["transformed"]["q_control"]["answer"],
            f"{gold['original']['q_control']['answer']} -> {gold['transformed']['q_control']['answer']}",
        )

    weather = families[0]
    w0, w1 = np.asarray(weather["original_evidence"]), np.asarray(weather["transformed_evidence"])
    check("weather:non_target_channels_unchanged", np.array_equal(w0[:, 1:], w1[:, 1:]), "pressure and max-wind columns exact")
    ptb = families[1]
    p0, p1 = np.asarray(ptb["original_evidence"]), np.asarray(ptb["transformed_evidence"])
    check("ptbxl:lead_ii_unchanged", np.array_equal(p0[:, 0], p1[:, 0]), "lead II exact after serialization")
    check("ptbxl:v1_first_half_unchanged", np.array_equal(p0[:50, 1], p1[:50, 1]), "V1 points 1-50 exact")
    tsb = families[2]
    t0, t1 = np.asarray(tsb["original_evidence"]), np.asarray(tsb["transformed_evidence"])
    check("tsbuad:half_swap_exact", np.array_equal(t1, np.concatenate([t0[80:], t0[:80]])), "all 160 values preserved and reordered")
    passed = all(item["passed"] for item in checks)
    return {"status": "PASS" if passed else "FAIL", "passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks}


def markdown(families: list[dict[str, Any]], report: dict[str, Any]) -> str:
    lines = [
        "# 真实时序 TSQA evidence-family demo（设计审阅版）",
        "",
        "> 这不是完整 benchmark，也不是模型结果。它只展示我们准备构造的最小评测单位长什么样。完整数值见 `families.jsonl`。",
        "",
        "## 建议的 benchmark 形态",
        "",
        "每个 family 固定一个真实记录窗口 `x`、两个问题 `q_target/q_control`，以及一个经过验证的受控变体 `T(x)`。四个单元的 gold 必须满足：目标问题答案更新，控制问题答案保持。原始真实表现与真实数据衍生的受控干预表现分开报告。",
        "",
        "| family | 真实来源 | 目标问题 | 控制问题 | 四格 gold |",
        "|---|---|---|---|---|",
    ]
    for f in families:
        g = f["gold"]
        lines.append(
            f"| `{f['family_id']}` | {f['source']['dataset']} | {f['questions']['q_target']['text_zh']} | {f['questions']['q_control']['text_zh']} | "
            f"x: ({g['original']['q_target']['answer']}, {g['original']['q_control']['answer']}); T(x): ({g['transformed']['q_target']['answer']}, {g['transformed']['q_control']['answer']}) |"
        )
    lines += [
        "",
        "建议正式版同时保留三种接口，但不混算：A 受控属性读出；B question-conditioned supervised QA；C native QA。三者只有在使用同一 family、同一语义目标和同一 split 时才做直接对照。",
        "",
        "## Demo 1：气象多变量窗口",
        "",
    ]
    weather = families[0]
    lines += [
        f"- 来源：`{weather['source']['path']}`，记录 `{weather['source']['record_id']}`。",
        "- 原始窗口：2020-01-01 00:10 到 02:00，12 个点，10 分钟间隔。证据含温度、气压、最大风速，单位来自 CSV 表头。",
        "- 干预：后 6 个温度值各加 0.60 °C；气压、风速和时间戳完全不变。",
        f"- 目标问题原始计算：{weather['gold']['original']['q_target']['calculation']}，答案 A；干预后：{weather['gold']['transformed']['q_target']['calculation']}，答案 B。",
        f"- 控制问题计算保持：{weather['gold']['original']['q_control']['calculation']}，答案 B。",
        "",
        "原始证据（完整 12 点）：",
        "",
        "| 时间 | 温度 °C | 气压 mbar | 最大风速 m/s |",
        "|---|---:|---:|---:|",
    ]
    for stamp, row in zip(weather["evidence_schema"]["timestamps"], weather["original_evidence"]):
        lines.append(f"| {stamp} | {row[0]:.2f} | {row[1]:.2f} | {row[2]:.2f} |")
    ptb = families[1]
    lines += [
        "",
        "## Demo 2：PTB-XL 双导联 ECG 窗口",
        "",
        f"- 来源：归档内 `{ptb['source']['member_path']}`，`{ptb['source']['record_id']}`，`{ptb['source']['patient_group_id']}`，官方 stratified fold={ptb['source']['official_strat_fold']}。",
        "- 原始窗口：100 Hz 下第 490–589 号样本，II 与 V1 两导联，共 1 秒；原文件头给出单位 mV。",
        "- 干预：仅将 V1 后 50 点乘 1.15；II 导联保持逐点相同。",
        f"- 目标问题原始计算：{ptb['gold']['original']['q_target']['calculation']}，答案 A；干预后：{ptb['gold']['transformed']['q_target']['calculation']}，答案 B。",
        f"- 控制问题计算保持：{ptb['gold']['original']['q_control']['calculation']}，答案 B。",
        "- 这条不评判病理诊断；它只测试问题能否选择正确导联、区间和统计操作。正式划分按 patient_id 分组。",
        "",
        "数值预览（完整 100 点在 JSONL）：",
        "",
        "| 窗口内点号 | II (mV) | V1 (mV) |",
        "|---:|---:|---:|",
    ]
    preview_indices = list(range(5)) + list(range(47, 53)) + list(range(95, 100))
    for idx in preview_indices:
        row = ptb["original_evidence"][idx]
        lines.append(f"| {idx + 1} | {row[0]:.3f} | {row[1]:.3f} |")
    tsb = families[2]
    lines += [
        "",
        "## Demo 3：TSB-UAD ECG 异常窗口",
        "",
        f"- 来源：`{tsb['source']['path']}`，记录组 `{tsb['source']['group_id']}`。",
        "- 原始窗口：源文件第 100–259 行的 160 个真实点；第二列原始 0/1 标注只作 gold oracle，不提供给模型。",
        "- 干预：交换前后 80 点，并让源异常标注随对应数据点一起移动；没有合成新数值。",
        f"- 目标问题原始计算：{tsb['gold']['original']['q_target']['calculation']}，答案 A；干预后：{tsb['gold']['transformed']['q_target']['calculation']}，答案 B。",
        f"- 控制问题：异常总数和占比保持 {tsb['gold']['original']['q_control']['calculation']}，答案 B。",
        "- TSB-UAD 将这里的异常定义为心室早搏（PVC），并提供逐点 0/1 标注。上游 MIT-BIH 是 360 Hz，但在核准该 `.out` 文件的预处理映射和未重采样证据前，只按点号提问，不换算成秒。",
        "- TSB-UAD 代码为 Apache-2.0；上游 MIT-BIH 数据为 ODC-By-1.0。正式发布仍需保存本地 `.out` 到上游记录/处理脚本的逐文件 lineage，而不能拿代码许可替代数据许可。",
        "- 这条只能称 PVC 异常检测，不能扩大为一般临床诊断。",
        "",
        "## 正式 benchmark 的核心指标",
        "",
        "- `Cell accuracy`：四个单元分别报告，不能只给总平均。",
        "- `Target update success`：同一 family 上目标问题在 x 与 T(x) 都答对，且答案按 gold 正确更新。",
        "- `Control invariance success`：控制问题在 x 与 T(x) 都答对，并保持 gold 指定答案。",
        "- `Four-cell joint success`：a、b、a′、b′ 四题全部正确；这是最严格的 family 级指标。",
        "- `Question-switch joint accuracy`：同一 x 上 q_target 和 q_control 同时答对，用于测量是否随问题选择不同证据。",
        "- 各指标按 family、数据域和接口分别 macro-average；置信区间按原始记录/患者等 group bootstrap，不能把同一 family 的题当独立样本。",
        "",
        "不能使用“答案有没有变化”作为成功判据；只有变化方向和最终答案都匹配 gold 才算成功。数值问题另报预注册容差内准确率，分类/选择题报 exact semantic accuracy。",
        "",
        "## 当前有效性检查",
        "",
        f"状态：**{report['status']}**（{report['passed']}/{report['total']}）",
        "",
        "| 检查 | 结果 | 细节 |",
        "|---|---|---|",
    ]
    for item in report["checks"]:
        lines.append(f"| `{item['check']}` | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |")
    lines += [
        "",
        "## 还未锁定的事项",
        "",
        "- 气象本地副本的来源级许可仍需核验；TSB-UAD ECG 已查到异常语义及上游采样率/许可，但逐文件预处理 lineage 和原始记录映射仍需核验。",
        "- 干预幅度、允许的问题模板和答案容差要在任何新模型正式测试前锁定。",
        "- 完整 benchmark 要先按患者、设备、站点或原始记录分组，再切窗口；同一记录的所有问题和变体必须落在同一 split。",
        "- 原始真实窗口与 `REAL_DERIVED_CONTROLLED` 变体必须分表报告。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/tsqa_evidence/real_demo_v0"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    families = [make_weather_family(), make_ptb_family(), make_tsb_family()]
    report = validate(families)
    with (args.output_dir / "families.jsonl").open("w", encoding="utf-8") as handle:
        for family in families:
            handle.write(json.dumps(family, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README_zh.md").write_text(markdown(families, report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "families": len(families), "validation": report["status"], "checks": f"{report['passed']}/{report['total']}"}, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
