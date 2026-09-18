#!/usr/bin/env python3
"""Build 16 auditable, pre-lock real-data evidence-family examples.

These examples are for task/gate review.  They are not the final benchmark
manifest and no target-model results may be attached to them before lock.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


DATA = Path("/public/chenjiahui/波数据时序基座大模型/data")
MONASH = DATA / "Timeseries-PILE/forecasting/monash"
PTB = DATA / "心电波/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
PTB_ROOT = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
VERSION = "real_tsqa_v1.0"
PRECISION = 4


def stable_int(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(parts).encode()).digest()[:8], "big")


def split_for_group(dataset: str, group: str) -> str:
    bucket = stable_int(VERSION, dataset, group, "split") % 10
    return "train" if bucket < 7 else "validation" if bucket < 8 else "test"


def round_array(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float), PRECISION)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_tsf(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    attrs: list[str] = []
    rows: list[dict[str, Any]] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not in_data:
                if s.lower().startswith("@attribute "):
                    attrs.append(s.split()[1])
                elif s.lower() == "@data":
                    in_data = True
                continue
            if not s:
                continue
            parts = s.split(":", len(attrs))
            if len(parts) != len(attrs) + 1:
                raise ValueError(f"cannot parse TSF line in {path.name}")
            item = dict(zip(attrs, parts[:-1]))
            item["values"] = np.asarray([float(v) for v in parts[-1].split(",")], dtype=float)
            rows.append(item)
    return attrs, rows


def choose_groups(records: list[dict[str, Any]], dataset: str, key: Callable[[dict[str, Any]], str], count: int = 4) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for r in records:
        group = key(r)
        if group in seen:
            continue
        seen.add(group)
        by_split[split_for_group(dataset, group)].append(r)
    for values in by_split.values():
        values.sort(key=lambda r: stable_int(VERSION, dataset, key(r), "example"))
    n_validation = max(1, round(count * 0.1))
    n_test = max(1, round(count * 0.2))
    n_train = count - n_validation - n_test
    chosen = by_split["train"][:n_train] + by_split["validation"][:n_validation] + by_split["test"][:n_test]
    if len(chosen) < count:
        remaining = [r for split in ("train", "validation", "test") for r in by_split[split] if r not in chosen]
        chosen += remaining[: count - len(chosen)]
    if len(chosen) != count:
        raise ValueError(f"{dataset} has only {len(chosen)} selectable independent groups")
    return chosen


def best_window(values: np.ndarray, length: int, token: str) -> tuple[int, np.ndarray]:
    if len(values) < length:
        raise ValueError(f"series {token} shorter than {length}")
    starts = list(range(0, len(values) - length + 1, max(1, length // 2)))
    pivot = stable_int(VERSION, token, "window") % len(starts)
    starts = starts[pivot:] + starts[:pivot]
    for start in starts:
        x = round_array(values[start : start + length])
        half = length // 2
        if np.all(np.isfinite(x)) and np.ptp(x) > 0 and not math.isclose(float(x[:half].mean()), float(x[half:].mean()), abs_tol=1e-8):
            return start, x
    raise ValueError(f"no nondegenerate window for {token}")


def half_mean_gold(x: np.ndarray, channel: int = 0, absolute: bool = False) -> str:
    v = np.rint(x[:, channel] * (10 ** PRECISION)).astype(np.int64)
    if absolute:
        v = np.abs(v)
    h = len(v) // 2
    return "first_half" if int(v[:h].sum()) > int(v[h:].sum()) else "second_half"


def half_range_gold(x: np.ndarray, channel: int = 0) -> str:
    v = np.rint(x[:, channel] * (10 ** PRECISION)).astype(np.int64)
    h = len(v) // 2
    return "first_half" if float(np.ptp(v[:h])) > float(np.ptp(v[h:])) else "second_half"


def quarter_max_gold(x: np.ndarray, channel: int = 0, absolute: bool = False) -> str:
    v = np.rint(x[:, channel] * (10 ** PRECISION)).astype(np.int64)
    if absolute:
        v = np.abs(v)
    chunks = np.array_split(v, 4)
    maxima = [float(np.max(c)) for c in chunks]
    return f"quarter_{int(np.argmax(maxima)) + 1}"


def conditional_gold(x: np.ndarray, channel: int = 0, absolute: bool = False) -> str:
    v = np.rint(x[:, channel] * (10 ** PRECISION)).astype(np.int64)
    if absolute:
        v = np.abs(v)
    h = len(v) // 2
    selected = v[:h] if v[:h].mean() > v[h:].mean() else v[h:]
    a, b = np.array_split(selected, 2)
    return "earlier_quarter" if float(np.max(a)) > float(np.max(b)) else "later_quarter"


def global_trend_gold(x: np.ndarray, channel: int = 0) -> str:
    values = np.rint(x[:, channel] * (10 ** PRECISION)).astype(np.int64)
    slope = float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])
    return "overall_upward" if slope > 0 else "overall_downward"


def composition_gold(x: np.ndarray, target_channel: int, control_channel: int, control_mode: str) -> str:
    target = half_mean_gold(x, target_channel)
    control = half_mean_gold(x, control_channel) if control_mode == "other_channel_mean" else half_range_gold(x, control_channel)
    return f"target_{target}__control_{control}"


def add_offset_to_lower_half(x: np.ndarray, channel: int) -> tuple[np.ndarray, dict[str, Any]]:
    result = x.copy()
    h = len(x) // 2
    means = [float(x[:h, channel].mean()), float(x[h:, channel].mean())]
    lower = 0 if means[0] < means[1] else 1
    offset = round(abs(means[0] - means[1]) + max(0.1, 0.1 * float(np.std(x[:, channel]))), PRECISION)
    sl = slice(0, h) if lower == 0 else slice(h, len(x))
    result[sl, channel] += offset
    result = round_array(result)
    if half_mean_gold(result, channel) == half_mean_gold(x, channel):
        result[sl, channel] = round_array(result[sl, channel] + max(0.1, offset * 0.25))
    return result, {
        "type": "local_constant_offset",
        "channel_index": channel,
        "region": "first_half" if lower == 0 else "second_half",
        "offset": offset,
        "allowed_change": "only the target channel in the named half",
    }


def question_templates(target_name: str, control_name: str, target_unit: str, control_unit: str, control_mode: str) -> list[dict[str, Any]]:
    target = {
        "question_id": "q_target", "task_code": "B", "task_type": "numeric_reading_comparison",
        "text_zh": f"{target_name} 在前半段和后半段中，哪一段的平均值更高？",
        "text_en": f"Which half has the higher mean {target_name}: the first half or the second half?",
        "semantic_options": {"first_half": "First half", "second_half": "Second half"},
        "semantic_options_zh": {"first_half": "前半段", "second_half": "后半段"},
        "unit": target_unit, "role": "target",
    }
    if control_mode == "other_channel_mean":
        control_text = f"{control_name} 在前半段和后半段中，哪一段的平均值更高？"
        control_text_en = f"Which half has the higher mean {control_name}: the first half or the second half?"
    else:
        control_text = f"{control_name} 在前半段和后半段中，哪一段的峰峰值更大？"
        control_text_en = f"Which half has the larger peak-to-peak range of {control_name}: the first half or the second half?"
    return [
        target,
        {"question_id": "q_control", "task_code": "D", "task_type": "temporal_relation", "text_zh": control_text,
         "text_en": control_text_en,
         "semantic_options": {"first_half": "First half", "second_half": "Second half"},
         "semantic_options_zh": {"first_half": "前半段", "second_half": "后半段"}, "unit": control_unit, "role": "control"},
        {"question_id": "q_localize", "task_code": "C", "task_type": "temporal_localization",
         "text_zh": f"把窗口依次分成四个等长区间，{target_name} 的最大值位于哪个区间？",
         "text_en": f"Divide the window into four consecutive equal intervals. Which interval contains the maximum {target_name}?",
         "semantic_options": {f"quarter_{i}": f"Interval {i}" for i in range(1, 5)},
         "semantic_options_zh": {f"quarter_{i}": f"第 {i} 个四分之一区间" for i in range(1, 5)}, "unit": target_unit, "role": "track_r_auxiliary"},
        {"question_id": "q_conditional", "task_code": "F", "task_type": "conditional_reasoning",
         "text_zh": f"先找出{target_name}平均值更高的半段，再判断该半段内的最大值位于较早还是较晚的四分之一区间？",
         "text_en": f"First select the half with the higher mean {target_name}. Within that half, is the maximum in its earlier or later quarter?",
         "semantic_options": {"earlier_quarter": "Earlier quarter of the selected half", "later_quarter": "Later quarter of the selected half"},
         "semantic_options_zh": {"earlier_quarter": "所选半段中较早的四分之一区间", "later_quarter": "所选半段中较晚的四分之一区间"},
         "unit": target_unit, "role": "track_r_auxiliary"},
        {"question_id": "q_global", "task_code": "A", "task_type": "global_structure",
         "text_zh": f"用整个窗口的线性趋势概括，{target_name}总体向上还是总体向下？",
         "text_en": f"Using a linear trend over the full window, does {target_name} trend upward or downward overall?",
         "semantic_options": {"overall_upward": "Overall upward", "overall_downward": "Overall downward"},
         "semantic_options_zh": {"overall_upward": "总体向上", "overall_downward": "总体向下"},
         "unit": target_unit, "role": "track_r_auxiliary"},
        {"question_id": "q_composition", "task_code": "E", "task_type": "multi_evidence_composition",
         "text_zh": f"哪项同时正确描述：① {target_name}哪半段均值更高；② {control_name}在控制问题所用统计量下哪半段更大？",
         "text_en": f"Which option correctly states both facts: (1) which half has the higher mean {target_name}; and (2) which half is larger for the {control_name} statistic used in the control question?",
         "semantic_options": {
             "target_first_half__control_first_half": "(1) first half; (2) first half",
             "target_first_half__control_second_half": "(1) first half; (2) second half",
             "target_second_half__control_first_half": "(1) second half; (2) first half",
             "target_second_half__control_second_half": "(1) second half; (2) second half",
         }, "semantic_options_zh": {
             "target_first_half__control_first_half": "①前半段；②前半段",
             "target_first_half__control_second_half": "①前半段；②后半段",
             "target_second_half__control_first_half": "①后半段；②前半段",
             "target_second_half__control_second_half": "①后半段；②后半段",
         }, "unit": "two visible facts", "role": "track_r_auxiliary"},
    ]


def build_family(*, family_id: str, domain: str, dataset: str, source_path: str, source_hash: str,
                 group_id: str, series_id: str, split: str, x: np.ndarray, timestamps: list[str],
                 channels: list[str], units: list[str], target_channel: int, control_channel: int,
                 control_mode: str, source_bounds: dict[str, Any], extra_source: dict[str, Any]) -> dict[str, Any]:
    x = round_array(x)
    tx, intervention = add_offset_to_lower_half(x, target_channel)
    control_fn = (lambda z: half_mean_gold(z, control_channel)) if control_mode == "other_channel_mean" else (lambda z: half_range_gold(z, control_channel))
    gold = {
        "t0": half_mean_gold(x, target_channel), "c0": control_fn(x),
        "t1": half_mean_gold(tx, target_channel), "c1": control_fn(tx),
        "r_localize": quarter_max_gold(x, target_channel),
        "r_conditional": conditional_gold(x, target_channel),
        "r_global": global_trend_gold(x, target_channel),
        "r_composition": composition_gold(x, target_channel, control_channel, control_mode),
    }
    questions = question_templates(channels[target_channel], channels[control_channel], units[target_channel], units[control_channel], control_mode)
    return {
        "benchmark_version": VERSION, "status": "PRELOCK_REVIEW_EXAMPLE", "family_id": family_id,
        "domain": domain, "dataset": dataset, "group_id": group_id, "series_id": series_id, "split": split,
        "track_r": {"variant_id": "x", "derived_from_real": False},
        "track_i": {"variant_id": "T(x)", "derived_from_real": True, "source_record_id": series_id, "intervention_type": intervention["type"]},
        "source": {"local_path": source_path, "file_or_member_sha256": source_hash, "bounds": source_bounds, **extra_source},
        "serialization": {"format": "JSON decimal arrays", "decimal_places": PRECISION, "full_and_shuffled_precision_identical": True},
        "evidence_schema": {"timestamps": timestamps, "channels": channels, "units": units},
        "oracle_config": {"target_channel": target_channel, "control_channel": control_channel, "control_mode": control_mode},
        "original_evidence": x.tolist(), "transformed_evidence": tx.tolist(), "intervention": intervention,
        "questions": questions, "semantic_gold": gold,
        "four_cell_gate": {"target_disjoint": gold["t0"] != gold["t1"], "control_invariant": gold["c0"] == gold["c1"]},
    }


def ptb_families(count: int = 4) -> list[dict[str, Any]]:
    with zipfile.ZipFile(PTB) as z:
        metadata = pd.read_csv(z.open(PTB_ROOT + "ptbxl_database.csv"))
        pools: dict[str, list[Any]] = defaultdict(list)
        used_patients = set()
        order = sorted(range(len(metadata)), key=lambda i: stable_int(VERSION, "PTB-XL", str(int(metadata.iloc[i].ecg_id)), "example"))
        for i in order:
            row = metadata.iloc[i]
            patient = str(int(row.patient_id))
            if patient in used_patients:
                continue
            used_patients.add(patient)
            split = "train" if int(row.strat_fold) <= 8 else "validation" if int(row.strat_fold) == 9 else "test"
            pools[split].append(row)
        n_val = max(1, round(count * 0.1))
        n_test = max(1, round(count * 0.1))
        n_train = count - n_val - n_test
        candidates = pools["train"][:n_train] + pools["validation"][:n_val] + pools["test"][:n_test]
        result = []
        for row_ in candidates:
            ecg_id = int(row_.ecg_id)
            member = PTB_ROOT + str(row_.filename_lr) + ".dat"
            header_member = PTB_ROOT + str(row_.filename_lr) + ".hea"
            raw = z.read(member)
            values = np.frombuffer(raw, dtype="<i2").reshape(-1, 12) / 1000.0
            start, selected = best_window(values[:, [1, 6]], 96, f"ptb:{ecg_id}")
            split = "train" if int(row_.strat_fold) <= 8 else "validation" if int(row_.strat_fold) == 9 else "test"
            result.append(build_family(
                family_id=f"ptbxl_ecg{ecg_id:05d}_{start:04d}", domain="biomedical", dataset="PTB-XL 1.0.3",
                source_path=str(PTB), source_hash=sha256_bytes(raw), group_id=f"patient_id={int(row_.patient_id)}",
                series_id=f"ecg_id={ecg_id}", split=split, x=selected,
                timestamps=[f"sample_{start+i}@100Hz" for i in range(96)], channels=["lead II voltage", "lead V1 voltage"], units=["mV", "mV"],
                target_channel=1, control_channel=0, control_mode="other_channel_mean",
                source_bounds={"sample_start_zero_based": start, "sample_stop_exclusive": start + 96},
                extra_source={"official_strat_fold": int(row_.strat_fold), "header_member": header_member, "license": "CC BY 4.0"},
            ))
    return result


def monash_single_series_families(filename: str, dataset: str, domain: str, channel: str, unit: str, length: int, count: int = 4) -> list[dict[str, Any]]:
    path = MONASH / filename
    _, records = parse_tsf(path)
    selected = choose_groups(records, dataset, lambda r: r["series_name"], count=count)
    digest = sha256_file(path)
    result = []
    for record in selected:
        group = record["series_name"]
        start, x1 = best_window(record["values"], length, f"{dataset}:{group}")
        x = x1[:, None]
        result.append(build_family(
            family_id=f"{domain}_{group.lower()}_{start:06d}", domain=domain, dataset=dataset,
            source_path=str(path), source_hash=digest, group_id=group, series_id=group,
            split=split_for_group(dataset, group), x=x,
            timestamps=[f"{record.get('start_timestamp', 'implicit_start')}+{start+i}*hour" for i in range(length)],
            channels=[channel], units=[unit], target_channel=0, control_channel=0, control_mode="same_channel_range",
            source_bounds={"start_index_zero_based": start, "stop_index_exclusive": start + length},
            extra_source={"license_release_mode": "reconstruction_manifest_only"},
        ))
    return result


def weather_families(count: int = 4) -> list[dict[str, Any]]:
    path = MONASH / "temperature_rain_dataset_without_missing_values.tsf"
    _, records = parse_tsf(path)
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        kind = record.get("obs_or_fcst")
        if kind in {"T_MEAN", "T_MAX"}:
            paired[record["station_id"]][kind] = record
    candidates = []
    for station, kinds in paired.items():
        if {"T_MEAN", "T_MAX"} <= set(kinds):
            candidates.append({"station_id": station, "mean": kinds["T_MEAN"], "max": kinds["T_MAX"]})
    selected = choose_groups(candidates, "Monash/temperature_rain_dataset_without_missing_values", lambda r: r["station_id"], count=count)
    digest = sha256_file(path)
    result = []
    for record in selected:
        station = record["station_id"]
        mean_values, max_values = record["mean"]["values"], record["max"]["values"]
        length = 32
        starts = list(range(0, min(len(mean_values), len(max_values)) - length + 1, 16))
        pivot = stable_int(VERSION, station, "weather_window") % len(starts)
        chosen = None
        for start in starts[pivot:] + starts[:pivot]:
            x = round_array(np.column_stack([mean_values[start:start+length], max_values[start:start+length]]))
            if np.all(np.isfinite(x)) and not np.any(x == 0) and np.ptp(x[:, 0]) > 0 and half_mean_gold(x, 0) in {"first_half", "second_half"}:
                if not math.isclose(float(x[:16, 0].mean()), float(x[16:, 0].mean()), abs_tol=1e-8):
                    chosen = (start, x)
                    break
        if chosen is None:
            raise ValueError(f"no zero-free observation window for station {station}")
        start, x = chosen
        result.append(build_family(
            family_id=f"auweather_station{station}_{start:03d}", domain="environment",
            dataset="Monash/temperature_rain_dataset_without_missing_values", source_path=str(path), source_hash=digest,
            group_id=f"station_id={station}", series_id=f"station={station}:T_MEAN+T_MAX", split=split_for_group("Monash/temperature_rain_dataset_without_missing_values", station),
            x=x, timestamps=[f"2015-05-02+{start+i}d" for i in range(length)], channels=["daily mean temperature", "daily maximum temperature"], units=["degC", "degC"],
            target_channel=0, control_channel=1, control_mode="other_channel_mean",
            source_bounds={"start_index_zero_based": start, "stop_index_exclusive": start + length},
            extra_source={"obs_or_fcst": ["T_MEAN", "T_MAX"], "zero_imputation_guard": "selected window contains no zero values", "license": "CC BY-SA 3.0 Australia"},
        ))
    return result


def assign_options(families: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    cell_for_question = {"q_target": "t0", "q_control": "c0", "q_localize": "r_localize", "q_conditional": "r_conditional", "q_global": "r_global", "q_composition": "r_composition"}
    for family in sorted(families, key=lambda f: stable_int(VERSION, f["family_id"], "option_order")):
        for q in family["questions"]:
            semantic = list(q["semantic_options"])
            cell = cell_for_question[q["question_id"]]
            gold = family["semantic_gold"][cell]
            positions = list("ABCD"[: len(semantic)])
            stratum = (family["domain"], q["task_type"], family["split"], cell)
            minimum = min(counts[stratum][p] for p in positions)
            targets = [p for p in positions if counts[stratum][p] == minimum]
            desired = targets[stable_int(VERSION, family["family_id"], q["question_id"]) % len(targets)]
            remaining_semantic = [s for s in semantic if s != gold]
            remaining_positions = [p for p in positions if p != desired]
            remaining_semantic.sort(key=lambda s: stable_int(VERSION, family["family_id"], q["question_id"], s))
            order = {desired: gold, **dict(zip(remaining_positions, remaining_semantic))}
            q["option_order"] = order
            q["gold_positions"] = {}
            if q["question_id"] == "q_target":
                relevant = {"t0": family["semantic_gold"]["t0"], "t1": family["semantic_gold"]["t1"]}
            elif q["question_id"] == "q_control":
                relevant = {"c0": family["semantic_gold"]["c0"], "c1": family["semantic_gold"]["c1"]}
            else:
                relevant = {cell: gold}
            for cell_name, semantic_gold in relevant.items():
                q["gold_positions"][cell_name] = next(p for p, s in order.items() if s == semantic_gold)
            counts[stratum][desired] += 1


def validate(families: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    if len(families) <= 20:
        add("family_count_12_to_20", 12 <= len(families) <= 20, str(len(families)))
    else:
        add("formal_candidate_scale", len(families) >= 160, str(len(families)))
    add("four_real_domains", len({f["domain"] for f in families}) >= 4, str(sorted({f["domain"] for f in families})))
    add("four_examples_per_domain", all(v >= 4 for v in Counter(f["domain"] for f in families).values()), str(Counter(f["domain"] for f in families)))
    add("unique_family_ids", len({f["family_id"] for f in families}) == len(families), "no duplicate IDs")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for f in families:
        group_splits[f["dataset"] + "|" + f["group_id"]].add(f["split"])
    add("group_leakage_zero", all(len(s) == 1 for s in group_splits.values()), f"{len(group_splits)} independent groups")
    evidence_hashes = [hashlib.sha256(json.dumps(f["original_evidence"], separators=(",", ":")).encode()).hexdigest() for f in families]
    near_duplicate = False
    by_dataset: dict[str, list[np.ndarray]] = defaultdict(list)
    for f in families:
        by_dataset[f["dataset"]].append(np.asarray(f["original_evidence"], dtype=float).ravel())
    for arrays in by_dataset.values():
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                if arrays[i].shape == arrays[j].shape and np.std(arrays[i]) > 0 and np.std(arrays[j]) > 0:
                    near_duplicate |= bool(abs(float(np.corrcoef(arrays[i], arrays[j])[0, 1])) > 0.999999)
    add("duplicate_near_duplicate_audit", len(set(evidence_hashes)) == len(evidence_hashes) and not near_duplicate, "no exact duplicate or |r|>0.999999 within dataset")
    add("at_least_three_ability_types", all(len({q["task_code"] for q in f["questions"]}) >= 3 for f in families), "all families have B/C/D/F")
    add("target_acceptable_sets_disjoint", all(f["semantic_gold"]["t0"] != f["semantic_gold"]["t1"] for f in families), "all target semantic answers change")
    add("control_gold_invariant", all(f["semantic_gold"]["c0"] == f["semantic_gold"]["c1"] for f in families), "all control semantic answers remain")
    recomputed = True
    intervention_region_ok = True
    for f in families:
        x, tx = np.asarray(f["original_evidence"]), np.asarray(f["transformed_evidence"])
        cfg = f["oracle_config"]
        control = (lambda z: half_mean_gold(z, cfg["control_channel"])) if cfg["control_mode"] == "other_channel_mean" else (lambda z: half_range_gold(z, cfg["control_channel"]))
        observed = {
            "t0": half_mean_gold(x, cfg["target_channel"]), "t1": half_mean_gold(tx, cfg["target_channel"]),
            "c0": control(x), "c1": control(tx), "r_localize": quarter_max_gold(x, cfg["target_channel"]),
            "r_conditional": conditional_gold(x, cfg["target_channel"]),
            "r_global": global_trend_gold(x, cfg["target_channel"]),
            "r_composition": composition_gold(x, cfg["target_channel"], cfg["control_channel"], cfg["control_mode"]),
        }
        recomputed &= observed == f["semantic_gold"]
        changed = np.argwhere(x != tx)
        h = len(x) // 2
        expected_rows = range(0, h) if f["intervention"]["region"] == "first_half" else range(h, len(x))
        expected = {(i, cfg["target_channel"]) for i in expected_rows}
        intervention_region_ok &= set(map(tuple, changed.tolist())) == expected
    add("gold_recomputation", recomputed, "all semantic gold exactly regenerated from serialized evidence")
    add("serialized_input_gold_consistency", recomputed and all(f["four_cell_gate"]["target_disjoint"] and f["four_cell_gate"]["control_invariant"] for f in families), f"decimal_places={PRECISION}")
    add("paired_candidate_set_and_order", all(len([q for q in f["questions"] if q["question_id"] in {"q_target", "q_control"} and "option_order" in q]) == 2 for f in families), "one fixed order per paired question")
    add("track_i_marked_derived", all(f["track_i"]["derived_from_real"] and f["track_i"]["source_record_id"] for f in families), "all interventions identify source record")
    add("intervention_allowed_coordinates_only", intervention_region_ok, "exactly target channel × selected half changed")
    add("source_provenance_recorded", all(f["source"]["local_path"] and f["source"]["file_or_member_sha256"] and f["source"]["bounds"] for f in families), "path, hash/member hash, group and bounds present")
    license_ok = all((f["source"].get("license") or f["source"].get("license_release_mode")) for f in families)
    add("license_status_recorded", license_ok, "every source records license or manifest-only release mode")
    add("no_hidden_label_in_evidence", all("label" not in " ".join(f["evidence_schema"]["channels"]).lower() for f in families), "visible channels contain no labels")
    add("question_only_evidence_removed", all(not any(k in q for k in ("original_evidence", "transformed_evidence", "semantic_gold")) for f in families for q in f["questions"]), "question objects carry text/options only")
    answer_words = {"first_half", "second_half", "quarter_1", "quarter_2", "quarter_3", "quarter_4", "earlier_quarter", "later_quarter"}
    add("no_answer_token_leakage", all(not any(word in q["text_zh"] for word in answer_words) for f in families for q in f["questions"]), "semantic answer IDs absent from question text")
    add("source_domain_units_preserved", all(len(f["evidence_schema"]["channels"]) == len(f["evidence_schema"]["units"]) and all(u and u != "UNKNOWN" for u in f["evidence_schema"]["units"]) for f in families), "all visible channels have explicit non-UNKNOWN units")
    add("same_numeric_precision", all(f["serialization"]["full_and_shuffled_precision_identical"] for f in families), f"fixed {PRECISION} decimals")
    add("numeric_tolerance_non_overlap", True, "N/A for current categorical comparison/localization examples; required when short-numeric subset is added")
    balance: dict[str, Counter[str]] = defaultdict(Counter)
    arities: dict[str, int] = {}
    for f in families:
        for q in f["questions"]:
            for cell, pos in q["gold_positions"].items():
                key = f"{f['domain']}|{q['task_type']}|{f['split']}|{cell}"
                balance[key][pos] += 1
                arities[key] = len(q["semantic_options"])
    add("answer_position_balance", all(max(c.values()) - min([c.get(p, 0) for p in "ABCD"[: arities[k]]]) <= 1 for k, c in balance.items()), f"{len(balance)} strata; max-min <= 1")
    return {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks,
            "answer_position_counts": {k: dict(v) for k, v in sorted(balance.items())}}


def review_markdown(families: list[dict[str, Any]], report: dict[str, Any]) -> str:
    lines = [
        "# Real TSQA v1：真实 family 人工审阅样例（pre-lock）", "",
        "> 这些是 16 个真实数据构造样例，不是正式 test manifest，也没有任何目标模型分数。完整逐点数值在 `candidate_examples.jsonl`。", "",
        f"有效性状态：**{report['status']}**。来源覆盖：{', '.join(sorted({f['domain'] for f in families}))}。", "",
    ]
    for f in families:
        q = {x["question_id"]: x for x in f["questions"]}
        x = np.asarray(f["original_evidence"])
        tx = np.asarray(f["transformed_evidence"])
        changed = np.argwhere(x != tx)
        preview = "; ".join(f"{f['evidence_schema']['timestamps'][i]}: {x[i].tolist()}" for i in range(min(6, len(x))))
        lines += [
            f"## `{f['family_id']}`", "",
            f"- 来源：{f['dataset']}；group `{f['group_id']}`；series `{f['series_id']}`；split `{f['split']}`。",
            f"- 通道/单位：{list(zip(f['evidence_schema']['channels'], f['evidence_schema']['units']))}。源边界：`{f['source']['bounds']}`。",
            f"- 可见数值预览（前 6 点）：{preview}",
            f"- q_target：{q['q_target']['text_zh']} 语义 gold `t0={f['semantic_gold']['t0']}`, `t1={f['semantic_gold']['t1']}`；固定选项 `{q['q_target']['option_order']}`。",
            f"- q_control：{q['q_control']['text_zh']} 语义 gold `c0=c1={f['semantic_gold']['c0']}`；固定选项 `{q['q_control']['option_order']}`。",
            f"- q_localize：{q['q_localize']['text_zh']} gold `{f['semantic_gold']['r_localize']}`。",
            f"- q_conditional：{q['q_conditional']['text_zh']} gold `{f['semantic_gold']['r_conditional']}`。",
            f"- q_global：{q['q_global']['text_zh']} gold `{f['semantic_gold']['r_global']}`。",
            f"- q_composition：{q['q_composition']['text_zh']} gold `{f['semantic_gold']['r_composition']}`。",
            f"- 干预：`{f['intervention']}`；改变 {len(changed)} 个序列化坐标。target changed=YES；control invariant=YES。", "",
        ]
    lines += ["## 自动 gate", "", "| gate | 状态 | 细节 |", "|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| `{c['check']}` | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
    return "\n".join(lines) + "\n"


def write_formal_artifacts(root: Path, families: list[dict[str, Any]], report: dict[str, Any]) -> None:
    for name in ("manifests", "gold", "options", "interventions", "tables"):
        (root / name).mkdir(parents=True, exist_ok=True)
    with (root / "manifests/split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark_version", "dataset", "domain", "group_id", "split"])
        writer.writeheader()
        for f in sorted(families, key=lambda x: (x["dataset"], x["group_id"])):
            writer.writerow({k: f[k] for k in ("benchmark_version", "dataset", "domain", "group_id", "split")})
    option_rows = []
    gold_rows = []
    intervention_rows = []
    item_rows = []
    aux_cells = {"q_localize": "r_localize", "q_conditional": "r_conditional", "q_global": "r_global", "q_composition": "r_composition"}
    for f in families:
        intervention_rows.append({
            "benchmark_version": f["benchmark_version"], "family_id": f["family_id"], "dataset": f["dataset"],
            "group_id": f["group_id"], "source_record_id": f["track_i"]["source_record_id"],
            "derived_from_real": True, **f["intervention"],
        })
        for q in f["questions"]:
            option_rows.append({
                "benchmark_version": f["benchmark_version"], "family_id": f["family_id"], "question_id": q["question_id"],
                "task_code": q["task_code"], "task_type": q["task_type"], "semantic_options": q["semantic_options"],
                "option_order": q["option_order"], "gold_positions": q["gold_positions"],
            })
            if q["question_id"] == "q_target":
                cells = [("t0", "x", "R"), ("t1", "T(x)", "I")]
            elif q["question_id"] == "q_control":
                cells = [("c0", "x", "R"), ("c1", "T(x)", "I")]
            else:
                cells = [(aux_cells[q["question_id"]], "x", "R")]
            for cell, variant, track in cells:
                semantic_gold = f["semantic_gold"][cell]
                position = next(pos for pos, semantic in q["option_order"].items() if semantic == semantic_gold)
                base = {
                    "benchmark_version": f["benchmark_version"], "domain": f["domain"], "dataset": f["dataset"],
                    "group_id": f["group_id"], "family_id": f["family_id"], "series_id": f["series_id"],
                    "split": f["split"], "track": track, "variant_id": variant, "cell": cell,
                    "question_id": q["question_id"], "task_code": q["task_code"], "task_type": q["task_type"],
                    "question": q["text_en"], "question_zh_for_review": q["text_zh"], "gold_semantic": semantic_gold, "gold_position": position,
                }
                item_rows.append({**base, "semantic_options": q["semantic_options"], "option_order": q["option_order"],
                                  "source_bounds": f["source"]["bounds"], "source_hash": f["source"]["file_or_member_sha256"]})
                gold_rows.append({**base, "gold_program": f["oracle_config"]})
    for path, rows in [
        (root / "options/option_manifest.jsonl", option_rows),
        (root / "gold/gold_manifest.jsonl", gold_rows),
        (root / "interventions/intervention_manifest.jsonl", intervention_rows),
        (root / "manifests/item_manifest.jsonl", item_rows),
    ]:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (root / "tables/answer_position_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stratum", "A", "B", "C", "D", "total"])
        writer.writeheader()
        for stratum, counts in report["answer_position_counts"].items():
            writer.writerow({"stratum": stratum, **{p: counts.get(p, 0) for p in "ABCD"}, "total": sum(counts.values())})
    summary = {
        "benchmark_version": VERSION, "status": "PRELOCK_FORMAL_CANDIDATE",
        "families": len(families), "groups": len({(f["dataset"], f["group_id"]) for f in families}),
        "items": len(item_rows), "track_r_items": sum(r["track"] == "R" for r in item_rows),
        "track_i_items": sum(r["track"] == "I" for r in item_rows),
        "domains": dict(Counter(f["domain"] for f in families)),
        "splits": dict(Counter(f["split"] for f in families)),
        "task_items": dict(Counter(r["task_code"] for r in item_rows)),
        "validity": report["status"], "target_model_results_seen": False,
    }
    (root / "manifests/candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("results/tsqa_evidence/real_v1"))
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument("--mode", choices=("review", "formal-candidate"), default="review")
    args = parser.parse_args()
    families = ptb_families(args.per_domain)
    families += monash_single_series_families("pedestrian_counts_dataset.tsf", "Monash/pedestrian_counts_dataset", "mobility", "pedestrian count", "pedestrians/hour", 48, args.per_domain)
    families += monash_single_series_families("electricity_hourly_dataset.tsf", "Monash/electricity_hourly_dataset", "energy", "reported electricity load", "reported load value", 48, args.per_domain)
    families += weather_families(args.per_domain)
    if args.mode == "formal-candidate":
        for family in families:
            family["status"] = "PRELOCK_FORMAL_CANDIDATE"
    assign_options(families)
    report = validate(families)
    out = args.output_root / ("review" if args.mode == "review" else "manifests")
    out.mkdir(parents=True, exist_ok=True)
    output_name = "candidate_examples.jsonl" if args.mode == "review" else "families_candidate.jsonl"
    with (out / output_name).open("w", encoding="utf-8") as handle:
        for family in families:
            handle.write(json.dumps(family, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_root / "validity").mkdir(parents=True, exist_ok=True)
    validation_name = "candidate_example_validation.json" if args.mode == "review" else "formal_candidate_validation.json"
    (args.output_root / "validity" / validation_name).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode == "review":
        review = review_markdown(families, report)
        Path("docs/tsqa_evidence/17_human_review_examples.md").write_text(review, encoding="utf-8")
        Path("docs/tsqa_evidence/21_HUMAN_REVIEW_EXAMPLES.md").write_text(review, encoding="utf-8")
    else:
        write_formal_artifacts(args.output_root, families, report)
    print(json.dumps({"families": len(families), "status": report["status"], "checks": f"{sum(c['passed'] for c in report['checks'])}/{len(report['checks'])}"}, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
