#!/usr/bin/env python3
"""Create the source and eligibility inventories for real TSQA v1.

This program is deliberately conservative: a local file is evidence of local
availability, not evidence of provenance, license, units, or independent group
identity.  Unknown fields remain explicit and make a source ineligible instead
of being guessed from a directory name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DATA = Path("/public/chenjiahui/波数据时序基座大模型/data")
PILE = DATA / "Timeseries-PILE"
MONASH = PILE / "forecasting/monash"
AUTOFORMER = PILE / "forecasting/autoformer"
UCR = PILE / "classification/UCR"
TSB = PILE / "anomaly_detection/TSB-UAD-Public"
UNITS = DATA / "UniTS_data/dataset"

FIELDS = [
    "dataset_name", "local_path", "source_project", "upstream_url_citation",
    "domain", "real_vs_synthetic", "number_of_records", "number_of_channels",
    "sampling_rate", "timestamp_available", "units", "record_group_identifier",
    "existing_labels", "event_labels", "anomaly_labels", "original_split",
    "license", "redistribution_status", "local_preprocessing", "source_version",
    "file_hashes", "known_lineage_gaps", "inventory_status",
]

TASKS = {
    "A": "global_structure",
    "B": "numeric_reading",
    "C": "temporal_localization",
    "D": "temporal_relation",
    "E": "multi_evidence_composition",
    "F": "conditional_reasoning",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(paths: Iterable[Path], max_bytes: int = 256 * 1024 * 1024) -> str:
    paths = sorted(paths)
    parts = []
    tree = hashlib.sha256()
    for path in paths:
        size = path.stat().st_size
        digest = sha256(path) if size <= max_bytes else "PENDING_FULL_SHA256"
        item = f"{path.name}:{size}:{digest}"
        parts.append(item)
        tree.update((item + "\n").encode("utf-8"))
    if len(parts) > 20:
        pending = sum(x.endswith("PENDING_FULL_SHA256") for x in parts)
        return f"tree_sha256:{tree.hexdigest()}; files={len(parts)}; pending_large_files={pending}"
    return " | ".join(parts)


def tsf_metadata(path: Path) -> dict[str, Any]:
    attributes: list[str] = []
    frequency = "UNKNOWN"
    relation = path.stem
    records = 0
    header: list[str] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not in_data:
                header.append(stripped)
                lower = stripped.lower()
                if lower.startswith("@attribute "):
                    attributes.append(stripped.split()[1])
                elif lower.startswith("@frequency "):
                    frequency = stripped.split(maxsplit=1)[1]
                elif lower.startswith("@relation "):
                    relation = stripped.split(maxsplit=1)[1]
                elif lower == "@data":
                    in_data = True
            elif stripped:
                records += 1
    return {
        "attributes": attributes,
        "frequency": frequency,
        "relation": relation,
        "records": records,
        "header": "\n".join(header),
    }


def csv_shape(path: Path) -> tuple[int, int, list[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    channels = len(header) - (1 if header and header[0].lower() in {"date", "timestamp"} else 0)
    return rows, channels, header


def row(**kwargs: Any) -> dict[str, Any]:
    base = {field: "UNKNOWN" for field in FIELDS}
    base.update(kwargs)
    return base


MONASH_OVERRIDES: dict[str, dict[str, str]] = {
    "australian_electricity_demand_dataset": {
        "domain": "energy", "real_vs_synthetic": "REAL",
        "upstream_url_citation": "tsibbledata package v0.3.0; https://cran.r-project.org/package=tsibbledata",
        "units": "electricity demand; exact physical unit not stated in local TSF",
        "record_group_identifier": "state (5 independent geographic series)",
        "local_preprocessing": "Monash TSF conversion; no further local transform audited",
    },
    "electricity_hourly_dataset": {
        "domain": "energy", "real_vs_synthetic": "DERIVED_REAL",
        "upstream_url_citation": "UCI ElectricityLoadDiagrams20112014; https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
        "units": "source is kW; semantics of hourly aggregation require confirmation",
        "record_group_identifier": "client series_name (321 clients in derived file)",
        "local_preprocessing": "370 15-minute client series aggregated/filtered to 321 hourly series by Lai et al.; exact aggregation mapping pending",
        "license": "CC BY 4.0 (official UCI source); derived Monash TSF released through Monash archive",
        "lineage_gap": "exact 370-to-321 client filtering and 15-minute-to-hour aggregation operator/unit semantics",
    },
    "pedestrian_counts_dataset": {
        "domain": "mobility", "real_vs_synthetic": "REAL",
        "upstream_url_citation": "City of Melbourne Pedestrian Counting System; Monash record https://zenodo.org/records/4656626",
        "units": "pedestrians per hour", "record_group_identifier": "sensor series_name (T1..T66)",
        "local_preprocessing": "Monash TSF conversion; observations through 2020-04-30",
        "license": "CC BY (official City of Melbourne portal); Monash archive record retained for attribution",
        "lineage_gap": "Monash TSF maps sensors to T1..T66; retain archive version and source attribution",
    },
    "traffic_hourly_dataset": {
        "domain": "mobility", "real_vs_synthetic": "DERIVED_REAL",
        "upstream_url_citation": "Caltrans PeMS; Monash record https://zenodo.org/records/4656132",
        "units": "road occupancy rate", "record_group_identifier": "freeway sensor series_name (862 series)",
        "local_preprocessing": "hourly version used by Lai et al.; sensor mapping pending",
    },
    "weather_dataset": {
        "domain": "environment", "real_vs_synthetic": "DERIVED_REAL",
        "upstream_url_citation": "Australian Bureau of Meteorology via bomrang; Monash record https://zenodo.org/records/4654822",
        "units": "rain/mm; min/max temperature/degC; solar radiation/MJ m^-2 (source semantics; local series mapping pending)",
        "record_group_identifier": "weather station mapping absent from TSF attributes; series_name alone is insufficient",
        "local_preprocessing": "extracted with bomrang and converted to TSF",
    },
    "temperature_rain_dataset_without_missing_values": {
        "domain": "environment", "real_vs_synthetic": "DERIVED_REAL",
        "upstream_url_citation": "Australian Government weather verification data (2015-05 to 2017-04); URLs embedded in TSF header",
        "units": "T_MEAN/T_MAX/T_MIN degC; PRCP_SUM mm; forecast-variable-specific units",
        "record_group_identifier": "station_id (422 stations); series also keyed by obs_or_fcst",
        "local_preprocessing": "rounded to 5 decimals; missing values replaced by zero; source TSF provides explicit station_id",
        "license": "CC BY-SA 3.0 Australia (official data.gov.au source)",
        "lineage_gap": "zero-imputation is irreversible; candidate construction excludes every window containing zero",
    },
}


AUTOFORMER_META = {
    "ETTh1": ("energy", "DERIVED_REAL", "hourly", "7", "Electricity transformer temperature benchmark; Informer/ETDataset"),
    "ETTh2": ("energy", "DERIVED_REAL", "hourly", "7", "Electricity transformer temperature benchmark; Informer/ETDataset"),
    "ETTm1": ("energy", "DERIVED_REAL", "15 minutes", "7", "Electricity transformer temperature benchmark; Informer/ETDataset"),
    "ETTm2": ("energy", "DERIVED_REAL", "15 minutes", "7", "Electricity transformer temperature benchmark; Informer/ETDataset"),
    "electricity": ("energy", "DERIVED_REAL", "hourly", "321", "UCI ElectricityLoadDiagrams20112014 via Lai et al."),
    "exchange_rate": ("finance", "REAL", "daily", "8", "Lai et al. multivariate time-series data"),
    "national_illness": ("public_health", "DERIVED_REAL", "weekly", "7", "US CDC influenza-like illness surveillance"),
    "traffic": ("mobility", "DERIVED_REAL", "hourly", "862", "Caltrans PeMS via Lai et al."),
    "weather": ("environment", "REAL", "10 minutes", "21", "Max Planck Institute Jena weather station"),
}


def inventory_autoformer() -> list[dict[str, Any]]:
    result = []
    for path in sorted(AUTOFORMER.glob("*.csv")):
        n, channels, header = csv_shape(path)
        domain, reality, rate, expected_channels, citation = AUTOFORMER_META.get(
            path.stem, ("UNKNOWN", "UNKNOWN", "UNKNOWN", str(channels), "UNKNOWN")
        )
        result.append(row(
            dataset_name=f"Autoformer/{path.stem}", local_path=str(path),
            source_project="Informer/Autoformer benchmark copy in Time-series Pile",
            upstream_url_citation=citation, domain=domain, real_vs_synthetic=reality,
            number_of_records=n, number_of_channels=channels, sampling_rate=rate,
            timestamp_available="YES (date column)", units="UNKNOWN unless encoded in column name",
            record_group_identifier="single multivariate continuous record; no independent group key",
            existing_labels="none", event_labels="none", anomaly_labels="none",
            original_split="no split stored in CSV", license="collection README says MIT; source-level data license not established locally",
            redistribution_status="PENDING_SOURCE_LEVEL_LICENSE", local_preprocessing="benchmark CSV conversion; exact mapping pending",
            source_version="local copy; version not recorded", file_hashes=file_identity([path]),
            known_lineage_gaps=f"source version, per-source license, independent groups; columns={header}; expected_channels={expected_channels}",
            inventory_status="AUDITED_LOCAL_METADATA",
        ))
    return result


def inventory_monash() -> list[dict[str, Any]]:
    result = []
    for path in sorted(MONASH.glob("*.tsf")):
        meta = tsf_metadata(path)
        stem = path.stem
        override = MONASH_OVERRIDES.get(stem, {})
        attributes = meta["attributes"]
        group_attrs = [x for x in attributes if x not in {"start_timestamp", "series_name"}]
        result.append(row(
            dataset_name=f"Monash/{stem}", local_path=str(path), source_project="Monash Time Series Forecasting Archive",
            upstream_url_citation=override.get("upstream_url_citation", "citation/source URL may be present in TSF header; not yet source-verified"),
            domain=override.get("domain", "UNKNOWN"), real_vs_synthetic=override.get("real_vs_synthetic", "UNKNOWN"),
            number_of_records=meta["records"], number_of_channels="1 per TSF series",
            sampling_rate=meta["frequency"], timestamp_available="start timestamp attribute" if "start_timestamp" in attributes else "implicit regular index only",
            units=override.get("units", "UNKNOWN"),
            record_group_identifier=override.get("record_group_identifier", f"series_name; extra attrs={group_attrs or 'none'}; independence not verified"),
            existing_labels="series attributes only", event_labels="none", anomaly_labels="none",
            original_split="none in TSF", license=override.get("license", "Monash archive distribution metadata reports CC BY 4.0; upstream source terms still audited per dataset"),
            redistribution_status="MANIFEST_ONLY_UNTIL_SOURCE_TERMS_VERIFIED",
            local_preprocessing=override.get("local_preprocessing", "Monash TSF conversion; details in local header/upstream record"),
            source_version="local TSF; archive record version not embedded", file_hashes=file_identity([path]),
            known_lineage_gaps=override.get("lineage_gap", "; ".join(x for x in [
                "upstream record/version and source-level license not fully verified",
                "physical units absent" if override.get("units", "UNKNOWN") == "UNKNOWN" else "",
                "independent group identity not verified" if stem not in MONASH_OVERRIDES else "",
            ] if x) or "none identified in current audit"),
            inventory_status="AUDITED_LOCAL_METADATA",
        ))
    return result


def parse_ts_count(path: Path) -> int:
    in_data = False
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            s = line.strip()
            if s.lower() == "@data":
                in_data = True
            elif in_data and s and not s.startswith("#"):
                count += 1
    return count


def inventory_ucr() -> list[dict[str, Any]]:
    result = []
    for directory in sorted(p for p in UCR.iterdir() if p.is_dir()):
        files = sorted(directory.glob("*.ts"))
        counts = {p.stem.rsplit("_", 1)[-1].lower(): parse_ts_count(p) for p in files}
        result.append(row(
            dataset_name=f"UCR/{directory.name}", local_path=str(directory), source_project="UCR/UEA time-series classification archive",
            upstream_url_citation="https://www.timeseriesclassification.com/; dataset-specific citation in .ts comments when present",
            domain="UNKNOWN", real_vs_synthetic="UNKNOWN", number_of_records=sum(counts.values()),
            number_of_channels="from @univariate/@dimensions; not normalized in inventory", sampling_rate="UNKNOWN",
            timestamp_available="usually NO; per-file metadata", units="UNKNOWN",
            record_group_identifier="sample row; subject/device grouping generally unavailable",
            existing_labels="class label", event_labels="none", anomaly_labels="none",
            original_split=json.dumps(counts, sort_keys=True), license="dataset-specific; NOT established by archive membership",
            redistribution_status="EXCLUDE_UNTIL_DATASET_SPECIFIC_LICENSE", local_preprocessing="UCR/UEA .ts conversion",
            source_version="2018 archive copy; exact revision pending", file_hashes=file_identity(files),
            known_lineage_gaps="real/synthetic status, sampling, units, independent group, and license require dataset-specific audit",
            inventory_status="INVENTORIED_EXCLUDED_UNKNOWN",
        ))
    return result


def inventory_tsb() -> list[dict[str, Any]]:
    result = []
    known_real = {"Daphnet", "Dodgers", "ECG", "GHL", "Genesis", "IOPS", "MITDB", "NAB", "NASA-MSL", "NASA-SMAP", "OPPORTUNITY", "Occupancy", "SMD", "SVDB", "SensorScope", "YAHOO"}
    for directory in sorted(p for p in TSB.iterdir() if p.is_dir()):
        files = sorted(directory.glob("*.out"))
        reality = "DERIVED_REAL" if directory.name in known_real else "UNKNOWN"
        result.append(row(
            dataset_name=f"TSB-UAD/{directory.name}", local_path=str(directory), source_project="TSB-UAD-Public",
            upstream_url_citation="Paparrizos et al., VLDB 2022; https://github.com/TheDatumOrg/TSB-UAD",
            domain="anomaly_detection", real_vs_synthetic=reality, number_of_records=len(files), number_of_channels=1,
            sampling_rate="dataset/record specific; not stored in .out", timestamp_available="NO (sample index only)",
            units="not stored in .out", record_group_identifier=".out file, but mapping to upstream subject/device often incomplete",
            existing_labels="pointwise binary labels", event_labels="not separately stored", anomaly_labels="second .out column",
            original_split="none in local layout", license="TSB-UAD code Apache-2.0; this does not establish every upstream dataset license",
            redistribution_status="INTERNAL_ONLY_PENDING_UPSTREAM_LINEAGE", local_preprocessing="TSB-UAD two-column derived .out; exact transforms dataset-specific",
            source_version="local TSB-UAD-Public copy; git revision unavailable", file_hashes=file_identity(files),
            known_lineage_gaps="upstream record mapping, sampling/resampling, filtering/normalization, label conversion, and data license",
            inventory_status="LINEAGE_PARTIAL",
        ))
    return result


def inventory_units() -> list[dict[str, Any]]:
    result = []
    if not UNITS.exists():
        return result
    for directory in sorted(p for p in UNITS.iterdir() if p.is_dir()):
        files = [p for p in directory.rglob("*") if p.is_file()]
        result.append(row(
            dataset_name=f"UniTS_data/{directory.name}", local_path=str(directory), source_project="local UniTS/WaveFormer data mirror",
            upstream_url_citation="UNKNOWN", domain="UNKNOWN", real_vs_synthetic="UNKNOWN",
            number_of_records="requires format-specific parse", number_of_channels="requires format-specific parse",
            sampling_rate="UNKNOWN", timestamp_available="UNKNOWN", units="UNKNOWN",
            record_group_identifier="UNKNOWN", existing_labels="format-specific", event_labels="UNKNOWN", anomaly_labels="UNKNOWN",
            original_split="local directory may contain splits; lineage not audited", license="UNKNOWN",
            redistribution_status="EXCLUDE_UNTIL_LINEAGE_VERIFIED", local_preprocessing="UNKNOWN",
            source_version="UNKNOWN", file_hashes=f"{len(files)} files; content hash deferred (97.9 GB symlinked tree)",
            known_lineage_gaps="symlink points outside project; source, license, version, group identity and transforms not documented in current tree",
            inventory_status="INVENTORIED_EXCLUDED_UNKNOWN",
        ))
    return result


def inventory_raw() -> list[dict[str, Any]]:
    ptb = DATA / "心电波/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    mit = DATA / "心电波/mit-bih-arrhythmia-database-1.0.0.zip"
    plasticc = DATA / "光变曲线/PLAsTiCC/PLAsTiCC-2018.zip"
    quake = DATA / "地震波/merged.zip"
    gravity_meta = DATA / "引力波/Gravity-Spy/trainingset_v1d1_metadata.csv"
    gravity_h5 = DATA / "引力波/Gravity-Spy/trainingsetv1d1.h5"
    paderborn = DATA / "振动波/Paderborn"
    rows = [
        row(dataset_name="PTB-XL 1.0.3", local_path=str(ptb), source_project="PhysioNet PTB-XL",
            upstream_url_citation="https://physionet.org/content/ptb-xl/1.0.3/; Wagner et al. 2020",
            domain="biomedical_ecg", real_vs_synthetic="REAL", number_of_records=21799, number_of_channels=12,
            sampling_rate="100 Hz and 500 Hz", timestamp_available="record start metadata; samples have regular index", units="mV",
            record_group_identifier="patient_id (18,869 patients); ecg_id record", existing_labels="scp_codes and diagnostic metadata",
            event_labels="none at beat level", anomaly_labels="no generic anomaly label", original_split="strat_fold 1-10; patient-respecting",
            license="CC BY 4.0 (LICENSE.txt inside archive)", redistribution_status="RELEASE_ALLOWED_WITH_ATTRIBUTION; benchmark should still prefer reconstruction manifests",
            local_preprocessing="official 100 Hz downsampled and 500 Hz records; no project transform yet", source_version="1.0.3",
            file_hashes=f"archive:{ptb.stat().st_size}:PENDING_FULL_SHA256; official member SHA256SUMS.txt embedded",
            known_lineage_gaps="local metadata has 21,799 rows after v1.0.3 duplicate removals; official landing-page headline may round/reference 21,801",
            inventory_status="SOURCE_LICENSE_GROUP_VERIFIED"),
        row(dataset_name="MIT-BIH Arrhythmia 1.0.0", local_path=str(mit), source_project="PhysioNet MIT-BIH Arrhythmia Database",
            upstream_url_citation="https://physionet.org/content/mitdb/1.0.0/", domain="biomedical_ecg", real_vs_synthetic="REAL",
            number_of_records=48, number_of_channels=2, sampling_rate="360 Hz", timestamp_available="regular sample index", units="record-header-specific ADC calibration",
            record_group_identifier="record; 47 subjects (records 201/202 share one subject)", existing_labels="expert beat/rhythm annotations",
            event_labels="beat annotations", anomaly_labels="arrhythmia annotations, not a generic binary anomaly label", original_split="none",
            license="ODC-By 1.0 (official PhysioNet record)", redistribution_status="RELEASE_ALLOWED_WITH_ATTRIBUTION; prefer reconstruction manifest",
            local_preprocessing="official archive; no project transform", source_version="1.0.0", file_hashes=file_identity([mit]),
            known_lineage_gaps="subject grouping exception must be encoded before a split", inventory_status="SOURCE_LICENSE_VERIFIED_GROUP_NEEDS_ENCODING"),
        row(dataset_name="PLAsTiCC 2018", local_path=str(plasticc), source_project="Kaggle/LSST PLAsTiCC",
            upstream_url_citation="PLAsTiCC 2018 challenge; exact competition/data terms pending", domain="astronomy", real_vs_synthetic="SYNTHETIC",
            number_of_records="archive members include train/test observations and metadata", number_of_channels="6 passbands", sampling_rate="irregular",
            timestamp_available="YES (MJD)", units="flux units per challenge", record_group_identifier="object_id",
            existing_labels="object class", event_labels="astronomical transient class", anomaly_labels="none", original_split="challenge train/test",
            license="PENDING", redistribution_status="EXCLUDED: synthetic and terms pending", local_preprocessing="none audited", source_version="2018",
            file_hashes=f"archive:{plasticc.stat().st_size}:PENDING_FULL_SHA256", known_lineage_gaps="license/terms and local extraction not audited",
            inventory_status="EXCLUDED_SYNTHETIC"),
        row(dataset_name="LANL Earthquake Prediction merged archive", local_path=str(quake), source_project="likely Kaggle LANL Earthquake Prediction",
            upstream_url_citation="PENDING_CONFIRMATION", domain="seismic", real_vs_synthetic="UNKNOWN", number_of_records="merge.csv + merge.hdf5",
            number_of_channels="UNKNOWN", sampling_rate="UNKNOWN", timestamp_available="UNKNOWN", units="UNKNOWN",
            record_group_identifier="UNKNOWN", existing_labels="UNKNOWN", event_labels="UNKNOWN", anomaly_labels="UNKNOWN", original_split="UNKNOWN",
            license="UNKNOWN", redistribution_status="EXCLUDE", local_preprocessing="unknown merged archive", source_version="UNKNOWN",
            file_hashes=f"archive:{quake.stat().st_size}:PENDING_FULL_SHA256", known_lineage_gaps="all provenance fields; filename is insufficient evidence",
            inventory_status="INVENTORIED_EXCLUDED_UNKNOWN"),
        row(dataset_name="Gravity Spy training set v1d1", local_path=str(gravity_meta.parent), source_project="Gravity Spy",
            upstream_url_citation="Gravity Spy v1d1; exact release page/terms pending", domain="gravitational_wave_detector", real_vs_synthetic="DERIVED_REAL",
            number_of_records=7966, number_of_channels="time-frequency image views; not a direct 1-D waveform", sampling_rate="not applicable to stored spectrogram views",
            timestamp_available="event GPS metadata", units="spectrogram representation; exact scaling pending", record_group_identifier="event/sample id and detector IFO",
            existing_labels="22 glitch classes", event_labels="glitch class", anomaly_labels="none", original_split="train/validation/test in metadata",
            license="PENDING_SOURCE_VERIFICATION", redistribution_status="EXCLUDE_UNTIL_LICENSE", local_preprocessing="stored HDF5 time-frequency representations",
            source_version="v1d1", file_hashes=file_identity([gravity_meta] + ([gravity_h5] if gravity_h5.exists() else [])),
            known_lineage_gaps="not 1-D series; exact HDF5 representation and license require audit", inventory_status="INVENTORIED_NOT_INITIAL_CANDIDATE"),
        row(dataset_name="Paderborn Bearing Data Center", local_path=str(paderborn), source_project="Paderborn University Bearing Data Center",
            upstream_url_citation="https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter; Lessmeier et al. PHM 2016",
            domain="industrial_vibration", real_vs_synthetic="REAL", number_of_records=len(list(paderborn.glob("*.rar"))),
            number_of_channels="MAT files contain vibration/current/metadata; schema inspection pending", sampling_rate="64 kHz for the two main signals per official paper; supplemental channels require MAT verification",
            timestamp_available="sample index", units="PENDING", record_group_identifier="bearing code (K/KA/KB/KI) plus operating condition",
            existing_labels="bearing condition encoded by archive/record name", event_labels="none", anomaly_labels="fault type/condition, not pointwise anomaly",
            original_split="none", license="CC BY-NC 4.0 (official Paderborn DataCenter page)", redistribution_status="MANIFEST_ONLY; noncommercial academic use with attribution",
            local_preprocessing="30 per-bearing RAR archives of MATLAB files", source_version="local download; version unknown",
            file_hashes=f"30 RAR archives; full hashes deferred; total_bytes={sum(p.stat().st_size for p in paderborn.glob('*.rar'))}",
            known_lineage_gaps="local channel/unit schema and per-channel sampling confirmation; official paper reports 64 kHz for two main signals",
            inventory_status="LINEAGE_PENDING_HIGH_VALUE_CANDIDATE"),
    ]
    return rows


def make_eligibility() -> list[dict[str, str]]:
    datasets = {
        "PTB-XL 1.0.3": dict(source="PASS", license="PASS", group="PASS", sampling="PASS", units="PASS", release="PASS_MANIFEST_AND_CC_BY", native="PASS", supervised="PASS", safe="PASS", reason="Visible waveform facts only; no inferred diagnosis; patient split."),
        "Monash/pedestrian_counts_dataset": dict(source="PASS", license="PASS", group="PASS", sampling="PASS", units="PASS", release="PASS_MANIFEST_ONLY", native="PASS", supervised="PASS", safe="PASS", reason="Hourly counts and sensor IDs support deterministic visible-value QA."),
        "Monash/electricity_hourly_dataset": dict(source="PASS", license="PASS", group="PASS", sampling="PASS", units="PENDING", release="PASS_MANIFEST_ONLY", native="PASS_RESTRICTED", supervised="PASS_RESTRICTED", safe="PASS", reason="Use relative/reported-value questions until hourly aggregation unit semantics are confirmed."),
        "Monash/temperature_rain_dataset_without_missing_values": dict(source="PASS", license="PASS_CC_BY_SA_3_AU", group="PASS", sampling="PASS", units="PASS_OBSERVATIONS", release="PASS_MANIFEST_ONLY", native="PASS_RESTRICTED", supervised="PASS_RESTRICTED", safe="PASS_RESTRICTED", reason="Official source is CC BY-SA 3.0 AU; station IDs and observation units exist; zero-imputed spans are excluded and forecast series are out of scope."),
        "Monash/weather_dataset": dict(source="PASS", license="PASS", group="FAIL", sampling="PASS", units="PASS", release="PASS_MANIFEST_ONLY", native="FAIL", supervised="FAIL_CONFIRMATORY", safe="PENDING", reason="TSF omits station mapping, so independent group split is not auditable."),
        "Monash/traffic_hourly_dataset": dict(source="PASS", license="PENDING", group="PENDING", sampling="PASS", units="PASS", release="PENDING", native="PENDING", supervised="PENDING", safe="PASS", reason="Road-occupancy semantics are clear; source terms and sensor mapping remain pending."),
        "Paderborn Bearing Data Center": dict(source="PASS", license="PASS_CC_BY_NC", group="PASS", sampling="PASS", units="PENDING", release="MANIFEST_ONLY_NONCOMMERCIAL", native="PENDING", supervised="PENDING", safe="PENDING", reason="Official source confirms CC BY-NC 4.0, 64 kHz main signals, bearing groups and operating conditions; local MAT channel/unit schema still needs extraction."),
        "TSB-UAD/ECG": dict(source="PARTIAL", license="PARTIAL", group="FAIL", sampling="FAIL", units="FAIL", release="FAIL", native="FAIL", supervised="EXPLORATORY_ONLY", safe="PARTIAL", reason="Derived .out-to-upstream record and label mapping unresolved; supervision cannot repair provenance."),
    }
    result: list[dict[str, str]] = []
    for dataset, state in datasets.items():
        for code, task in TASKS.items():
            status = "PASS" if all(state[k].startswith("PASS") for k in ("source", "license", "group", "sampling")) and state["native"].startswith("PASS") else "PENDING_OR_EXCLUDED"
            reason = state["reason"]
            if dataset == "PTB-XL 1.0.3" and code in {"A", "B", "C", "D", "E", "F"}:
                reason += " Task operates on visible lead/interval statistics."
            result.append({
                "dataset_name": dataset, "task_code": code, "task_type": task,
                "source_verified": state["source"], "license_verified": state["license"],
                "group_verified": state["group"], "sampling_verified": state["sampling"],
                "units_verified": state["units"], "gold_recomputable": "PASS" if state["source"] != "PARTIAL" else "PASS_NUMERIC_ONLY",
                "intervention_safe": state["safe"], "native_QA_eligible": state["native"],
                "supervised_QA_eligible": state["supervised"], "release_eligible": state["release"],
                "eligibility": status, "reason": reason,
            })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_inventory_doc(path: Path, rows: list[dict[str, Any]]) -> None:
    statuses = Counter(str(r["inventory_status"]) for r in rows)
    realities = Counter(str(r["real_vs_synthetic"]) for r in rows)
    selected = [
        ("PTB-XL 1.0.3", "biomedical", "可进入 candidate；只问可见波形事实，按 patient_id 分组"),
        ("Monash/pedestrian_counts_dataset", "mobility", "可进入 candidate；按 66 个 sensor 分组"),
        ("Monash/electricity_hourly_dataset", "energy", "限制进入 candidate；先只问相对量/报告值，单位聚合语义待补"),
        ("Monash/temperature_rain_dataset_without_missing_values", "environment", "可进入 candidate；仅使用 observation series，排除含零填补的窗口"),
    ]
    lines = [
        "# 真实时序数据盘点（Real TSQA v1）", "",
        "> 生成命令：`python scripts/real_tsqa_inventory.py`。这是 protocol lock 前的审计，不是 benchmark 发布清单。", "",
        "## 审计口径", "",
        "本表区分 `REAL`、`DERIVED_REAL`、`SYNTHETIC` 和 `UNKNOWN`。目录名、任务标签或集合级许可证不能替代数据源级 provenance。`UNKNOWN` 不进入正式 benchmark；无法确认再分发条件的数据只保存 reconstruction manifest，不复制原始值。", "",
        f"机器清单共 {len(rows)} 行。现实属性计数：`{dict(realities)}`。审计状态计数：`{dict(statuses)}`。", "",
        "## 首批候选", "", "| 数据集 | 域 | 当前判断 |", "|---|---|---|",
    ]
    lines.extend(f"| {a} | {b} | {c} |" for a, b, c in selected)
    lines += [
        "", "## 明确排除或隔离", "",
        "- `PLAsTiCC 2018` 是合成挑战数据，不能充当真实世界主 benchmark。",
        "- UCR/UEA 的 158 个本地子集不能按集合名一律判为真实；逐数据集的许可、采样、单位和 group 大多缺失，当前全部隔离。",
        "- UniTS/WaveFormer 镜像是指向项目外 97.9 GB 数据树的符号链接；当前目录没有统一来源/许可清单，先逐条列为 `UNKNOWN`。",
        "- Gravity Spy 本地对象主要是时频表示而非直接 1-D 波形，且许可仍待核验，不进入首版。",
        "- Paderborn 官方已确认 CC BY-NC 4.0、bearing/工况结构以及两条主信号 64 kHz；本机解压工具不支持该 RAR 压缩方法，MAT 通道名与单位尚未本地复核，因此暂不进入 candidate。",
        "- TSB-UAD 的 `.out` 是派生格式。代码的 Apache-2.0 不能替代上游数据许可；详见 lineage 专页。",
        "", "## 已知重叠", "",
        "Autoformer CSV、Monash TSF、UCR 与 TSB-UAD 在 Timeseries-PILE 中是不同打包视图，部分来源重叠。正式 benchmark 只保留一个经核验的 canonical source，并按原始 group 去重。",
        "", "## 待完成后才能锁协议", "",
        "1. 澳洲 weather verification 与 Paderborn 许可已核验；Caltrans traffic 的来源级再分发条件仍待核验。",
        "2. 为候选源计算/保存最终文件哈希，并固定源版本。",
        "3. 先固定 group split，再构造窗口；不得从窗口随机切 split。",
        "4. 对 electricity_hourly 明确 15 分钟到小时的聚合算子及单位。",
        "5. TSB-UAD 若无法补齐逐文件 lineage，则只保留内部 exploratory，排除 native headline 与公开原始数据。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eligibility_doc(path: Path, rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r["dataset_name"], []).append(r)
    lines = [
        "# Dataset × Task Eligibility（Real TSQA v1）", "",
        "> 机器可读原表：`results/tsqa_evidence/real_v1/dataset_task_eligibility.csv`。只有 `eligibility=PASS` 的组合可以进入正式 manifest。", "",
        "| 数据集 | A | B | C | D | E | F | 关键限制 |", "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, items in grouped.items():
        values = {r["task_code"]: r["eligibility"] for r in items}
        lines.append(f"| {dataset} | {values['A']} | {values['B']} | {values['C']} | {values['D']} | {values['E']} | {values['F']} | {items[0]['reason']} |")
    lines += [
        "", "`PASS_RESTRICTED` 表示接口仅能问清单中已核验的可见事实；它不自动令整行正式通过。当前 candidate 仍需完成 source-level license、最终哈希和 family-level validity gates，之后才生成 lock。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("results/tsqa_evidence/real_v1"))
    args = parser.parse_args()
    rows = inventory_autoformer() + inventory_monash() + inventory_ucr() + inventory_tsb() + inventory_units() + inventory_raw()
    rows.sort(key=lambda r: str(r["dataset_name"]).lower())
    eligibility = make_eligibility()
    write_csv(args.output_root / "source_inventory.csv", rows, FIELDS)
    write_csv(args.output_root / "dataset_task_eligibility.csv", eligibility, list(eligibility[0]))
    write_inventory_doc(Path("docs/tsqa_evidence/10_real_data_inventory.md"), rows)
    write_eligibility_doc(Path("docs/tsqa_evidence/12_dataset_task_eligibility.md"), eligibility)
    print(json.dumps({"sources": len(rows), "eligibility_rows": len(eligibility), "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
