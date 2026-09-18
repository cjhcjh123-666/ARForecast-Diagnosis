# TSB-UAD `MBA_ECG801_data.out` lineage

Status: **LINEAGE_PARTIAL / excluded from confirmatory native QA and public benchmark data**

## Local object

- Path: `/public/chenjiahui/波数据时序基座大模型/data/Timeseries-PILE/anomaly_detection/TSB-UAD-Public/ECG/MBA_ECG801_data.out`
- Size: 1,666,773 bytes
- SHA-256: `82e82bd8f9e9c37ef9a080fd9b115c3c1480ad963ba2028bf5f4cba1db14af9f`
- Stored schema: two comma-separated columns. The first is a scalar signal value; the second is a binary point label.
- Stored timestamps, physical unit, channel name and sampling rate: absent.

## Confirmed facts

1. The local file is distributed in the `ECG` portion of the TSB-UAD-derived Time-series Pile copy.
2. The TSB-UAD paper describes its ECG anomalies as ventricular premature contractions and represents anomaly supervision pointwise.
3. The upstream MIT-BIH Arrhythmia Database contains 48 two-channel half-hour ECG records digitized at 360 Hz and expert beat annotations.
4. The official MIT-BIH release is under ODC-By 1.0. The TSB-UAD code repository is Apache-2.0. The code license does not set the license of a derived data file.
5. TSB-UAD also includes a long `MBA_ECG14046` object and 47 numbered segments of that object. That observation does not identify `MBA_ECG801` with a specific MIT-BIH record.

Sources:

- TSB-UAD repository: <https://github.com/TheDatumOrg/TSB-UAD>
- TSB-UAD paper: <https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf>
- MIT-BIH official release: <https://physionet.org/content/mitdb/1.0.0/>

## Unresolved mapping

The available local tree does not contain a manifest or preprocessing program that establishes:

- the exact upstream record behind `MBA_ECG801`;
- the upstream channel;
- whether 360 Hz samples were retained, resampled or filtered;
- any normalization and segmentation operations;
- the beat-symbol-to-point-label conversion rule;
- whether the local `.out` may itself be redistributed under the upstream license.

Searching local filenames, collection README files and the official TSB-UAD repository documentation did not yield this mapping. Similar length or filename digits are not accepted as proof.

## Allowed and prohibited uses

Allowed while status remains `LINEAGE_PARTIAL`:

- an internal exploratory supervised diagnostic, described as **TSB-UAD benchmark-labelled PVC anomaly**;
- sample-index questions whose gold comes directly from the stored point labels;
- numeric questions on the visible first column that do not claim anomaly understanding.

Prohibited:

- converting sample indices to seconds or stating that this derived file is 360 Hz;
- treating the point labels as a general clinical diagnosis;
- splitting `.out` windows as if they were independent patients or records;
- including it in confirmatory native-QA headlines;
- redistributing the `.out` values in a public benchmark package.

Supervision does not repair these provenance gaps. If no authoritative mapping appears before protocol lock, the status remains `LINEAGE_PARTIAL` and the dataset stays outside the formal native-QA matrix.
